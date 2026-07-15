import asyncio
import json

import httpx
import pytest

from ai_server.llm import LLMError, OpenAICompatibleLLMClient, _strip_think_tags, build_orchestrator_llm_client
from ai_server.settings import get_settings

_RealAsyncClient = httpx.AsyncClient


def _settings(monkeypatch, **overrides):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AI_SERVER_LLM_MODEL", "test-model")
    monkeypatch.setenv("AI_SERVER_LLM_API_KEY", "test-key")
    monkeypatch.setenv("AI_SERVER_LLM_BASE_URL", "https://llm.example.test/v1")
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    return get_settings()


def _response(*, finish_reason: str, content: str, model: str = "test-model") -> dict:
    return {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_complete_retries_with_doubled_max_tokens_on_truncation(monkeypatch):
    settings = _settings(monkeypatch, AI_SERVER_LLM_MAX_TOKENS="100")
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(200, json=_response(finish_reason="length", content='{"answer": "cut off'))
        return httpx.Response(200, json=_response(finish_reason="stop", content='{"answer": "ok"}'))

    monkeypatch.setattr(
        "ai_server.llm.httpx.AsyncClient",
        lambda *args, **kwargs: _RealAsyncClient(transport=httpx.MockTransport(handler)),
    )

    client = OpenAICompatibleLLMClient(settings)
    completion = asyncio.run(
        client.complete(agent_id="test", messages=[{"role": "user", "content": "hi"}], json_mode=True)
    )

    assert len(requests) == 2
    assert requests[0]["max_tokens"] == 100
    assert requests[1]["max_tokens"] == 200
    assert completion.json_content() == {"answer": "ok"}


def test_complete_does_not_retry_when_not_truncated(monkeypatch):
    settings = _settings(monkeypatch, AI_SERVER_LLM_MAX_TOKENS="100")
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_response(finish_reason="stop", content='{"answer": "ok"}'))

    monkeypatch.setattr(
        "ai_server.llm.httpx.AsyncClient",
        lambda *args, **kwargs: _RealAsyncClient(transport=httpx.MockTransport(handler)),
    )

    client = OpenAICompatibleLLMClient(settings)
    completion = asyncio.run(
        client.complete(agent_id="test", messages=[{"role": "user", "content": "hi"}], json_mode=True)
    )

    assert len(requests) == 1
    assert completion.json_content() == {"answer": "ok"}


def test_complete_raises_on_provider_error(monkeypatch):
    settings = _settings(monkeypatch, AI_SERVER_LLM_MAX_TOKENS="100")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"message": "boom"}})

    monkeypatch.setattr(
        "ai_server.llm.httpx.AsyncClient",
        lambda *args, **kwargs: _RealAsyncClient(transport=httpx.MockTransport(handler)),
    )

    client = OpenAICompatibleLLMClient(settings)
    with pytest.raises(LLMError, match="boom"):
        asyncio.run(client.complete(agent_id="test", messages=[{"role": "user", "content": "hi"}]))


# ---------------------------------------------------------------------------
# Reasoning mode
# ---------------------------------------------------------------------------


def test_strip_think_tags_removes_think_block():
    assert _strip_think_tags("<think>internal</think>answer") == "answer"


def test_strip_think_tags_handles_multiline():
    raw = '<think>\nreasoning line 1\nreasoning line 2\n</think>\n{"answer": "ok"}'
    assert _strip_think_tags(raw) == '{"answer": "ok"}'


def test_strip_think_tags_noop_when_no_tags():
    assert _strip_think_tags('{"answer": "ok"}') == '{"answer": "ok"}'


def test_reasoning_mode_skips_json_format_header(monkeypatch):
    settings = _settings(monkeypatch)
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_response(finish_reason="stop", content='{"answer": "ok"}'))

    monkeypatch.setattr(
        "ai_server.llm.httpx.AsyncClient",
        lambda *args, **kwargs: _RealAsyncClient(transport=httpx.MockTransport(handler)),
    )

    client = OpenAICompatibleLLMClient(settings, reasoning=True)
    asyncio.run(client.complete(agent_id="test", messages=[{"role": "user", "content": "hi"}], json_mode=True))

    assert "response_format" not in captured[0]


def test_reasoning_mode_strips_think_from_content(monkeypatch):
    settings = _settings(monkeypatch)
    think_response = _response(
        finish_reason="stop",
        content='<think>internal reasoning</think>{"answer": "clean"}',
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=think_response)

    monkeypatch.setattr(
        "ai_server.llm.httpx.AsyncClient",
        lambda *args, **kwargs: _RealAsyncClient(transport=httpx.MockTransport(handler)),
    )

    client = OpenAICompatibleLLMClient(settings, reasoning=True)
    completion = asyncio.run(client.complete(agent_id="test", messages=[{"role": "user", "content": "hi"}]))

    assert completion.content == '{"answer": "clean"}'
    assert completion.json_content() == {"answer": "clean"}


def test_build_orchestrator_llm_client_uses_orchestrator_settings(monkeypatch):
    settings = _settings(
        monkeypatch,
        AI_SERVER_ORCHESTRATOR_LLM_MODEL="deepseek-reasoner",
        AI_SERVER_ORCHESTRATOR_LLM_REASONING="true",
        AI_SERVER_ORCHESTRATOR_LLM_TIMEOUT_SECONDS="120",
    )

    client = build_orchestrator_llm_client(settings)

    assert client._model == "deepseek-reasoner"
    assert client._reasoning is True


def test_build_orchestrator_llm_client_falls_back_to_main_settings(monkeypatch):
    settings = _settings(monkeypatch)

    client = build_orchestrator_llm_client(settings)

    assert client._model == settings.llm_model
    assert client._reasoning is False


def test_build_orchestrator_llm_client_uses_flash_model_when_routing_enabled(monkeypatch):
    settings = _settings(
        monkeypatch,
        AI_SERVER_LLM_ROUTING_ENABLED="true",
        AI_SERVER_LLM_FLASH_MODEL="deepseek-v4-flash",
        AI_SERVER_LLM_PRO_MODEL="deepseek-v4-pro",
    )

    client = build_orchestrator_llm_client(settings)

    assert client._model == "deepseek-v4-flash"
    assert client._fallback_model == "deepseek-v4-pro"
    assert client._reasoning is False


def test_complete_falls_back_to_pro_model_when_flash_returns_invalid_json(monkeypatch):
    settings = _settings(monkeypatch, AI_SERVER_LLM_MAX_TOKENS="100")
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if payload["model"] == "flash-model":
            return httpx.Response(200, json=_response(finish_reason="stop", content="not json", model="flash-model"))
        return httpx.Response(200, json=_response(finish_reason="stop", content='{"answer": "ok"}', model="pro-model"))

    monkeypatch.setattr(
        "ai_server.llm.httpx.AsyncClient",
        lambda *args, **kwargs: _RealAsyncClient(transport=httpx.MockTransport(handler)),
    )

    client = OpenAICompatibleLLMClient(settings, model="flash-model", fallback_model="pro-model")
    completion = asyncio.run(
        client.complete(agent_id="test", messages=[{"role": "user", "content": "hi"}], json_mode=True)
    )

    assert [request["model"] for request in requests] == ["flash-model", "pro-model"]
    assert completion.json_content() == {"answer": "ok"}
    assert completion.model_usage.model == "pro-model"
