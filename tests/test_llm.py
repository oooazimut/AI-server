import asyncio
import json

import httpx
import pytest

from ai_server.llm import LLMError, OpenAICompatibleLLMClient
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


def _response(*, finish_reason: str, content: str) -> dict:
    return {
        "model": "test-model",
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
