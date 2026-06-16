from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ai_server.models import ModelUsageRecord
from ai_server.settings import Settings, get_settings
from ai_server.utils import optional_int


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    async def complete(
        self,
        *,
        agent_id: str,
        messages: list[dict[str, str]],
        json_mode: bool = False,
    ) -> LLMCompletion:
        pass


@dataclass(frozen=True)
class LLMCompletion:
    content: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any]

    def json_content(self) -> dict[str, Any]:
        text = _strip_json_fence(self.content)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("LLM returned JSON, but root value is not an object")
        return parsed


class OpenAICompatibleLLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.timeout = httpx.Timeout(60.0)

    async def complete(
        self,
        *,
        agent_id: str,
        messages: list[dict[str, str]],
        json_mode: bool = False,
    ) -> LLMCompletion:
        settings = self._settings
        if not settings.llm_configured:
            raise LLMError("LLM is not configured")

        max_tokens = settings.llm_max_tokens
        data = await self._request(agent_id=agent_id, messages=messages, json_mode=json_mode, max_tokens=max_tokens)
        if _is_length_truncated(data) and max_tokens > 0:
            data = await self._request(
                agent_id=agent_id,
                messages=messages,
                json_mode=json_mode,
                max_tokens=max_tokens * 2,
            )

        content = _extract_content(data)
        usage = _model_usage(agent_id=agent_id, data=data, settings=settings)
        return LLMCompletion(content=content, model_usage=usage, raw=data)

    async def _request(
        self,
        *,
        agent_id: str,
        messages: list[dict[str, str]],
        json_mode: bool,
        max_tokens: int,
    ) -> dict[str, Any]:
        settings = self._settings
        url = _chat_completions_url(settings.llm_provider, settings.llm_base_url)
        payload: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if settings.llm_temperature is not None:
            payload["temperature"] = settings.llm_temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
            )

        data = _response_json(response)
        if "error" in data:
            raise LLMError(_format_provider_error(data["error"]))
        if response.is_error:
            raise LLMError(f"LLM HTTP error {response.status_code}: {_response_text(response)}")
        return data


def build_llm_client(settings: Settings | None = None) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(settings or get_settings())


def _chat_completions_url(provider: str, base_url: str) -> str:
    normalized_provider = provider.strip().casefold()
    if base_url.strip():
        return base_url.rstrip("/") + "/chat/completions"
    if normalized_provider == "deepseek":
        return "https://api.deepseek.com/v1/chat/completions"
    raise LLMError("AI_SERVER_LLM_BASE_URL is required for this LLM provider")


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError("LLM response does not contain choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMError("LLM response choice is not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMError("LLM response choice does not contain message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LLMError("LLM response content is empty")
    return content


def _model_usage(*, agent_id: str, data: dict[str, Any], settings: Settings) -> ModelUsageRecord:
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return ModelUsageRecord(
        agent_id=agent_id,
        provider=settings.llm_provider,
        model=str(data.get("model") or settings.llm_model),
        status="used",
        input_tokens=optional_int(usage.get("prompt_tokens")),
        output_tokens=optional_int(usage.get("completion_tokens")),
    )


def _is_length_truncated(data: dict[str, Any]) -> bool:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    return isinstance(first, dict) and first.get("finish_reason") == "length"


def _strip_json_fence(text: str) -> str:
    value = text.strip()
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    return text[:500] + "..." if len(text) > 500 else text


def _format_provider_error(error: object) -> str:
    if isinstance(error, dict):
        message = error.get("message") or error.get("error") or error
        return f"LLM provider error: {message}"
    return f"LLM provider error: {error}"
