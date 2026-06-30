from __future__ import annotations

import json
import re
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
        text = _json_object_text(self.content)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("LLM returned JSON, but root value is not an object")
        return parsed


class OpenAICompatibleLLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning: bool = False,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        s = settings or get_settings()
        self._settings = s
        self._model = model or s.llm_model
        self._base_url = base_url or s.llm_base_url
        self._api_key = api_key or s.llm_api_key
        self._reasoning = reasoning
        self._reasoning_effort = reasoning_effort
        self.timeout = httpx.Timeout(timeout_seconds)

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

        content = _extract_content(data, strip_thinking=self._reasoning)
        usage = _model_usage(agent_id=agent_id, data=data, settings=settings, model_default=self._model)
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
        url = _chat_completions_url(settings.llm_provider, self._base_url)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if settings.llm_temperature is not None:
            payload["temperature"] = settings.llm_temperature
        if json_mode and not self._reasoning:
            payload["response_format"] = {"type": "json_object"}
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort

        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
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


def build_orchestrator_llm_client(settings: Settings) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        settings,
        model=settings.orchestrator_llm_model or None,
        base_url=settings.orchestrator_llm_base_url or None,
        api_key=settings.orchestrator_llm_api_key or None,
        reasoning=settings.orchestrator_llm_reasoning,
        reasoning_effort=settings.orchestrator_llm_reasoning_effort or None,
        timeout_seconds=settings.orchestrator_llm_timeout_seconds,
    )


def _chat_completions_url(provider: str, base_url: str) -> str:
    normalized_provider = provider.strip().casefold()
    if base_url.strip():
        return base_url.rstrip("/") + "/chat/completions"
    if normalized_provider == "deepseek":
        return "https://api.deepseek.com/v1/chat/completions"
    raise LLMError("AI_SERVER_LLM_BASE_URL is required for this LLM provider")


def _extract_content(data: dict[str, Any], *, strip_thinking: bool = False) -> str:
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
    if strip_thinking:
        content = _strip_think_tags(content)
    return content


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _model_usage(
    *, agent_id: str, data: dict[str, Any], settings: Settings, model_default: str | None = None
) -> ModelUsageRecord:
    _raw_usage = data.get("usage")
    usage: dict[str, Any] = _raw_usage if isinstance(_raw_usage, dict) else {}
    return ModelUsageRecord(
        agent_id=agent_id,
        provider=settings.llm_provider,
        model=str(data.get("model") or model_default or settings.llm_model),
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


def _json_object_text(text: str) -> str:
    value = _strip_json_fence(text)
    if value.startswith("{"):
        return value
    extracted = _extract_first_json_object(value)
    return extracted or value


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


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
