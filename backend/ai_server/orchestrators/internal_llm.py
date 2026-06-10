from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentResult, AgentTask, ModelUsageRecord
from ai_server.settings import Settings, get_settings
from ai_server.utils import confidence


class InternalOrchestratorLLM(Protocol):
    async def route(
        self,
        *,
        task: AgentTask,
        manifests: list[AgentManifest],
    ) -> InternalRouteResult:
        pass

    async def synthesize(
        self,
        *,
        task: AgentTask,
        specialist_results: list[tuple[str, AgentResult]],
    ) -> InternalSynthesisResult:
        pass


@dataclass(frozen=True)
class InternalRouteDecision:
    status: str
    answer: str
    handoff_to: list[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass(frozen=True)
class InternalRouteResult:
    decision: InternalRouteDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InternalSynthesisResult:
    answer: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class InternalLLMRouter:
    def __init__(self, client: LLMClient | None = None, settings: Settings | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()
        self._settings = settings or get_settings()

    async def route(
        self,
        *,
        task: AgentTask,
        manifests: list[AgentManifest],
    ) -> InternalRouteResult:
        completion = await self.client.complete(
            agent_id="internal_orchestrator",
            messages=[
                {"role": "system", "content": _system_prompt(manifests)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "user": task.user.model_dump(),
                            "dialog_history": task.context.get("dialog_history") or [],
                            "runtime_context": {
                                "llm_provider": self._settings.llm_provider,
                                "llm_model": self._settings.llm_model,
                                "llm_configured": self._settings.llm_configured,
                            },
                            "available_specialists": [
                                {
                                    "id": manifest.id,
                                    "name": manifest.name,
                                    "kind": manifest.kind,
                                    "handoff_description": manifest.handoff_description,
                                    "capabilities": manifest.capabilities,
                                }
                                for manifest in manifests
                                if manifest.kind == "specialist"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return InternalRouteResult(
            decision=_parse_decision(completion.json_content(), manifests),
            model_usage=completion.model_usage,
            raw=completion.raw,
        )

    async def synthesize(
        self,
        *,
        task: AgentTask,
        specialist_results: list[tuple[str, AgentResult]],
    ) -> InternalSynthesisResult:
        completion = await self.client.complete(
            agent_id="internal_orchestrator",
            messages=[
                {"role": "system", "content": _synthesis_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "specialist_answers": [
                                {"specialist": agent_id, "answer": sr.answer}
                                for agent_id, sr in specialist_results
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        raw = completion.json_content()
        return InternalSynthesisResult(
            answer=str(raw.get("answer") or "").strip(),
            model_usage=completion.model_usage,
            raw=raw,
        )


def _system_prompt(manifests: list[AgentManifest]) -> str:
    routing_hints = _specialist_routing_hints(manifests)
    return (
        "Ты LLM-оркестратор корпоративного AI-server. "
        "Ты не выполняешь бизнес-действия и не вызываешь инструменты. "
        "Твоя задача: понять запрос, выбрать одного или несколько доступных специалистов, "
        "или вернуть уточняющий/информационный ответ, если специалист не нужен или не найден. "
        "Если предоставлена dialog_history — учитывай контекст предыдущих сообщений для понимания текущего запроса. "
        "Никогда не притворяйся специалистом и не выполняй их работу сам. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|failed",'
        '"answer":"короткий ответ, если handoff_to пустой",'
        '"handoff_to":["specialist_id"],'
        '"confidence":0.0}. '
        f"{routing_hints}"
        "Если подходящего специалиста нет, handoff_to=[] и честно скажи, что специалист еще не подключен."
    )


def _specialist_routing_hints(manifests: list[AgentManifest]) -> str:
    hints = [
        f"{m.handoff_description} → выбери {m.id}."
        for m in manifests
        if m.kind == "specialist" and m.handoff_description
    ]
    return " ".join(hints) + " " if hints else ""


def _synthesis_prompt() -> str:
    return (
        "Ты LLM-оркестратор. Несколько специалистов выполнили задание и вернули ответы. "
        "Объедини их в единый связный ответ для пользователя. "
        "Не дублируй информацию. Не добавляй ничего от себя — только синтез ответов специалистов. "
        'Верни только JSON-объект без markdown: {"answer": "..."}.'
    )


def _parse_decision(data: dict[str, Any], manifests: list[AgentManifest]) -> InternalRouteDecision:
    known_specialists = {manifest.id for manifest in manifests if manifest.kind == "specialist"}
    raw_handoff = data.get("handoff_to")
    handoff_to = []
    if isinstance(raw_handoff, list):
        for item in raw_handoff:
            value = str(item or "").strip()
            if value in known_specialists and value not in handoff_to:
                handoff_to.append(value)
    return InternalRouteDecision(
        status=_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        handoff_to=handoff_to,
        confidence=confidence(data.get("confidence")),
    )


def _status(value: object) -> str:
    status = str(value or "completed").strip()
    return status if status in {"completed", "needs_clarification", "failed"} else "completed"


