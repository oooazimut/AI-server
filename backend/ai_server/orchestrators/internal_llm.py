from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord
from ai_server.settings import get_settings


class InternalOrchestratorLLM(Protocol):
    async def route(
        self,
        *,
        task: AgentTask,
        manifests: list[AgentManifest],
    ) -> "InternalRouteResult":
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


class InternalLLMRouter:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def route(
        self,
        *,
        task: AgentTask,
        manifests: list[AgentManifest],
    ) -> InternalRouteResult:
        settings = get_settings()
        completion = await self.client.complete(
            agent_id="internal_orchestrator",
            messages=[
                {"role": "system", "content": _system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "user": task.user.model_dump(),
                            "runtime_context": {
                                "llm_provider": settings.llm_provider,
                                "llm_model": settings.llm_model,
                                "llm_configured": settings.llm_configured,
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


def _system_prompt() -> str:
    return (
        "Ты LLM-оркестратор корпоративного AI-server. "
        "Ты не выполняешь бизнес-действия и не вызываешь инструменты. "
        "Твоя задача: понять запрос, выбрать одного или несколько доступных специалистов, "
        "или вернуть уточняющий/информационный ответ, если специалист не нужен или не найден. "
        "Никогда не притворяйся Bitrix/ПТО/сетевым специалистом и не выполняй их работу сам. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|failed",'
        '"answer":"короткий ответ, если handoff_to пустой",'
        '"handoff_to":["bitrix24"],'
        '"confidence":0.0}. '
        "Если запрос относится к Bitrix24, задачам, проектам, CRM, документам портала или заявкам, "
        "выбери bitrix24. Если подходящего специалиста нет, handoff_to=[] и честно скажи, что специалист еще не подключен."
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
        confidence=_confidence(data.get("confidence")),
    )


def _status(value: object) -> str:
    status = str(value or "completed").strip()
    return status if status in {"completed", "needs_clarification", "failed"} else "completed"


def _confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(number, 0.0), 1.0)
