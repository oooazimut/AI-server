from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ai_server.agents.specialist_llm_shared import (
    DIALOG_HISTORY_PROMPT_FRAGMENT,
    SKILLS_PROMPT_FRAGMENT,
    allowed_tool_definitions,
    compact_tool_result,
    decision_status,
    result_status,
    skills_context,
)
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.utils import confidence

logger = logging.getLogger(__name__)

ALLOWED_TOOL_NAMES = {
    "diagnost_search_events",
    "diagnost_get_incident",
    "diagnost_list_incidents",
    "diagnost_create_incident",
    "diagnost_error_report",
    "none",
}


class DiagnostAgentLLM(Protocol):
    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        dialog_history: list[dict[str, str]] | None = None,
        available_skills: list | None = None,
    ) -> DiagnostLLMDecisionResult:
        pass

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: DiagnostLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> DiagnostLLMFinalResult:
        pass


@dataclass(frozen=True)
class DiagnostLLMToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class DiagnostLLMDecision:
    status: str
    answer: str
    tool_calls: list[DiagnostLLMToolCall]
    confidence: float = 0.5


@dataclass(frozen=True)
class DiagnostLLMDecisionResult:
    decision: DiagnostLLMDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnostLLMFinalResult:
    status: str
    answer: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class DiagnostLLMService:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        dialog_history: list[dict[str, str]] | None = None,
        available_skills: list | None = None,
    ) -> DiagnostLLMDecisionResult:
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _decision_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent": {"id": manifest.id, "name": manifest.name},
                            "request": task.request,
                            "user": task.user.model_dump() if task.user else {},
                            "current_datetime": datetime.now(UTC).astimezone().isoformat(),
                            "available_skills": skills_context(available_skills or []),
                            "dialog_history": dialog_history or [],
                            "tools": allowed_tool_definitions(tool_definitions, ALLOWED_TOOL_NAMES),
                            "tool_results": [compact_tool_result(r) for r in (tool_results or [])],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return DiagnostLLMDecisionResult(
            decision=_parse_decision(completion.json_content()),
            model_usage=completion.model_usage,
            raw=completion.raw,
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: DiagnostLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> DiagnostLLMFinalResult:
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _compose_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "initial_decision": _decision_dict(decision),
                            "tool_results": [compact_tool_result(r) for r in tool_results],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        parsed = completion.json_content()
        return DiagnostLLMFinalResult(
            status=result_status(parsed.get("status")),
            answer=str(parsed.get("answer") or "").strip() or "Готово.",
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def diagnost_llm_failure_result(message: str, agent_id: str = "diagnost") -> DiagnostLLMFinalResult:
    return DiagnostLLMFinalResult(
        status="failed",
        answer=f"Не смог обработать запрос ИИ-Диагноста: {message}",
        model_usage=ModelUsageRecord(
            agent_id=agent_id,
            provider="",
            model="",
            status="error",
            notes=[message],
        ),
    )


def _decision_system_prompt() -> str:
    return (
        "Ты ИИ-Диагност корпоративного AI-server. "
        "Анализируешь качество работы системы: находишь инциденты, ищешь паттерны ошибок, "
        "формируешь отчёты. Не работаешь с Bitrix, задачами или бизнес-данными. "
        "Используй инструменты для получения данных из хранилища диагностики. "
        f"{SKILLS_PROMPT_FRAGMENT}"
        f"{DIALOG_HISTORY_PROMPT_FRAGMENT}"
        "Верни только JSON без markdown: "
        '{"status":"completed|needs_clarification","answer":"предварительный ответ",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"diagnost_search_events|diagnost_get_incident|diagnost_list_incidents|diagnost_create_incident|diagnost_error_report|none","args":{},"summary":""}]}.'
    )


def _compose_system_prompt() -> str:
    return (
        "Ты ИИ-Диагност. Сформируй итоговый ответ по данным из инструментов диагностики. "
        "Опирайся только на данные из tool_results, не выдумывай. "
        "Верни только JSON без markdown: "
        '{"status":"completed|needs_clarification|failed","answer":"ответ человеку"}.'
    )


def _parse_decision(data: dict[str, Any]) -> DiagnostLLMDecision:
    raw_calls = data.get("tool_calls")
    tool_calls: list[DiagnostLLMToolCall] = []
    if isinstance(raw_calls, list):
        for raw in raw_calls:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            if name not in ALLOWED_TOOL_NAMES:
                continue
            args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
            tool_calls.append(DiagnostLLMToolCall(name=name, args=args, summary=str(raw.get("summary") or "")))
    if not tool_calls:
        tool_calls = [DiagnostLLMToolCall(name="none")]
    return DiagnostLLMDecision(
        status=decision_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        confidence=confidence(data.get("confidence")),
        tool_calls=tool_calls,
    )


def _decision_dict(decision: DiagnostLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [{"name": c.name, "args": c.args, "summary": c.summary} for c in decision.tool_calls],
    }
