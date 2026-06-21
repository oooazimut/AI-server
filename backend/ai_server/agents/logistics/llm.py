from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ai_server.agents.specialist_llm_shared import (
    DIALOG_HISTORY_PROMPT_FRAGMENT,
    allowed_tool_definitions,
    compact_tool_result,
    decision_status,
    load_instructions,
    result_status,
    retrieval_context,
)
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.utils import confidence

logger = logging.getLogger(__name__)

ALLOWED_TOOL_NAMES = {
    "vehicle_usage_context",
    "vehicle_usage_save_draft",
    "vehicle_usage_save_report",
    "none",
}


class LogisticsAgentLLM(Protocol):
    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        dialog_history: list[dict[str, str]] | None = None,
    ) -> LogisticsLLMDecisionResult:
        pass

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: LogisticsLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> LogisticsLLMFinalResult:
        pass


@dataclass(frozen=True)
class LogisticsLLMToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class LogisticsLLMDecision:
    status: str
    answer: str
    tool_calls: list[LogisticsLLMToolCall]
    confidence: float = 0.5


@dataclass(frozen=True)
class LogisticsLLMDecisionResult:
    decision: LogisticsLLMDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LogisticsLLMFinalResult:
    status: str
    answer: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class LogisticsLLMService:
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
    ) -> LogisticsLLMDecisionResult:
        instructions = load_instructions(manifest)
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _decision_system_prompt(instructions)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent": {
                                "id": manifest.id,
                                "name": manifest.name,
                                "handoff_description": manifest.handoff_description,
                            },
                            "request": task.request,
                            "user": task.user.model_dump(),
                            "context": task.context,
                            "current_datetime": datetime.now(UTC).astimezone().isoformat(),
                            "dialog_history": dialog_history or [],
                            "retrieval_context": retrieval_context(retrieval_hits),
                            "tools": allowed_tool_definitions(tool_definitions, ALLOWED_TOOL_NAMES),
                            "tool_results": [compact_tool_result(result) for result in (tool_results or [])],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return LogisticsLLMDecisionResult(
            decision=_parse_decision(completion.json_content()),
            model_usage=completion.model_usage,
            raw=completion.raw,
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: LogisticsLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> LogisticsLLMFinalResult:
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
                            "tool_results": [compact_tool_result(result) for result in tool_results],
                            "approval_actions": approval_actions or [],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        parsed = completion.json_content()
        return LogisticsLLMFinalResult(
            status=result_status(parsed.get("status")),
            answer=str(parsed.get("answer") or "").strip() or "Готово.",
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def logistics_llm_failure_result(message: str, agent_id: str = "logistics") -> LogisticsLLMFinalResult:
    return LogisticsLLMFinalResult(
        status="failed",
        answer=f"Не смог обработать запрос Логиста через LLM: {message}",
        model_usage=ModelUsageRecord(
            agent_id=agent_id,
            provider="",
            model="",
            status="error",
            notes=[message],
        ),
    )


def _decision_system_prompt(instructions: str = "") -> str:
    extra = f"\n\nДополнительные инструкции:\n{instructions}" if instructions else ""
    return (
        "Ты LLM-специалист Логист внутри корпоративного AI-server. "
        "Твоя зона: ежедневный учет служебных автомобилей, статусы сотрудников, смены, выезды, "
        "утренние отчеты и уточнения к ним. "
        "Ты не вызываешь Bitrix напрямую, не пишешь в чат сам и не пишешь в SQLite сам: выбирай vehicle_usage tools. "
        "Backend-tools только читают/пишут структурированные данные; "
        "они не решают, что имел в виду человек. "
        f"{DIALOG_HISTORY_PROMPT_FRAGMENT}"
        "Сначала получи vehicle_usage_context, если в tool_results еще нет roster/vehicles/latest_request. "
        "Сам распознавай естественный язык: кто работает, кто в отпуске/болеет/на объекте, какая машина за кем, "
        "является ли ответ подтверждением, исправлением или просьбой начать заново. "
        "Если данных не хватает, не сохраняй финальный отчет: сохрани черновик при необходимости и задай уточнение. "
        "Если задача пришла от scheduler и пора отправить утренний запрос или повторное напоминание, "
        "сформулируй точный текст сообщения в answer и не вызывай send tools. "
        "Если scheduler сообщает, что ответа нет к времени эскалации, сформулируй точный текст уведомления "
        "в answer и не вызывай notify tools. "
        "vehicle_usage_save_report вызывай только когда отчет явно подтвержден человеком или задача от scheduler "
        "содержит уже подтвержденный структурированный отчет. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"короткий предварительный ответ",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"vehicle_usage_context|vehicle_usage_save_draft|vehicle_usage_save_report|none","args":{},"summary":""}]}.'
        f"{extra}"
    )


def _compose_system_prompt() -> str:
    return (
        "Ты тот же Логист. Сформируй итоговый текст для Переговорщика по результатам vehicle_usage tools. "
        "Не выдумывай сохраненные записи. Если сохранен черновик, попроси проверить/подтвердить. "
        "Если сохранен финальный отчет, скажи кратко что сохранено. "
        "Если задача от scheduler про напоминание или эскалацию, answer должен быть точным текстом сообщения, "
        "которое Переговорщик отправит людям. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human|failed","answer":"ответ человеку"}.'
    )


def _parse_decision(data: dict[str, Any]) -> LogisticsLLMDecision:
    raw_tool_calls = data.get("tool_calls")
    tool_calls: list[LogisticsLLMToolCall] = []
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if name not in ALLOWED_TOOL_NAMES:
                continue
            args = raw_call.get("args") if isinstance(raw_call.get("args"), dict) else {}
            tool_calls.append(
                LogisticsLLMToolCall(
                    name=name,
                    args=args,
                    summary=str(raw_call.get("summary") or "").strip(),
                )
            )
    if not tool_calls:
        tool_calls = [LogisticsLLMToolCall(name="none")]
    return LogisticsLLMDecision(
        status=decision_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        confidence=confidence(data.get("confidence")),
        tool_calls=tool_calls,
    )


def _decision_dict(decision: LogisticsLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [{"name": call.name, "args": call.args, "summary": call.summary} for call in decision.tool_calls],
    }
