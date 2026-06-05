from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit


ALLOWED_TOOL_NAMES = {
    "vehicle_usage_context",
    "vehicle_usage_save_draft",
    "vehicle_usage_save_report",
    "vehicle_usage_mark_request_sent",
    "vehicle_usage_notify_admins",
    "vehicle_usage_send_message",
    "none",
}
RESULT_STATUSES = {"completed", "needs_clarification", "needs_human", "failed"}


class LogisticsAgentLLM(Protocol):
    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
    ) -> "LogisticsLLMDecisionResult":
        pass

    async def compose(
        self,
        *,
        task: AgentTask,
        decision: "LogisticsLLMDecision",
        tool_results: list[ToolResult],
    ) -> "LogisticsLLMFinalResult":
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
    ) -> LogisticsLLMDecisionResult:
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _decision_system_prompt()},
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
                            "current_datetime": datetime.now(timezone.utc).astimezone().isoformat(),
                            "retrieval_context": _retrieval_context(retrieval_hits),
                            "tools": _allowed_tool_definitions(tool_definitions),
                            "tool_results": [_compact_tool_result(result) for result in (tool_results or [])],
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
        task: AgentTask,
        decision: LogisticsLLMDecision,
        tool_results: list[ToolResult],
    ) -> LogisticsLLMFinalResult:
        completion = await self.client.complete(
            agent_id="logistics",
            messages=[
                {"role": "system", "content": _compose_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "initial_decision": _decision_dict(decision),
                            "tool_results": [_compact_tool_result(result) for result in tool_results],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        parsed = completion.json_content()
        return LogisticsLLMFinalResult(
            status=_result_status(parsed.get("status")),
            answer=str(parsed.get("answer") or "").strip() or "Готово.",
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def logistics_llm_failure_result(message: str) -> LogisticsLLMFinalResult:
    return LogisticsLLMFinalResult(
        status="failed",
        answer=f"Не смог обработать запрос Логиста через LLM: {message}",
        model_usage=ModelUsageRecord(
            agent_id="logistics",
            provider="",
            model="",
            status="error",
            notes=[message],
        ),
    )


def _decision_system_prompt() -> str:
    return (
        "Ты LLM-специалист Логист внутри корпоративного AI-server. "
        "Твоя зона: ежедневный учет служебных автомобилей, статусы сотрудников, смены, выезды, "
        "утренние отчеты и уточнения к ним. "
        "Ты не вызываешь Bitrix напрямую и не пишешь в SQLite сам: выбирай vehicle_usage tools. "
        "Backend-tools только читают/пишут структурированные данные и отправляют сообщения; "
        "они не решают, что имел в виду человек. "
        "Сначала получи vehicle_usage_context, если в tool_results еще нет roster/vehicles/latest_request. "
        "Сам распознавай естественный язык: кто работает, кто в отпуске/болеет/на объекте, какая машина за кем, "
        "является ли ответ подтверждением, исправлением или просьбой начать заново. "
        "Если данных не хватает, не сохраняй финальный отчет: сохрани черновик при необходимости и задай уточнение. "
        "Если задача пришла от scheduler и пора отправить утренний запрос или повторное напоминание, "
        "сформулируй сообщение, вызови vehicle_usage_send_message, затем vehicle_usage_mark_request_sent. "
        "Если scheduler сообщает, что ответа нет к времени эскалации, сформулируй уведомление и вызови "
        "vehicle_usage_notify_admins. "
        "vehicle_usage_save_report вызывай только когда отчет явно подтвержден человеком или задача от scheduler "
        "содержит уже подтвержденный структурированный отчет. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"короткий предварительный ответ",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"vehicle_usage_context|vehicle_usage_save_draft|vehicle_usage_save_report|vehicle_usage_mark_request_sent|vehicle_usage_notify_admins|vehicle_usage_send_message|none","args":{},"summary":""}]}.'
    )


def _compose_system_prompt() -> str:
    return (
        "Ты тот же Логист. Сформируй итоговый ответ человеку по результатам vehicle_usage tools. "
        "Не выдумывай сохраненные записи. Если сохранен черновик, попроси проверить/подтвердить. "
        "Если сохранен финальный отчет, скажи кратко что сохранено. "
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
        status=_decision_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        confidence=_confidence(data.get("confidence")),
        tool_calls=tool_calls,
    )


def _decision_status(value: object) -> str:
    status = str(value or "completed").strip()
    return status if status in {"completed", "needs_clarification", "needs_human"} else "completed"


def _result_status(value: object) -> str:
    status = str(value or "completed").strip()
    return status if status in RESULT_STATUSES else "completed"


def _confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(number, 0.0), 1.0)


def _retrieval_context(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    return [
        {
            "topic": hit.chunk.topic,
            "section": hit.chunk.section,
            "score": hit.score,
            "text": hit.chunk.text[:1200],
        }
        for hit in hits[:5]
    ]


def _allowed_tool_definitions(definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [definition for definition in definitions if definition.get("name") in ALLOWED_TOOL_NAMES]


def _decision_dict(decision: LogisticsLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [
            {"name": call.name, "args": call.args, "summary": call.summary}
            for call in decision.tool_calls
        ],
    }


def _compact_tool_result(result: ToolResult) -> dict[str, Any]:
    return {"status": result.status, "tool": result.tool, "data": result.data, "error": result.error}
