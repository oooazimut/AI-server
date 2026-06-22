from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ai_server.agents.ports import SchedulerPort
from ai_server.agents.specialist_llm_shared import (
    compact_tool_result,
    load_instructions,
    result_status,
    retrieval_context,
)
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import ActionRecord, AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.utils import confidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class ScheduledTaskDecision:
    agent_id: str
    job_id: str
    trigger: dict[str, Any]
    task_description: str


@dataclass(frozen=True)
class OrchestratorDecision:
    status: str
    answer: str
    tool_calls: list[OrchestratorToolCall] = field(default_factory=list)
    scheduled_tasks: list[ScheduledTaskDecision] = field(default_factory=list)
    confidence: float = 0.5


@dataclass(frozen=True)
class OrchestratorDecisionResult:
    decision: OrchestratorDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorFinalResult:
    answer: str
    status: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class OrchestratorLLM(Protocol):
    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        dialog_history: list[dict[str, str]],
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
    ) -> OrchestratorDecisionResult:
        pass

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: OrchestratorDecision,
        tool_results: list[ToolResult],
    ) -> OrchestratorFinalResult:
        pass


# ---------------------------------------------------------------------------
# Production service
# ---------------------------------------------------------------------------


class OrchestratorLLMService:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        dialog_history: list[dict[str, str]],
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
    ) -> OrchestratorDecisionResult:
        instructions = load_instructions(manifest)
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _decide_system_prompt(instructions)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent": {"id": manifest.id, "name": manifest.name},
                            "request": task.request,
                            "user": task.user.model_dump(),
                            "context": {k: v for k, v in task.context.items() if not k.startswith("_")},
                            "current_datetime": datetime.now(UTC).astimezone().isoformat(),
                            "dialog_history": dialog_history or [],
                            "retrieval_context": retrieval_context(retrieval_hits),
                            "tools": tool_definitions,
                            "tool_results": [compact_tool_result(r) for r in (tool_results or [])],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return OrchestratorDecisionResult(
            decision=_parse_decision(completion.json_content(), tool_definitions),
            model_usage=completion.model_usage,
            raw=completion.raw,
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: OrchestratorDecision,
        tool_results: list[ToolResult],
    ) -> OrchestratorFinalResult:
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _compose_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "initial_decision_answer": decision.answer,
                            "tool_results": [compact_tool_result(r) for r in tool_results],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        raw = completion.json_content()
        return OrchestratorFinalResult(
            answer=str(raw.get("answer") or "").strip() or decision.answer or "",
            status=result_status(raw.get("status")),
            model_usage=completion.model_usage,
            raw=raw,
        )


# ---------------------------------------------------------------------------
# Scheduler helper (used by InternalOrchestrator)
# ---------------------------------------------------------------------------


def apply_scheduled_tasks(
    tasks: list[ScheduledTaskDecision],
    scheduler: SchedulerPort | None,
) -> list[ActionRecord]:
    if not tasks or scheduler is None:
        return []
    actions: list[ActionRecord] = []
    for task in tasks:
        job_id = task.job_id or str(uuid.uuid4())[:8]
        try:
            job = scheduler.schedule_task(task.agent_id, job_id, task.trigger, task.task_description)
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            actions.append(
                ActionRecord(
                    name="orchestrator_schedule_task",
                    status="scheduled",
                    details={"agent_id": task.agent_id, "job_id": f"{task.agent_id}:{job_id}", "next_run": next_run},
                )
            )
            logger.info("Orchestrator scheduled task %s:%s next=%s", task.agent_id, job_id, next_run)
        except Exception as exc:
            logger.exception("Orchestrator failed to schedule task for %s", task.agent_id)
            actions.append(
                ActionRecord(
                    name="orchestrator_schedule_task",
                    status="error",
                    details={"agent_id": task.agent_id, "job_id": job_id, "error": f"{type(exc).__name__}: {exc}"},
                )
            )
    return actions


def orchestrator_llm_failure_result(message: str) -> OrchestratorFinalResult:
    return OrchestratorFinalResult(
        answer=f"Не смог обработать запрос через Переговорщика: {message}",
        status="failed",
        model_usage=ModelUsageRecord(
            agent_id="internal_orchestrator",
            provider="",
            model="",
            status="error",
            notes=[message],
        ),
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _decide_system_prompt(instructions: str = "") -> str:
    extra = f"\n\n{instructions}" if instructions else ""
    return (
        "Ты Переговорщик — старший AI-агент корпоративного AI-server. "
        "Ты посредник между людьми и специалистами-субагентами. "
        'Твоя задача: понять запрос, выбрать нужный инструмент (call_<id>) или ответить самому (tool_calls=[{"name":"none"}]). '
        "Если предыдущие tool_results уже содержат нужные данные — не вызывай те же инструменты снова. "
        'Если ни один инструмент не нужен — дай ответ в answer и верни tool_calls=[{"name":"none"}]. '
        "Никогда не притворяйся специалистом и не выполняй их работу сам. "
        "Если task.context содержит _source — задачу инициировал специалист. "
        "Читай context._intent и принимай решение какой инструмент вызвать: "
        "  _intent=deliver_to_dialog → call_bitrix24 для отправки в context.dialog_id; "
        "  _intent=escalate → call_bitrix24 для уведомления context.admin_user_ids. "
        "Верни только JSON-объект без markdown. Формат: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"предварительный ответ",'
        '"tool_calls":[{"name":"call_<id>|none","args":{"request":"задача для специалиста"},"summary":""}],'
        '"scheduled_tasks":[],'
        '"confidence":0.0}.'
        f"{extra}"
    )


def _compose_system_prompt() -> str:
    return (
        "Ты Переговорщик — составляешь финальный ответ пользователю на основе результатов специалистов. "
        "Если tool_results содержат ответы специалистов — объедини их в единый связный ответ. "
        "Если tool_results пусты или содержат только ошибки — используй initial_decision_answer. "
        "Не дублируй информацию. Пиши от первого лица. Не раскрывай внутренние tool calls и имена специалистов. "
        'Верни только JSON-объект без markdown: {"answer":"...","status":"completed|needs_clarification|failed"}.'
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_decision(data: dict[str, Any], tool_definitions: list[dict[str, Any]]) -> OrchestratorDecision:
    known_tools = {td["name"] for td in tool_definitions} | {"none"}
    raw_calls = data.get("tool_calls")
    tool_calls: list[OrchestratorToolCall] = []
    if isinstance(raw_calls, list):
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name not in known_tools:
                continue
            tool_calls.append(
                OrchestratorToolCall(
                    name=name,
                    args=item.get("args") or {},
                    summary=str(item.get("summary") or ""),
                )
            )
    if not tool_calls:
        tool_calls = [OrchestratorToolCall(name="none")]
    scheduled_tasks = _parse_scheduled_tasks(data.get("scheduled_tasks"), known_tools)
    return OrchestratorDecision(
        status=_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        tool_calls=tool_calls,
        scheduled_tasks=scheduled_tasks,
        confidence=confidence(data.get("confidence")),
    )


def _parse_scheduled_tasks(raw: object, known_tools: set[str]) -> list[ScheduledTaskDecision]:
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        agent_id = str(item.get("agent_id") or "").strip()
        trigger = item.get("trigger")
        task_description = str(item.get("task_description") or "").strip()
        if not agent_id or not isinstance(trigger, dict) or not task_description:
            continue
        job_id = str(item.get("job_id") or "").strip() or agent_id
        result.append(
            ScheduledTaskDecision(
                agent_id=agent_id,
                job_id=job_id,
                trigger=trigger,
                task_description=task_description,
            )
        )
    return result


def _status(value: object) -> str:
    s = str(value or "completed").strip()
    return s if s in {"completed", "needs_clarification", "failed"} else "completed"
