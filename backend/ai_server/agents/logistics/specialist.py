from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from ai_server.agent_store import AgentStore
from ai_server.agents.base import BaseSpecialist
from ai_server.agents.logistics.llm import (
    LogisticsAgentLLM,
    LogisticsLLMService,
    LogisticsLLMToolCall,
    logistics_llm_failure_result,
)
from ai_server.agents.ports import SchedulerPort, VehicleUsageToolsetPort
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentTask, ToolResult, ToolStatus, UserContext
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import Settings
from ai_server.skills import SkillStore
from ai_server.tools.vehicle_usage import SentRequestData
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)


class LogisticsSpecialist(BaseSpecialist):
    max_steps = 5
    action_prefix = "logistics"

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        tools: VehicleUsageToolsetPort | None = None,
        llm: LogisticsAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        store: AgentStore | None = None,
        deliver_fn: Callable[[str, str], Awaitable[None]] | None = None,
        notify_fn: Callable[[int, str], Awaitable[None]] | None = None,
        settings: Settings,
    ) -> None:
        self._settings = settings
        self._deliver_fn = deliver_fn
        self._notify_fn = notify_fn
        super().__init__(
            manifest,
            knowledge_base=knowledge_base,
            skill_store=skill_store,
            retriever=retriever,
            tools=tools,
            llm=llm,
            scheduler=scheduler,
            store=store,
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        vehicle_usage_tools: VehicleUsageToolsetPort | None = None,
        logistics_retriever: HybridKnowledgeRetriever | None = None,
        logistics_llm: LogisticsAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        settings: Settings | None = None,
        **_: Any,
    ) -> LogisticsSpecialist:
        from ai_server.settings import get_settings

        return cls(
            manifest,
            retriever=logistics_retriever,
            tools=vehicle_usage_tools,
            llm=logistics_llm or LogisticsLLMService(),
            scheduler=scheduler,
            settings=settings or get_settings(),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._deliver_fn and self._settings.vehicle_usage_enabled:
            self._setup_morning_cron()

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _setup_morning_cron(self) -> None:
        hour, minute = _parse_hhmm(self._settings.vehicle_usage_request_time)
        self.schedule_job_cron(
            "morning_report",
            self._morning_handler,
            hour,
            minute,
            replace_existing=False,
        )
        logger.info(
            "LogisticsSpecialist: morning_report cron scheduled at %s МСК", self._settings.vehicle_usage_request_time
        )

    async def _morning_handler(self) -> None:
        await self._run_and_deliver(reminder_count=0)

    def _make_reminder_handler(self, count: int) -> Callable[[], Awaitable[None]]:
        async def _handler() -> None:
            await self._run_and_deliver(reminder_count=count)

        return _handler

    async def _run_llm_task(self, task: AgentTask) -> str | None:
        try:
            result = await self.handle(task)
            return result.answer
        except Exception:
            logger.exception("LogisticsSpecialist: LLM task failed")
            return None

    async def _run_and_deliver(self, reminder_count: int) -> None:
        settings = self._settings
        dialog_id = settings.vehicle_usage_dialog_id
        manager_user_id = settings.vehicle_usage_manager_user_id
        max_reminders = settings.vehicle_usage_max_reminders

        if reminder_count >= max_reminders:
            await self._escalate()
            return

        if not dialog_id:
            logger.warning("LogisticsSpecialist: VEHICLE_USAGE_DIALOG_ID not set, skipping")
            return

        task = AgentTask(
            task_id=f"vehicle_usage_{uuid.uuid4().hex[:8]}",
            request="Нужен утренний отчёт по использованию служебных автомобилей.",
            user=UserContext(id=str(manager_user_id) if manager_user_id else ""),
            context={
                "event": "vehicle_usage_morning_due",
                "dialog_id": dialog_id,
                "manager_user_id": manager_user_id,
                "reminder_count": reminder_count,
            },
        )

        answer = await self._run_llm_task(task)
        if answer is None:
            return

        if answer and self._deliver_fn:
            if not settings.vehicle_usage_dry_run:
                await self._deliver_fn(dialog_id, answer)
                if self.tools is not None:
                    self.tools.store.create_sent_request(
                        SentRequestData(
                            request_date=datetime.now(MOSCOW_TZ).date().isoformat(),
                            user_id=manager_user_id,
                            dialog_id=dialog_id,
                            message=answer,
                            sent_at=datetime.now(MOSCOW_TZ).isoformat(),
                            reminder_count=reminder_count + 1,
                        )
                    )
            else:
                logger.info("LogisticsSpecialist: dry_run, would send: %s", answer[:120])

        next_count = reminder_count + 1
        interval = settings.vehicle_usage_reminder_interval_minutes
        run_date = datetime.now(MOSCOW_TZ) + timedelta(minutes=interval)
        self.schedule_job_at(
            f"reminder_{next_count}",
            self._make_reminder_handler(next_count),
            run_date,
        )
        logger.info("LogisticsSpecialist: scheduled reminder_%d at %s", next_count, run_date.isoformat())

    async def _escalate(self) -> None:
        settings = self._settings
        admin_ids = settings.resolved_vehicle_usage_admin_notify_user_ids
        if not admin_ids:
            logger.warning("LogisticsSpecialist: VEHICLE_USAGE_ADMIN_NOTIFY_USER_IDS not set, skipping escalation")
            return

        task = AgentTask(
            task_id=f"vehicle_usage_esc_{uuid.uuid4().hex[:8]}",
            request="Сформируй уведомление об отсутствии утреннего отчёта по служебным автомобилям.",
            context={"event": "vehicle_usage_escalation_due"},
        )
        message = await self._run_llm_task(task) or "Утренний отчёт по служебным автомобилям не получен."

        if settings.vehicle_usage_dry_run:
            logger.info("LogisticsSpecialist: dry_run, escalation would notify %s", admin_ids)
            return

        for user_id in admin_ids:
            if self._notify_fn:
                try:
                    await self._notify_fn(user_id, message)
                except Exception:
                    logger.exception("LogisticsSpecialist: escalation notify failed for user_id=%d", user_id)

        if self.tools is not None:
            self.tools.store.mark_escalated(
                request_date=datetime.now(MOSCOW_TZ).date().isoformat(),
                user_id=settings.vehicle_usage_manager_user_id,
                escalated_at=datetime.now(MOSCOW_TZ).isoformat(),
            )
        logger.info("LogisticsSpecialist: escalation sent to %s", admin_ids)

    # ------------------------------------------------------------------
    # BaseSpecialist hooks
    # ------------------------------------------------------------------

    def tool_definitions(self) -> list[dict]:
        if self.tools is None:
            return []
        return [definition.model_dump() for definition in self.tools.definitions()]

    async def _execute_tool_call(
        self,
        tool_call: LogisticsLLMToolCall,
        task: AgentTask,
    ) -> tuple[ToolResult | None, ActionRecord | None, list[ActionRecord]]:
        tools: VehicleUsageToolsetPort | None = task.context.get("_vehicle_tools") or self.tools
        if tool_call.name == "none":
            return None, None, []
        if tools is None:
            result = ToolResult(
                status=ToolStatus.ERROR,
                tool=tool_call.name,
                error="Logistics toolset not configured",
            )
            return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump()), []
        if tool_call.name == "vehicle_usage_context":
            result = tools.vehicle_usage_context(tool_call.args)
            return (
                result,
                ActionRecord(name="logistics_vehicle_usage_context", status=result.status, details=result.model_dump()),
                [],
            )
        if tool_call.name == "vehicle_usage_save_draft":
            result = tools.vehicle_usage_save_draft(tool_call.args)
            return (
                result,
                ActionRecord(
                    name="logistics_vehicle_usage_save_draft", status=result.status, details=result.model_dump()
                ),
                [],
            )
        if tool_call.name == "vehicle_usage_save_report":
            result = tools.vehicle_usage_save_report(tool_call.args)
            if result.status == "ok":
                cancelled = self.cancel_jobs_by_prefix("reminder_")
                if cancelled:
                    logger.info("LogisticsSpecialist: cancelled %d reminder jobs after report saved", cancelled)
            return (
                result,
                ActionRecord(
                    name="logistics_vehicle_usage_save_report", status=result.status, details=result.model_dump()
                ),
                [],
            )

        result = ToolResult(
            status=ToolStatus.INVALID_TOOL_CALL,
            tool=tool_call.name,
            error=f"unknown Logistics tool call: {tool_call.name}",
        )
        return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump()), []

    def _llm_failure_result(self, message: str):
        return logistics_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Logistics specialist is an LLM specialist; vehicle tools only read/write structured state.",
            "User-facing delivery belongs to the Negotiator/channel runtime.",
            "Bitrix remains the channel/source layer; Logistics owns vehicle usage interpretation.",
        ]


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        logger.warning("LogisticsSpecialist: invalid vehicle_usage_request_time %r, defaulting to 08:00", value)
        return 8, 0
    return hour, minute
