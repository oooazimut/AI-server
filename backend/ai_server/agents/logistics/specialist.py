from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.logistics.llm import (
    LogisticsAgentLLM,
    LogisticsLLMService,
    LogisticsLLMToolCall,
    logistics_llm_failure_result,
)
from ai_server.agents.ports import SchedulerPort, SpecialistOutputPort, VehicleUsageToolsetPort
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentTask, ToolResult, ToolStatus, UserContext
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.vehicle_usage import SentRequestData
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)


@dataclass
class VehicleUsageSettings:
    dialog_id: str
    manager_user_id: int | None
    max_reminders: int
    reminder_interval_minutes: int
    admin_notify_user_ids: list[int] = field(default_factory=list)
    dry_run: bool = True
    request_time: str = "08:00"


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
        store: Any | None = None,
        output_fn: SpecialistOutputPort | None = None,
        vu_settings: VehicleUsageSettings | None = None,
    ) -> None:
        self._output_fn = output_fn
        self._vu_settings = vu_settings or VehicleUsageSettings(
            dialog_id="",
            manager_user_id=None,
            max_reminders=3,
            reminder_interval_minutes=60,
        )
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
        logistics_store: Any | None = None,
        scheduler: SchedulerPort | None = None,
        **_: Any,
    ) -> LogisticsSpecialist:
        return cls(
            manifest,
            retriever=logistics_retriever,
            tools=vehicle_usage_tools,
            llm=logistics_llm or LogisticsLLMService(),
            scheduler=scheduler,
            store=logistics_store,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._output_fn and self._vu_settings.dialog_id:
            self._setup_morning_cron()

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _setup_morning_cron(self) -> None:
        hour, minute = _parse_hhmm(self._vu_settings.request_time)
        self.schedule_job_cron(
            "morning_report",
            self._morning_handler,
            hour,
            minute,
            replace_existing=False,
        )
        logger.info("LogisticsSpecialist: morning_report cron scheduled at %s МСК", self._vu_settings.request_time)

    async def _morning_handler(self) -> None:
        await self._run_and_deliver(reminder_count=0)

    def _make_reminder_handler(self, count: int):  # noqa: ANN201
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
        vu = self._vu_settings
        dialog_id = vu.dialog_id
        manager_user_id = vu.manager_user_id

        if reminder_count >= vu.max_reminders:
            await self._escalate()
            return

        if not dialog_id:
            logger.warning("LogisticsSpecialist: dialog_id not set, skipping")
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

        if answer and self._output_fn:
            if not vu.dry_run:
                await self._output_fn(
                    AgentTask(
                        task_id=f"logistics_deliver_{uuid.uuid4().hex[:8]}",
                        request=answer,
                        user=UserContext(id=str(manager_user_id) if manager_user_id else ""),
                        context={
                            "_source": "logistics",
                            "_intent": "deliver_to_dialog",
                            "dialog_id": dialog_id,
                            "reminder_count": reminder_count,
                        },
                    )
                )
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
                logger.info("LogisticsSpecialist: dry_run, would deliver: %s", answer[:120])

        next_count = reminder_count + 1
        run_date = datetime.now(MOSCOW_TZ) + timedelta(minutes=vu.reminder_interval_minutes)
        self.schedule_job_at(
            f"reminder_{next_count}",
            self._make_reminder_handler(next_count),
            run_date,
        )
        logger.info("LogisticsSpecialist: scheduled reminder_%d at %s", next_count, run_date.isoformat())

    async def _escalate(self) -> None:
        vu = self._vu_settings
        admin_ids = vu.admin_notify_user_ids
        if not admin_ids:
            logger.warning("LogisticsSpecialist: admin_notify_user_ids not set, skipping escalation")
            return

        task = AgentTask(
            task_id=f"vehicle_usage_esc_{uuid.uuid4().hex[:8]}",
            request="Сформируй уведомление об отсутствии утреннего отчёта по служебным автомобилям.",
            context={"event": "vehicle_usage_escalation_due"},
        )
        message = await self._run_llm_task(task) or "Утренний отчёт по служебным автомобилям не получен."

        if vu.dry_run:
            logger.info("LogisticsSpecialist: dry_run, escalation would notify %s", admin_ids)
            return

        if self._output_fn:
            await self._output_fn(
                AgentTask(
                    task_id=f"logistics_esc_{uuid.uuid4().hex[:8]}",
                    request=message,
                    user=UserContext(id=""),
                    context={
                        "_source": "logistics",
                        "_intent": "escalate",
                        "admin_user_ids": admin_ids,
                    },
                )
            )

        if self.tools is not None:
            self.tools.store.mark_escalated(
                request_date=datetime.now(MOSCOW_TZ).date().isoformat(),
                user_id=vu.manager_user_id,
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

    def _llm_failure_result(self, message: str):  # noqa: ANN201
        return logistics_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Логист — LLM-специалист; инструменты только читают/пишут структурированное состояние.",
            "Доставка сообщений пользователю — зона Переговорщика и канального уровня.",
            "Bitrix остаётся слоем канала/источника; Логист владеет только интерпретацией vehicle_usage.",
        ]


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        logger.warning("LogisticsSpecialist: invalid request_time %r, defaulting to 08:00", value)
        return 8, 0
    return hour, minute
