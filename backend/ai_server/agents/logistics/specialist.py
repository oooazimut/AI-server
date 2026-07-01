from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.logistics.llm import (
    LogisticsAgentLLM,
    LogisticsLLMService,
    logistics_llm_failure_result,
)
from ai_server.agents.logistics.tools import VehicleContextTool, VehicleSaveDraftTool, VehicleSaveReportTool
from ai_server.agents.ports import SchedulerPort
from ai_server.agents.tool import AgentTool
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentManifest, AgentResult, AgentTask, ScheduledTask
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.vehicle_usage import VehicleUsageStorePort
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)


@dataclass
class VehicleUsageSettings:
    manager_user_id: int | None
    max_reminders: int
    reminder_interval_minutes: int
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
        agent_tools: list[AgentTool] | None = None,
        llm: LogisticsAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        store: Any | None = None,
        vu_settings: VehicleUsageSettings | None = None,
        result_publisher: Any | None = None,
    ) -> None:
        self._vu_settings = vu_settings
        super().__init__(
            manifest,
            knowledge_base=knowledge_base,
            skill_store=skill_store,
            retriever=retriever,
            agent_tools=agent_tools,
            llm=llm,
            scheduler=scheduler,
            store=store,
            result_publisher=result_publisher,
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        vehicle_usage_store: VehicleUsageStorePort | None = None,
        logistics_retriever: HybridKnowledgeRetriever | None = None,
        logistics_llm: LogisticsAgentLLM | None = None,
        logistics_store: Any | None = None,
        scheduler: SchedulerPort | None = None,
        logistics_vu_settings: VehicleUsageSettings | None = None,
        specialist_result_publisher: Any | None = None,
        **_: Any,
    ) -> LogisticsSpecialist:
        tools: list[AgentTool] = [
            VehicleContextTool(vehicle_usage_store),
            VehicleSaveDraftTool(vehicle_usage_store),
            VehicleSaveReportTool(vehicle_usage_store),
        ]
        return cls(
            manifest,
            retriever=logistics_retriever,
            agent_tools=tools,
            llm=logistics_llm or LogisticsLLMService(),
            scheduler=scheduler,
            store=logistics_store,
            vu_settings=logistics_vu_settings,
            result_publisher=specialist_result_publisher,
        )

    # ------------------------------------------------------------------
    # Scheduling: declare reminders via AgentResult.scheduled_tasks
    # ------------------------------------------------------------------

    async def handle(self, task: AgentTask) -> AgentResult:
        result = await super().handle(task)
        scheduled = self._build_scheduled_tasks(task, result)
        return result.model_copy(update={"scheduled_tasks": scheduled}) if scheduled else result

    def _build_scheduled_tasks(self, task: AgentTask, result: AgentResult) -> list[ScheduledTask]:
        vu = self._vu_settings
        if vu is None:
            return []

        today = datetime.now(MOSCOW_TZ).date().isoformat()
        request_date = str(task.context.get("request_date") or today)
        job_id = f"vu_reminder_{request_date}"

        # Cancel reminder when report is saved
        if any(a.name == "vehicle_usage_save_report" for a in result.actions_taken):
            return [ScheduledTask(job_id=job_id, agent_id="logistics", cancel=True)]

        # Schedule follow-up reminder after initial morning request or previous reminder
        event = str(task.context.get("event") or "")
        if event in ("vehicle_usage_morning", "vehicle_usage_reminder_due") and result.answer:
            reminder_count = int(task.context.get("reminder_count") or 0) + 1
            if reminder_count > vu.max_reminders:
                return []
            run_date = datetime.now(MOSCOW_TZ) + timedelta(minutes=vu.reminder_interval_minutes)
            channel_id = str(task.context.get("channel_id") or "")
            recipient_id = str(task.context.get("recipient_id") or "")
            reminder_task = AgentTask(
                task_id=f"vu_reminder_{uuid4().hex[:6]}",
                request=task.request,
                context={
                    "channel_id": channel_id,
                    "recipient_id": recipient_id,
                    "event": "vehicle_usage_reminder_due",
                    "request_date": request_date,
                    "reminder_count": reminder_count,
                },
            )
            return [
                ScheduledTask(
                    job_id=job_id,
                    agent_id="logistics",
                    trigger={"type": "date", "run_date": run_date.isoformat()},
                    task=reminder_task,
                    description=f"Vehicle usage reminder #{reminder_count}",
                )
            ]

        return []

    # ------------------------------------------------------------------
    # BaseSpecialist hooks
    # ------------------------------------------------------------------

    def _llm_failure_result(self, message: str):  # noqa: ANN201
        return logistics_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Логист — LLM-специалист; инструменты только читают/пишут структурированное состояние.",
            "Доставка сообщений пользователю — зона Переговорщика и канального уровня.",
            "Bitrix остаётся слоем канала/источника; Логист владеет только интерпретацией vehicle_usage.",
        ]
