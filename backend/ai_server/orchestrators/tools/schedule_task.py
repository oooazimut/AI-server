from __future__ import annotations

import logging
import uuid
from typing import Any

from ai_server.agents.ports import SchedulerPort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class ScheduleTaskTool:
    """Schedule a future task for a specialist via the orchestrator scheduler."""

    name = "schedule_task"

    def __init__(self, scheduler: SchedulerPort | None = None) -> None:
        self._scheduler = scheduler

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Запланировать задачу для агента-специалиста на будущее время.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string", "description": "ID агента-специалиста"},
                    "job_id": {"type": "string", "description": "Уникальный ID задания (опционально)"},
                    "trigger": {
                        "type": "object",
                        "description": 'Описание триггера, например {"type": "date", "run_date": "2026-06-28T09:00:00+03:00"}',
                    },
                    "task_description": {"type": "string", "description": "Описание задачи для агента"},
                },
                "required": ["agent_id", "trigger", "task_description"],
            },
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if self._scheduler is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="Scheduler не настроен",
            )

        agent_id = str(args.get("agent_id") or "").strip()
        trigger = args.get("trigger")
        task_description = str(args.get("task_description") or "").strip()

        if not agent_id or not isinstance(trigger, dict) or not task_description:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="agent_id, trigger (object) и task_description обязательны",
            )

        job_id = str(args.get("job_id") or "").strip() or str(uuid.uuid4())[:8]

        try:
            job = self._scheduler.schedule_task(agent_id, job_id, trigger, task_description)
            next_run = job.next_run_time.isoformat() if job and job.next_run_time else None
        except Exception as exc:
            logger.exception("ScheduleTaskTool: failed to schedule %s:%s", agent_id, job_id)
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=f"{type(exc).__name__}: {exc}",
            )

        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"agent_id": agent_id, "job_id": f"{agent_id}:{job_id}", "next_run": next_run},
        )
