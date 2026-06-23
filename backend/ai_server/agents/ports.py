from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from ai_server.models import AgentResult, AgentTask


class SpecialistOutputPort(Protocol):
    """Outbound port for specialist-initiated tasks. All outgoing specialist communication goes through this."""

    async def __call__(self, task: AgentTask) -> AgentResult: ...


class AgentDialogStorePort(Protocol):
    """Per-specialist async dialog history store backed by PostgreSQL."""

    async def ensure_schema(self) -> None: ...

    async def load_turns(self, dialog_key: str, *, limit: int = 20) -> list[dict[str, str]]: ...

    async def append_turn(self, dialog_key: str, user_text: str, agent_response: str) -> None: ...


class SchedulerPort(Protocol):
    def add_job(self, agent_id: str, job_id: str, func: Any, trigger: Any, **kwargs: Any) -> Any: ...

    def add_job_at(self, agent_id: str, job_id: str, func: Any, run_date: datetime, **kwargs: Any) -> Any: ...

    def add_job_cron(self, agent_id: str, job_id: str, func: Any, hour: int, minute: int, **kwargs: Any) -> Any: ...

    def remove_jobs_by_prefix(self, agent_id: str, prefix: str) -> int: ...

    def list_jobs(self, agent_id: str) -> list[dict[str, Any]]: ...

    def schedule_task(
        self,
        agent_id: str,
        job_id: str,
        trigger_data: dict[str, Any],
        task_description: str,
        context: dict[str, Any] | None = None,
    ) -> Any: ...
