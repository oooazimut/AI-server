from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ai_server.models import AgentResult, AgentTask


class AgentQueuePort(Protocol):
    """Per-agent event queue: publish messages, claim next message, ack/nack."""

    async def publish(self, message: dict[str, Any]) -> None: ...

    async def claim_next(
        self,
        agent_id: str,
        *,
        blocked_partition_keys: set[str] | None = None,
    ) -> dict[str, Any] | None: ...

    async def ack(self, message_id: str) -> None: ...

    async def nack(self, message_id: str, *, error: str) -> None: ...

    async def active_partition_keys(self, agent_id: str) -> set[str]: ...


class ChannelPort(Protocol):
    """Outbound port for delivering messages to a communication channel."""

    channel_id: str

    async def send(self, recipient_id: str, body: str) -> None: ...


class AgentStorePort(Protocol):
    """Per-specialist async dialog history store."""

    async def ensure_schema(self) -> None: ...

    async def load_turns(self, dialog_key: str, *, limit: int = 20) -> list[dict[str, str]]: ...

    async def append_turn(self, dialog_key: str, user_text: str, agent_response: str) -> None: ...


class OrchestratorStorePort(AgentStorePort, Protocol):
    """AgentStorePort + KV-state for orchestrator (pending/suspended specialists)."""

    async def get_kv(self, dialog_key: str, field: str) -> str | None: ...

    async def set_kv(self, dialog_key: str, field: str, value: str) -> None: ...

    async def delete_kv(self, dialog_key: str, field: str) -> None: ...


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


class ResultPublisherPort(Protocol):
    """Output port: publish orchestrator result for downstream consumers (e.g. DiagnostWorker)."""

    async def publish(self, task: AgentTask, result: AgentResult) -> None: ...
