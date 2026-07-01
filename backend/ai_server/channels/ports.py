from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from ai_server.integrations.ports import WebhookEnqueuePort

if TYPE_CHECKING:
    from ai_server.models import AgentTask

__all__ = [
    "WebhookEnqueuePort",
    "ReconcilerPort",
    "TaskSupervisorPort",
    "FeedbackReceiverPort",
]


class ReconcilerPort(Protocol):
    """One-shot reconciliation run triggered from an admin route."""

    async def __call__(self, *, status: dict[str, Any]) -> dict[str, Any]: ...


class TaskSupervisorPort(Protocol):
    """One-shot supervisor run triggered from an admin route."""

    async def __call__(self, *, status: dict[str, Any]) -> dict[str, Any]: ...


class FeedbackReceiverPort(Protocol):
    """Intercept feedback messages at the channel layer before routing to orchestrator.

    Returns True  → message handled as feedback; do NOT route to orchestrator.
    Returns False → regular request; route normally.
    """

    async def handle(self, task: AgentTask) -> bool: ...
