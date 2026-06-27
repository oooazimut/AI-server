from __future__ import annotations

from typing import Any, Protocol

from ai_server.integrations.ports import WebhookEnqueuePort

__all__ = [
    "WebhookEnqueuePort",
    "ReconcilerPort",
    "TaskSupervisorPort",
]


class ReconcilerPort(Protocol):
    """One-shot reconciliation run triggered from an admin route."""

    async def __call__(self, *, status: dict[str, Any]) -> dict[str, Any]: ...


class TaskSupervisorPort(Protocol):
    """One-shot supervisor run triggered from an admin route."""

    async def __call__(self, *, status: dict[str, Any]) -> dict[str, Any]: ...
