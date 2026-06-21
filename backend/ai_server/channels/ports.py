from __future__ import annotations

from typing import Any, Protocol

from ai_server.integrations.ports import WebhookEnqueuePort

__all__ = [
    "SearchWebhookHandlerPort",
    "QualityControlHandlerPort",
    "WebhookEnqueuePort",
    "ReconcilerPort",
    "TaskSupervisorPort",
]


class SearchWebhookHandlerPort(Protocol):
    """Processes a Bitrix disk webhook event against the portal search index."""

    async def handle(self, payload: dict[str, Any], *, status: dict[str, Any]) -> dict[str, Any]: ...


class QualityControlHandlerPort(Protocol):
    """Handles a Bitrix task webhook event for quality-control review."""

    async def handle(self, payload: dict[str, Any], *, status: dict[str, Any]) -> dict[str, Any]: ...


class ReconcilerPort(Protocol):
    """One-shot reconciliation run triggered from an admin route."""

    async def __call__(self, *, status: dict[str, Any]) -> dict[str, Any]: ...


class TaskSupervisorPort(Protocol):
    """One-shot supervisor run triggered from an admin route."""

    async def __call__(self, *, status: dict[str, Any]) -> dict[str, Any]: ...
