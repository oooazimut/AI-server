from __future__ import annotations

from typing import Any, Protocol


class SearchWebhookHandlerPort(Protocol):
    """Processes a Bitrix disk webhook event against the portal search index."""

    async def handle(self, payload: dict[str, Any], *, status: dict[str, Any]) -> dict[str, Any]: ...


class QualityControlHandlerPort(Protocol):
    """Handles a Bitrix task webhook event for quality-control review."""

    async def handle(self, payload: dict[str, Any], *, status: dict[str, Any]) -> dict[str, Any]: ...
