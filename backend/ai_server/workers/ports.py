"""Ports for the workers layer."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol


class WebhookConsumePort(Protocol):
    """Consume side of the webhook event queue (used by worker loops)."""

    async def claim_next(
        self,
        *,
        blocked_partition_keys: Collection[str] | None = None,
    ) -> dict[str, Any] | None: ...

    async def mark_done(self, event_id: int, result: dict[str, Any]) -> None: ...
    async def mark_failed(self, event_id: int, error: str) -> None: ...
