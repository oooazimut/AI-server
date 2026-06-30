"""Infrastructure-layer ports for persistent stores.

Defined here (inner layer) so that tools/ can import without violating
the dependency rule tools → integrations. PostgreSQL/SQLite implementations
satisfy these protocols via structural typing — no explicit import needed.
"""

from __future__ import annotations

from typing import Any, Protocol


class WebhookEnqueuePort(Protocol):
    """Enqueue side of the webhook event queue.

    Defined here (integrations/) so that both channels/ and workers/ can import
    without cross-layer violations: channels → integrations ✓, workers → integrations ✓.
    """

    async def enqueue(
        self,
        payload: dict[str, Any],
        *,
        event_type: str,
        dedupe_key: str | None = None,
    ) -> tuple[int, bool]: ...

    async def stats(self) -> dict[str, Any]: ...

    async def latest(self, *, limit: int = 20) -> list[dict[str, Any]]: ...
