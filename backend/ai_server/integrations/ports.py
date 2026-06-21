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


class VehicleUsageStorePort(Protocol):
    """Persistent store for vehicle usage logistics data."""

    def upsert_employees(self, members: list[Any]) -> None: ...
    def context(self, *, request_date: str, user_id: int | None, dialog_id: str) -> dict[str, Any]: ...
    def staff_roster(self) -> list[dict[str, Any]]: ...
    def latest_request(self, *, user_id: int | None, dialog_id: str) -> dict[str, Any] | None: ...
    def get_request(self, *, request_date: str, user_id: int | None) -> dict[str, Any] | None: ...
    def latest_requests(self, *, limit: int) -> list[dict[str, Any]]: ...
    def create_sent_request(self, data: Any) -> int: ...
    def mark_escalated(self, *, request_date: str, user_id: int | None, escalated_at: str) -> bool: ...
    def save_draft(
        self,
        *,
        request_date: str,
        user_id: int | None,
        dialog_id: str,
        response_text: str,
        parsed: dict[str, Any],
        status: str,
    ) -> int: ...
    def replace_day_report(
        self,
        *,
        status_date: str,
        employee_statuses: list[tuple[int, str, str]],
        vehicle_assignments: list[tuple[int, int | None, str]],
    ) -> None: ...
