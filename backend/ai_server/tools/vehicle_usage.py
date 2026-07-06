from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

from ai_server.utils import MOSCOW_TZ


class VehicleUsageStorePort(Protocol):
    """Persistent store for vehicle usage logistics data."""

    def upsert_employees(self, members: list[Any]) -> None: ...
    def context(self, *, request_date: str, user_id: int | None, dialog_id: str) -> dict[str, Any]: ...
    def staff_roster(self) -> list[dict[str, Any]]: ...
    def vehicles(self) -> list[dict[str, Any]]: ...
    def latest_request(self, *, user_id: int | None, dialog_id: str) -> dict[str, Any] | None: ...
    def get_request(self, *, request_date: str, user_id: int | None) -> dict[str, Any] | None: ...
    def get_day_report(self, *, report_date: str) -> dict[str, Any]: ...
    def get_employee_period_report(
        self, *, employee_name: str, date_from: str, date_to: str
    ) -> dict[str, Any]: ...
    def get_vehicle_period_report(
        self, *, vehicle_name: str, date_from: str, date_to: str
    ) -> dict[str, Any]: ...
    def vehicle_usage_operator_ids(self) -> set[int]: ...
    def set_vehicle_usage_operators(self, *, operator_user_ids: list[int], actor_user_id: int | None) -> list[int]: ...
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
        vehicle_assignments: list[tuple[int, int | None, str] | tuple[int, int | None, str, str]],
        actor_user_id: int | None = None,
    ) -> None: ...
    def update_day_report(
        self,
        *,
        report_date: str,
        people: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
        actor_user_id: int | None = None,
        change_summary: str = "",
    ) -> dict[str, Any]: ...
    def cancel_day_report(
        self,
        *,
        report_date: str,
        user_id: int | None,
        dialog_id: str,
        reason: str,
    ) -> int: ...


def resolve_vehicle_usage_operator_ids(
    store: VehicleUsageStorePort | None,
    fallback_user_ids: set[int] | frozenset[int],
) -> list[int]:
    getter = getattr(store, "vehicle_usage_operator_ids", None)
    if callable(getter):
        ids = {int(user_id) for user_id in getter() if int(user_id) > 0}
        if ids:
            return sorted(ids)
    return sorted({int(user_id) for user_id in fallback_user_ids if int(user_id) > 0})


@dataclass(frozen=True)
class SentRequestData:
    request_date: str
    user_id: int | None
    dialog_id: str
    message: str
    sent_at: str
    reminder_count: int


@dataclass(frozen=True)
class StaffMember:
    order: int
    name: str
    user_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"display_order": self.order, "full_name": self.name, "user_id": self.user_id}


def _request_date(value: object) -> str:
    raw = str(value or "").strip()
    if raw:
        try:
            return date.fromisoformat(raw[:10]).isoformat()
        except ValueError:
            pass
    return datetime.now(MOSCOW_TZ).date().isoformat()


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)
