from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from ai_server.integrations.ports import VehicleUsageStorePort
from ai_server.tools.bitrix_ports import BitrixUserPort
from ai_server.utils import MOSCOW_TZ, optional_int

logger = logging.getLogger(__name__)


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


class VehicleReportProcessor:
    def __init__(self, repo: VehicleUsageStorePort) -> None:
        self._repo = repo

    def save_report(
        self,
        *,
        request_date: str,
        user_id: int | None,
        dialog_id: str,
        source_text: str,
        parsed: dict[str, Any],
    ) -> dict[str, Any]:
        request_id = self._repo.save_draft(
            request_date=request_date,
            user_id=user_id,
            dialog_id=dialog_id,
            response_text=source_text,
            parsed=parsed,
            status="answered",
        )
        staff_entries = _staff_entries(parsed)
        vehicle_entries = _vehicle_entries(parsed)
        employees_by_name = {
            str(row["full_name"]).casefold(): int(row["display_order"]) for row in self._repo.staff_roster()
        }

        employee_statuses: list[tuple[int, str, str]] = []
        for entry in staff_entries:
            employee_id = optional_int(entry.get("staff_order")) or employees_by_name.get(
                str(entry.get("full_name") or "").casefold()
            )
            if employee_id is None:
                continue
            employee_statuses.append(
                (employee_id, str(entry.get("status") or "unknown"), str(entry.get("notes") or ""))
            )

        vehicle_assignments: list[tuple[int, int | None, str]] = []
        for entry in vehicle_entries:
            vehicle_id = optional_int(entry.get("vehicle_id"))
            if vehicle_id is None:
                continue
            resolved_employee_id = optional_int(entry.get("employee_id")) or employees_by_name.get(
                str(entry.get("employee_name") or "").casefold()
            )
            vehicle_assignments.append((vehicle_id, resolved_employee_id, str(entry.get("notes") or "")))

        self._repo.replace_day_report(
            status_date=request_date,
            employee_statuses=employee_statuses,
            vehicle_assignments=vehicle_assignments,
        )
        return {
            "request_id": request_id,
            "staff_entries_saved": len(employee_statuses),
            "vehicle_assignments_saved": len(vehicle_assignments),
        }


async def fetch_staff_roster(
    client: BitrixUserPort,
    *,
    exclude_user_ids: set[int] | None = None,
) -> list[StaffMember]:
    users = await client.list_all_users(
        filter_={"ACTIVE": True, "USER_TYPE": "employee"},
        select=["ID", "NAME", "LAST_NAME", "WORK_POSITION"],
    )
    excluded = exclude_user_ids or set()
    candidates: list[tuple[str, int, str]] = []
    for user in users:
        user_id = optional_int(user.get("ID") or user.get("id"))
        if user_id is None or user_id in excluded:
            continue
        first = str(user.get("NAME") or "").strip()
        last = str(user.get("LAST_NAME") or "").strip()
        name = f"{last} {first}".strip() if last else first
        if not name:
            continue
        candidates.append((last.casefold() or first.casefold(), user_id, name))
    candidates.sort(key=lambda x: x[0])
    return [StaffMember(order=i + 1, user_id=uid, name=name) for i, (_, uid, name) in enumerate(candidates)]


def _staff_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    people = parsed.get("people")
    return [item for item in people if isinstance(item, dict)] if isinstance(people, list) else []


def _vehicle_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    vehicles = parsed.get("vehicles")
    return [item for item in vehicles if isinstance(item, dict)] if isinstance(vehicles, list) else []


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
