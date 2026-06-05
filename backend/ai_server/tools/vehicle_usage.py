from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.models import ToolDefinition, ToolResult
from ai_server.settings import get_settings


MOSCOW_TZ = timezone(timedelta(hours=3))


@dataclass(frozen=True)
class ServiceVehicle:
    id: int
    name: str
    plate: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "plate": self.plate}


@dataclass(frozen=True)
class StaffMember:
    order: int
    name: str
    user_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"display_order": self.order, "full_name": self.name, "user_id": self.user_id}


SERVICE_VEHICLES: tuple[ServiceVehicle, ...] = (
    ServiceVehicle(1, "Лада Ларгус", "В490СР161"),
    ServiceVehicle(2, "Лада Ларгус", "У316ТО161"),
    ServiceVehicle(3, "Лада Ларгус", "В735ХА161"),
    ServiceVehicle(4, "Nissan Almera Classic", "М845КН761"),
    ServiceVehicle(5, "Nissan Almera Classic", "М017УН61"),
    ServiceVehicle(6, "Renault Logan", "О248АМ761"),
)


class VehicleUsageStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().vehicle_usage_db_path

    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_usage_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_date TEXT NOT NULL,
                    user_id INTEGER,
                    dialog_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    sent_at TEXT,
                    response_text TEXT,
                    responded_at TEXT,
                    parsed_json TEXT,
                    UNIQUE(request_date, user_id)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS employees (
                    id INTEGER PRIMARY KEY,
                    bitrix_user_id INTEGER,
                    full_name TEXT NOT NULL,
                    position TEXT NOT NULL DEFAULT ''
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY,
                    brand_model TEXT NOT NULL,
                    registration_number TEXT NOT NULL,
                    debit_card_number TEXT NOT NULL DEFAULT '',
                    ppr_card_number TEXT NOT NULL DEFAULT ''
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS employee_daily_statuses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status_date TEXT NOT NULL,
                    employee_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    UNIQUE(status_date, employee_id)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS vehicle_daily_assignments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assignment_date TEXT NOT NULL,
                    vehicle_id INTEGER NOT NULL,
                    employee_id INTEGER,
                    notes TEXT NOT NULL DEFAULT '',
                    UNIQUE(assignment_date, vehicle_id)
                )
                """
            )

    def bootstrap_reference_data(self) -> None:
        self.ensure_schema()
        roster = _staff_roster_from_settings()
        with self._connection() as db:
            for member in roster:
                db.execute(
                    """
                    INSERT INTO employees (id, bitrix_user_id, full_name, position)
                    VALUES (?, ?, ?, '')
                    ON CONFLICT(id) DO UPDATE SET
                        bitrix_user_id = excluded.bitrix_user_id,
                        full_name = excluded.full_name
                    """,
                    (member.order, member.user_id, member.name),
                )
            for vehicle in SERVICE_VEHICLES:
                db.execute(
                    """
                    INSERT INTO vehicles (id, brand_model, registration_number)
                    VALUES (?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        brand_model = excluded.brand_model,
                        registration_number = excluded.registration_number
                    """,
                    (vehicle.id, vehicle.name, vehicle.plate),
                )

    def context(self, *, request_date: str, user_id: int | None, dialog_id: str) -> dict[str, Any]:
        self.bootstrap_reference_data()
        return {
            "request_date": request_date,
            "staff_roster": self.staff_roster(),
            "vehicles": [vehicle.as_dict() for vehicle in SERVICE_VEHICLES],
            "latest_request": self.latest_request(user_id=user_id, dialog_id=dialog_id),
        }

    def staff_roster(self) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT id AS display_order, bitrix_user_id AS user_id, full_name, position
                FROM employees
                ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_request(self, *, user_id: int | None, dialog_id: str) -> dict[str, Any] | None:
        self.ensure_schema()
        if dialog_id:
            with self._connection() as db:
                row = db.execute(
                    """
                    SELECT *
                    FROM vehicle_usage_requests
                    WHERE dialog_id = ?
                    ORDER BY request_date DESC, id DESC
                    LIMIT 1
                    """,
                    (dialog_id,),
                ).fetchone()
            return _request_row_dict(row)
        if user_id is None:
            return None
        with self._connection() as db:
            row = db.execute(
                """
                SELECT *
                FROM vehicle_usage_requests
                WHERE user_id = ?
                ORDER BY request_date DESC, id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return _request_row_dict(row)

    def save_draft(
        self,
        *,
        request_date: str,
        user_id: int | None,
        dialog_id: str,
        response_text: str,
        parsed: dict[str, Any],
        status: str = "pending_confirmation",
    ) -> int:
        self.bootstrap_reference_data()
        now = _now().isoformat()
        with self._connection() as db:
            db.execute(
                """
                INSERT INTO vehicle_usage_requests (
                    request_date, user_id, dialog_id, status, response_text, responded_at, parsed_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_date, user_id) DO UPDATE SET
                    dialog_id = excluded.dialog_id,
                    status = excluded.status,
                    response_text = excluded.response_text,
                    responded_at = excluded.responded_at,
                    parsed_json = excluded.parsed_json
                """,
                (
                    request_date,
                    user_id,
                    dialog_id,
                    status,
                    response_text,
                    now,
                    json.dumps(parsed, ensure_ascii=False, sort_keys=True),
                ),
            )
            row = db.execute(
                """
                SELECT id
                FROM vehicle_usage_requests
                WHERE request_date = ? AND (user_id = ? OR (? IS NULL AND user_id IS NULL))
                """,
                (request_date, user_id, user_id),
            ).fetchone()
        return int(row["id"]) if row else 0

    def save_report(
        self,
        *,
        request_date: str,
        user_id: int | None,
        dialog_id: str,
        source_text: str,
        parsed: dict[str, Any],
    ) -> dict[str, Any]:
        request_id = self.save_draft(
            request_date=request_date,
            user_id=user_id,
            dialog_id=dialog_id,
            response_text=source_text,
            parsed=parsed,
            status="answered",
        )
        staff_entries = _staff_entries(parsed)
        vehicle_entries = _vehicle_entries(parsed)
        employees_by_name = {str(row["full_name"]).casefold(): int(row["display_order"]) for row in self.staff_roster()}
        with self._connection() as db:
            db.execute("DELETE FROM employee_daily_statuses WHERE status_date = ?", (request_date,))
            db.execute("DELETE FROM vehicle_daily_assignments WHERE assignment_date = ?", (request_date,))
            for entry in staff_entries:
                employee_id = _optional_int(entry.get("staff_order")) or employees_by_name.get(
                    str(entry.get("full_name") or "").casefold()
                )
                if employee_id is None:
                    continue
                db.execute(
                    """
                    INSERT INTO employee_daily_statuses (status_date, employee_id, status, notes)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(status_date, employee_id) DO UPDATE SET
                        status = excluded.status,
                        notes = excluded.notes
                    """,
                    (
                        request_date,
                        employee_id,
                        str(entry.get("status") or "unknown"),
                        str(entry.get("notes") or ""),
                    ),
                )
            for entry in vehicle_entries:
                vehicle_id = _optional_int(entry.get("vehicle_id"))
                if vehicle_id is None:
                    continue
                employee_id = _optional_int(entry.get("employee_id")) or employees_by_name.get(
                    str(entry.get("employee_name") or "").casefold()
                )
                db.execute(
                    """
                    INSERT INTO vehicle_daily_assignments (assignment_date, vehicle_id, employee_id, notes)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(assignment_date, vehicle_id) DO UPDATE SET
                        employee_id = excluded.employee_id,
                        notes = excluded.notes
                    """,
                    (request_date, vehicle_id, employee_id, str(entry.get("notes") or "")),
                )
        return {
            "request_id": request_id,
            "staff_entries_saved": len(staff_entries),
            "vehicle_assignments_saved": len(vehicle_entries),
        }

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()


class VehicleUsageToolset:
    def __init__(
        self,
        client: BitrixClient | None = None,
        *,
        store: VehicleUsageStore | None = None,
        user_id: int | None = None,
        dialog_id: str = "",
    ) -> None:
        self.client = client or BitrixClient()
        self.store = store or VehicleUsageStore()
        self.user_id = user_id
        self.dialog_id = dialog_id

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="vehicle_usage_context",
                description="Read staff roster, known vehicles and latest vehicle usage draft/request.",
                parameters={
                    "type": "object",
                    "properties": {"request_date": {"type": "string"}},
                },
            ),
            ToolDefinition(
                name="vehicle_usage_save_draft",
                description="Save the logistics LLM parsed draft; does not finalize daily report.",
                parameters={
                    "type": "object",
                    "properties": {
                        "request_date": {"type": "string"},
                        "response_text": {"type": "string"},
                        "parsed": {"type": "object"},
                        "status": {"type": "string"},
                    },
                    "required": ["parsed"],
                },
            ),
            ToolDefinition(
                name="vehicle_usage_save_report",
                description="Finalize confirmed daily vehicle/staff report using the JSON chosen by Logistics LLM.",
                parameters={
                    "type": "object",
                    "properties": {
                        "request_date": {"type": "string"},
                        "source_text": {"type": "string"},
                        "parsed": {"type": "object"},
                    },
                    "required": ["parsed"],
                },
            ),
            ToolDefinition(
                name="vehicle_usage_send_message",
                description="Send a logistics reminder or clarification message to a Bitrix dialog.",
                parameters={
                    "type": "object",
                    "properties": {
                        "dialog_id": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["dialog_id", "message"],
                },
            ),
        ]

    def vehicle_usage_context(self, args: dict[str, Any]) -> ToolResult:
        request_date = _request_date(args.get("request_date"))
        return ToolResult(
            status="ok",
            tool="vehicle_usage_context",
            data=self.store.context(request_date=request_date, user_id=self.user_id, dialog_id=self.dialog_id),
        )

    def vehicle_usage_save_draft(self, args: dict[str, Any]) -> ToolResult:
        parsed = args.get("parsed")
        if not isinstance(parsed, dict):
            return ToolResult(status="invalid_tool_call", tool="vehicle_usage_save_draft", error="parsed object is required")
        request_id = self.store.save_draft(
            request_date=_request_date(args.get("request_date") or parsed.get("date")),
            user_id=self.user_id,
            dialog_id=self.dialog_id,
            response_text=str(args.get("response_text") or ""),
            parsed=parsed,
            status=str(args.get("status") or "pending_confirmation"),
        )
        return ToolResult(status="ok", tool="vehicle_usage_save_draft", data={"request_id": request_id})

    def vehicle_usage_save_report(self, args: dict[str, Any]) -> ToolResult:
        parsed = args.get("parsed")
        if not isinstance(parsed, dict):
            return ToolResult(status="invalid_tool_call", tool="vehicle_usage_save_report", error="parsed object is required")
        saved = self.store.save_report(
            request_date=_request_date(args.get("request_date") or parsed.get("date")),
            user_id=self.user_id,
            dialog_id=self.dialog_id,
            source_text=str(args.get("source_text") or ""),
            parsed=parsed,
        )
        return ToolResult(status="ok", tool="vehicle_usage_save_report", data=saved)

    async def vehicle_usage_send_message(self, args: dict[str, Any]) -> ToolResult:
        dialog_id = str(args.get("dialog_id") or self.dialog_id or get_settings().vehicle_usage_dialog_id).strip()
        message = str(args.get("message") or "").strip()
        if not dialog_id or not message:
            return ToolResult(
                status="invalid_tool_call",
                tool="vehicle_usage_send_message",
                error="dialog_id and message are required",
            )
        if get_settings().vehicle_usage_dry_run:
            return ToolResult(
                status="dry_run",
                tool="vehicle_usage_send_message",
                data={"dialog_id": dialog_id, "message": message},
            )
        await self.client.send_bot_message(dialog_id, message)
        return ToolResult(status="ok", tool="vehicle_usage_send_message", data={"dialog_id": dialog_id})


def _staff_roster_from_settings() -> list[StaffMember]:
    roster: list[StaffMember] = []
    for raw_item in get_settings().vehicle_usage_staff_roster.replace(";", "\n").splitlines():
        item = raw_item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) == 3:
            order = _optional_int(parts[0])
            user_id = _optional_int(parts[1])
            name = parts[2]
        elif len(parts) == 2:
            order = _optional_int(parts[0])
            user_id = None
            name = parts[1]
        else:
            order = len(roster) + 1
            user_id = None
            name = item
        if order is not None and name:
            roster.append(StaffMember(order=order, user_id=user_id, name=name))
    return sorted(roster, key=lambda item: item.order)


def _request_row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    parsed_json = result.get("parsed_json")
    if parsed_json:
        try:
            result["parsed"] = json.loads(str(parsed_json))
        except json.JSONDecodeError:
            result["parsed"] = None
    result.pop("parsed_json", None)
    return result


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


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
