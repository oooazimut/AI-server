from __future__ import annotations

import json
from typing import Any

from ai_server.utils import MOSCOW_TZ

from .agent_schema import PostgresAgentSchema


def _now() -> str:
    from datetime import datetime

    return datetime.now(MOSCOW_TZ).isoformat()


class PostgresVehicleUsageStore(PostgresAgentSchema):
    """Vehicle usage store: dialog_history + operational tables in the 'logistics' schema.

    Async methods (ensure_schema, load_turns, append_turn) satisfy AgentStorePort.
    Sync vehicle methods satisfy VehicleUsageStorePort via structural typing.
    """

    _SCHEMA = "logistics"

    async def ensure_schema(self) -> None:
        await super().ensure_schema()  # creates logistics schema + dialog_history table
        with self._sync_connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS logistics.employees (
                    id INTEGER PRIMARY KEY,
                    bitrix_user_id INTEGER,
                    full_name TEXT NOT NULL,
                    position TEXT NOT NULL DEFAULT ''
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS logistics.vehicles (
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
                CREATE TABLE IF NOT EXISTS logistics.vehicle_usage_requests (
                    id SERIAL PRIMARY KEY,
                    request_date TEXT NOT NULL,
                    user_id INTEGER,
                    dialog_id TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    sent_at TEXT,
                    response_text TEXT,
                    responded_at TEXT,
                    parsed_json TEXT,
                    reminder_count INTEGER NOT NULL DEFAULT 0,
                    last_reminder_at TEXT,
                    escalated_at TEXT,
                    UNIQUE (request_date, user_id)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS logistics.employee_daily_statuses (
                    id SERIAL PRIMARY KEY,
                    status_date TEXT NOT NULL,
                    employee_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    UNIQUE (status_date, employee_id)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS logistics.vehicle_daily_assignments (
                    id SERIAL PRIMARY KEY,
                    assignment_date TEXT NOT NULL,
                    vehicle_id INTEGER NOT NULL,
                    employee_id INTEGER,
                    notes TEXT NOT NULL DEFAULT '',
                    UNIQUE (assignment_date, vehicle_id)
                )
                """
            )

    def upsert_employees(self, members: list[Any]) -> None:
        with self._sync_connect() as db:
            for member in members:
                db.execute(
                    """
                    INSERT INTO logistics.employees (id, bitrix_user_id, full_name, position)
                    VALUES (%s, %s, %s, '')
                    ON CONFLICT (id) DO UPDATE SET
                        bitrix_user_id = EXCLUDED.bitrix_user_id,
                        full_name = EXCLUDED.full_name
                    """,
                    (member.order, member.user_id, member.name),
                )

    def context(self, *, request_date: str, user_id: int | None, dialog_id: str) -> dict[str, Any]:
        return {
            "request_date": request_date,
            "staff_roster": self.staff_roster(),
            "vehicles": self._vehicles(),
            "latest_request": self.latest_request(user_id=user_id, dialog_id=dialog_id),
        }

    def _vehicles(self) -> list[dict[str, Any]]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT id, brand_model, registration_number, debit_card_number, ppr_card_number
                FROM logistics.vehicles
                ORDER BY id
                """
            ).fetchall()
        return list(rows)

    def staff_roster(self) -> list[dict[str, Any]]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT id AS display_order, bitrix_user_id AS user_id, full_name, position
                FROM logistics.employees
                ORDER BY id
                """
            ).fetchall()
        return list(rows)

    def latest_request(self, *, user_id: int | None, dialog_id: str) -> dict[str, Any] | None:
        if dialog_id:
            with self._sync_connect() as db:
                row = db.execute(
                    """
                    SELECT * FROM logistics.vehicle_usage_requests
                    WHERE dialog_id = %s
                    ORDER BY request_date DESC, id DESC
                    LIMIT 1
                    """,
                    (dialog_id,),
                ).fetchone()
            return _parse_row(row)
        if user_id is None:
            return None
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT * FROM logistics.vehicle_usage_requests
                WHERE user_id = %s
                ORDER BY request_date DESC, id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return _parse_row(row)

    def get_request(self, *, request_date: str, user_id: int | None) -> dict[str, Any] | None:
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT * FROM logistics.vehicle_usage_requests
                WHERE request_date = %s AND (user_id = %s OR (%s IS NULL AND user_id IS NULL))
                LIMIT 1
                """,
                (request_date, user_id, user_id),
            ).fetchone()
        return _parse_row(row)

    def latest_requests(self, *, limit: int = 10) -> list[dict[str, Any]]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT * FROM logistics.vehicle_usage_requests
                ORDER BY request_date DESC, id DESC
                LIMIT %s
                """,
                (max(1, min(limit, 100)),),
            ).fetchall()
        return [item for row in rows if (item := _parse_row(row)) is not None]

    def create_sent_request(self, data: Any) -> int:
        with self._sync_connect() as db:
            db.execute(
                """
                INSERT INTO logistics.vehicle_usage_requests (
                    request_date, user_id, dialog_id, status, message, sent_at,
                    reminder_count, last_reminder_at
                )
                VALUES (%s, %s, %s, 'sent', %s, %s, %s, %s)
                ON CONFLICT (request_date, user_id) DO UPDATE SET
                    dialog_id = EXCLUDED.dialog_id,
                    status = CASE
                        WHEN logistics.vehicle_usage_requests.status = 'answered'
                            THEN logistics.vehicle_usage_requests.status
                        ELSE 'sent'
                    END,
                    message = EXCLUDED.message,
                    sent_at = COALESCE(logistics.vehicle_usage_requests.sent_at, EXCLUDED.sent_at),
                    reminder_count = GREATEST(
                        logistics.vehicle_usage_requests.reminder_count, EXCLUDED.reminder_count
                    ),
                    last_reminder_at = EXCLUDED.last_reminder_at
                """,
                (
                    data.request_date,
                    data.user_id,
                    data.dialog_id,
                    data.message,
                    data.sent_at,
                    data.reminder_count,
                    data.sent_at,
                ),
            )
            row = db.execute(
                """
                SELECT id FROM logistics.vehicle_usage_requests
                WHERE request_date = %s AND (user_id = %s OR (%s IS NULL AND user_id IS NULL))
                """,
                (data.request_date, data.user_id, data.user_id),
            ).fetchone()
        return int(row["id"]) if row else 0

    def mark_escalated(self, *, request_date: str, user_id: int | None, escalated_at: str) -> bool:
        with self._sync_connect() as db:
            cur = db.execute(
                """
                UPDATE logistics.vehicle_usage_requests
                SET escalated_at = %s
                WHERE request_date = %s AND (user_id = %s OR (%s IS NULL AND user_id IS NULL))
                """,
                (escalated_at, request_date, user_id, user_id),
            )
        return cur.rowcount > 0

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
        now = _now()
        with self._sync_connect() as db:
            db.execute(
                """
                INSERT INTO logistics.vehicle_usage_requests (
                    request_date, user_id, dialog_id, status, response_text, responded_at, parsed_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (request_date, user_id) DO UPDATE SET
                    dialog_id = EXCLUDED.dialog_id,
                    status = EXCLUDED.status,
                    response_text = EXCLUDED.response_text,
                    responded_at = EXCLUDED.responded_at,
                    parsed_json = EXCLUDED.parsed_json
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
                SELECT id FROM logistics.vehicle_usage_requests
                WHERE request_date = %s AND (user_id = %s OR (%s IS NULL AND user_id IS NULL))
                """,
                (request_date, user_id, user_id),
            ).fetchone()
        return int(row["id"]) if row else 0

    def replace_day_report(
        self,
        *,
        status_date: str,
        employee_statuses: list[tuple[int, str, str]],
        vehicle_assignments: list[tuple[int, int | None, str]],
    ) -> None:
        with self._sync_connect() as db:
            db.execute(
                "DELETE FROM logistics.employee_daily_statuses WHERE status_date = %s",
                (status_date,),
            )
            db.execute(
                "DELETE FROM logistics.vehicle_daily_assignments WHERE assignment_date = %s",
                (status_date,),
            )
            for employee_id, status, notes in employee_statuses:
                db.execute(
                    """
                    INSERT INTO logistics.employee_daily_statuses
                        (status_date, employee_id, status, notes)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (status_date, employee_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        notes = EXCLUDED.notes
                    """,
                    (status_date, employee_id, status, notes),
                )
            for vehicle_id, employee_id, notes in vehicle_assignments:
                db.execute(
                    """
                    INSERT INTO logistics.vehicle_daily_assignments
                        (assignment_date, vehicle_id, employee_id, notes)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (assignment_date, vehicle_id) DO UPDATE SET
                        employee_id = EXCLUDED.employee_id,
                        notes = EXCLUDED.notes
                    """,
                    (status_date, vehicle_id, employee_id, notes),
                )


def _parse_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    if result.get("parsed_json"):
        try:
            result["parsed"] = json.loads(result["parsed_json"])
        except (json.JSONDecodeError, TypeError):
            result["parsed"] = None
    return result
