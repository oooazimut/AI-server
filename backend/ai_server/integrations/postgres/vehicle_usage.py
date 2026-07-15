from __future__ import annotations

import json
from typing import Any

from ai_server.utils import MOSCOW_TZ

from .agent_schema import PostgresAgentSchema

DEFAULT_VEHICLES: tuple[tuple[int, str], ...] = (
    (1, "Авто 1"),
    (2, "Авто 2"),
    (3, "Авто 3"),
    (4, "Авто 4"),
    (5, "Авто 5"),
    (6, "Авто 6"),
)


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
                    source TEXT NOT NULL DEFAULT '',
                    UNIQUE (request_date, user_id)
                )
                """
            )
            db.execute("ALTER TABLE logistics.employees ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE")
            db.execute("ALTER TABLE logistics.vehicles ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE")
            db.execute(
                "ALTER TABLE logistics.vehicle_usage_requests ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT ''"
            )
            self._seed_default_vehicles(db)
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS logistics.vehicle_payment_cards (
                    id SERIAL PRIMARY KEY,
                    vehicle_id INTEGER NOT NULL,
                    card_number TEXT NOT NULL,
                    card_label TEXT NOT NULL DEFAULT '',
                    card_type TEXT NOT NULL DEFAULT '',
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL DEFAULT ''
                )
                """
            )
            self._add_request_columns(db)
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
                "ALTER TABLE logistics.employee_daily_statuses ADD COLUMN IF NOT EXISTS updated_by_user_id INTEGER"
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
            db.execute(
                "ALTER TABLE logistics.vehicle_daily_assignments ADD COLUMN IF NOT EXISTS assignment_status TEXT NOT NULL DEFAULT ''"
            )
            db.execute(
                "ALTER TABLE logistics.vehicle_daily_assignments ADD COLUMN IF NOT EXISTS updated_by_user_id INTEGER"
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS logistics.vehicle_daily_drivers (
                    id SERIAL PRIMARY KEY,
                    assignment_date TEXT NOT NULL,
                    vehicle_id INTEGER NOT NULL,
                    employee_id INTEGER NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    updated_by_user_id INTEGER,
                    UNIQUE (assignment_date, vehicle_id, employee_id)
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS logistics.vehicle_usage_revisions (
                    id SERIAL PRIMARY KEY,
                    report_date TEXT NOT NULL,
                    actor_user_id INTEGER,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS logistics.vehicle_usage_operators (
                    user_id INTEGER PRIMARY KEY,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    updated_by_user_id INTEGER,
                    updated_at TEXT NOT NULL DEFAULT ''
                )
                """
            )

    def _seed_default_vehicles(self, db: Any) -> None:
        for vehicle_id, vehicle_name in DEFAULT_VEHICLES:
            db.execute(
                """
                INSERT INTO logistics.vehicles (id, brand_model, registration_number, active)
                VALUES (%s, %s, '', TRUE)
                ON CONFLICT (id) DO NOTHING
                """,
                (vehicle_id, vehicle_name),
            )

    def _add_request_columns(self, db: Any) -> None:
        db.execute("ALTER TABLE logistics.vehicle_usage_requests ADD COLUMN IF NOT EXISTS created_by_user_id INTEGER")
        db.execute("ALTER TABLE logistics.vehicle_usage_requests ADD COLUMN IF NOT EXISTS updated_by_user_id INTEGER")
        db.execute("ALTER TABLE logistics.vehicle_usage_requests ADD COLUMN IF NOT EXISTS finalized_by_user_id INTEGER")
        db.execute("ALTER TABLE logistics.vehicle_usage_requests ADD COLUMN IF NOT EXISTS cancelled_at TEXT")
        db.execute(
            "ALTER TABLE logistics.vehicle_usage_requests ADD COLUMN IF NOT EXISTS cancel_reason TEXT NOT NULL DEFAULT ''"
        )

    def upsert_employees(self, members: list[Any]) -> None:
        with self._sync_connect() as db:
            active_ids = [int(member.order) for member in members]
            for member in members:
                db.execute(
                    """
                    INSERT INTO logistics.employees (id, bitrix_user_id, full_name, position, active)
                    VALUES (%s, %s, %s, '', TRUE)
                    ON CONFLICT (id) DO UPDATE SET
                        bitrix_user_id = EXCLUDED.bitrix_user_id,
                        full_name = EXCLUDED.full_name,
                        active = TRUE
                    """,
                    (member.order, member.user_id, member.name),
                )
            if active_ids:
                db.execute(
                    """
                    UPDATE logistics.employees
                    SET active = FALSE
                    WHERE id <> ALL(%s)
                    """,
                    (active_ids,),
                )

    def context(self, *, request_date: str, user_id: int | None, dialog_id: str) -> dict[str, Any]:
        return {
            "request_date": request_date,
            "staff_roster": self.staff_roster(),
            "vehicles": self.vehicles(),
            "latest_request": self.latest_request(user_id=user_id, dialog_id=dialog_id),
            "day_report": self.get_day_report(report_date=request_date),
        }

    def vehicles(self) -> list[dict[str, Any]]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT id, brand_model, registration_number, debit_card_number, ppr_card_number, active
                FROM logistics.vehicles
                WHERE active IS TRUE
                ORDER BY id
                """
            ).fetchall()
        return list(rows)

    def staff_roster(self) -> list[dict[str, Any]]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT id AS display_order, bitrix_user_id AS user_id, full_name, position, active
                FROM logistics.employees
                WHERE active IS TRUE
                ORDER BY id
                """
            ).fetchall()
        return list(rows)

    def vehicle_usage_operator_ids(self) -> set[int]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT user_id
                FROM logistics.vehicle_usage_operators
                WHERE active IS TRUE
                ORDER BY user_id
                """
            ).fetchall()
        return {int(row["user_id"]) for row in rows if row.get("user_id") is not None}

    def set_vehicle_usage_operators(self, *, operator_user_ids: list[int], actor_user_id: int | None) -> list[int]:
        now = _now()
        cleaned = sorted({int(user_id) for user_id in operator_user_ids if int(user_id) > 0})
        with self._sync_connect() as db:
            db.execute(
                """
                UPDATE logistics.vehicle_usage_operators
                SET active = FALSE, updated_by_user_id = %s, updated_at = %s
                """,
                (actor_user_id, now),
            )
            for user_id in cleaned:
                db.execute(
                    """
                    INSERT INTO logistics.vehicle_usage_operators (user_id, active, updated_by_user_id, updated_at)
                    VALUES (%s, TRUE, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        active = TRUE,
                        updated_by_user_id = EXCLUDED.updated_by_user_id,
                        updated_at = EXCLUDED.updated_at
                    """,
                    (user_id, actor_user_id, now),
                )
            db.execute(
                """
                INSERT INTO logistics.vehicle_usage_revisions
                    (report_date, actor_user_id, action, payload_json, created_at)
                VALUES (%s, %s, 'set_operators', %s, %s)
                """,
                (
                    "config",
                    actor_user_id,
                    json.dumps({"operator_user_ids": cleaned}, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
        return cleaned

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

    def get_day_report(self, *, report_date: str) -> dict[str, Any]:
        with self._sync_connect() as db:
            employees = db.execute(
                """
                SELECT s.status_date, s.employee_id, e.full_name, s.status, vehicle_link.vehicle_name, s.notes
                FROM logistics.employee_daily_statuses s
                LEFT JOIN logistics.employees e ON e.id = s.employee_id
                LEFT JOIN (
                    SELECT d.assignment_date, d.employee_id,
                           string_agg(v.brand_model, ', ' ORDER BY v.id) AS vehicle_name
                    FROM logistics.vehicle_daily_drivers d
                    LEFT JOIN logistics.vehicles v ON v.id = d.vehicle_id
                    GROUP BY d.assignment_date, d.employee_id
                ) vehicle_link
                    ON vehicle_link.assignment_date = s.status_date
                   AND vehicle_link.employee_id = s.employee_id
                WHERE s.status_date = %s
                ORDER BY s.employee_id
                """,
                (report_date,),
            ).fetchall()
            vehicles = db.execute(
                """
                SELECT a.assignment_date, a.vehicle_id, v.brand_model, v.registration_number,
                       a.employee_id, e.full_name AS employee_name,
                       a.assignment_status, a.notes
                FROM logistics.vehicle_daily_assignments a
                LEFT JOIN logistics.vehicles v ON v.id = a.vehicle_id
                LEFT JOIN logistics.employees e ON e.id = a.employee_id
                WHERE a.assignment_date = %s
                ORDER BY a.vehicle_id
                """,
                (report_date,),
            ).fetchall()
            drivers = db.execute(
                """
                SELECT d.assignment_date, d.vehicle_id, d.employee_id, e.full_name, d.notes
                FROM logistics.vehicle_daily_drivers d
                LEFT JOIN logistics.employees e ON e.id = d.employee_id
                WHERE d.assignment_date = %s
                ORDER BY d.vehicle_id, d.employee_id
                """,
                (report_date,),
            ).fetchall()
            needs_legacy = (
                (not employees and not vehicles and not drivers)
                or not vehicles
                or not drivers
                or any(not row.get("vehicle_name") for row in employees)
            )
            if needs_legacy:
                request = db.execute(
                    """
                    SELECT * FROM logistics.vehicle_usage_requests
                    WHERE request_date = %s AND parsed_json IS NOT NULL
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (report_date,),
                ).fetchone()
            else:
                request = None
        legacy_report = _legacy_day_report(report_date, _parse_row(request))
        if legacy_report is not None:
            if not employees and not vehicles and not drivers:
                return legacy_report
            legacy_employees = (
                legacy_report.get("employee_statuses")
                if isinstance(legacy_report.get("employee_statuses"), list)
                else []
            )
            legacy_vehicles = (
                legacy_report.get("vehicle_assignments")
                if isinstance(legacy_report.get("vehicle_assignments"), list)
                else []
            )
            employee_statuses = _merge_legacy_employee_vehicles(
                list(employees),
                legacy_employees,
                legacy_vehicles,
            )
            vehicle_assignments = list(vehicles)
            vehicle_drivers = list(drivers)
            supplemented = False
            if not vehicle_assignments and legacy_report.get("vehicle_assignments"):
                vehicle_assignments = list(legacy_report["vehicle_assignments"])
                supplemented = True
            if not vehicle_drivers and legacy_report.get("vehicle_drivers"):
                vehicle_drivers = list(legacy_report["vehicle_drivers"])
                supplemented = True
            if employee_statuses != list(employees):
                supplemented = True
            if supplemented:
                return {
                    "report_date": report_date,
                    "source": "normalized_tables+vehicle_usage_requests.parsed_json",
                    "employee_statuses": employee_statuses,
                    "vehicle_assignments": vehicle_assignments,
                    "vehicle_drivers": vehicle_drivers,
                    "request": legacy_report.get("request"),
                }
        return {
            "report_date": report_date,
            "source": "normalized_tables",
            "employee_statuses": list(employees),
            "vehicle_assignments": list(vehicles),
            "vehicle_drivers": list(drivers),
        }

    def get_employee_period_report(self, *, employee_name: str, date_from: str, date_to: str) -> dict[str, Any]:
        pattern = f"%{employee_name.strip()}%"
        with self._sync_connect() as db:
            employee = db.execute(
                """
                SELECT id, full_name
                FROM logistics.employees
                WHERE full_name ILIKE %s
                ORDER BY id
                LIMIT 1
                """,
                (pattern,),
            ).fetchone()
            rows = []
            if employee:
                rows = db.execute(
                    """
                    SELECT s.status_date, s.status, s.notes,
                           v.brand_model AS vehicle_name, a.assignment_status
                    FROM logistics.employee_daily_statuses s
                    LEFT JOIN logistics.vehicle_daily_drivers d
                        ON d.assignment_date = s.status_date AND d.employee_id = s.employee_id
                    LEFT JOIN logistics.vehicles v ON v.id = d.vehicle_id
                    LEFT JOIN logistics.vehicle_daily_assignments a
                        ON a.assignment_date = d.assignment_date AND a.vehicle_id = d.vehicle_id
                    WHERE s.employee_id = %s AND s.status_date BETWEEN %s AND %s
                    ORDER BY s.status_date
                    """,
                    (employee["id"], date_from, date_to),
                ).fetchall()
            legacy_requests = db.execute(
                """
                SELECT * FROM logistics.vehicle_usage_requests
                WHERE request_date BETWEEN %s AND %s AND parsed_json IS NOT NULL
                ORDER BY request_date
                """,
                (date_from, date_to),
            ).fetchall()
        legacy_rows = _legacy_employee_period_rows(employee_name, [_parse_row(row) for row in legacy_requests])
        return {
            "subject": "employee",
            "employee_name": employee["full_name"] if employee else employee_name,
            "date_from": date_from,
            "date_to": date_to,
            "source": "normalized_tables" if rows else "vehicle_usage_requests.parsed_json",
            "days": list(rows) if rows else legacy_rows,
            "summary": _status_summary(list(rows) if rows else legacy_rows, "status"),
        }

    def get_vehicle_period_report(self, *, vehicle_name: str, date_from: str, date_to: str) -> dict[str, Any]:
        pattern = f"%{vehicle_name.strip()}%"
        with self._sync_connect() as db:
            vehicle = db.execute(
                """
                SELECT id, brand_model
                FROM logistics.vehicles
                WHERE brand_model ILIKE %s OR registration_number ILIKE %s
                ORDER BY id
                LIMIT 1
                """,
                (pattern, pattern),
            ).fetchone()
            rows = []
            if vehicle:
                rows = db.execute(
                    """
                    SELECT a.assignment_date, a.assignment_status AS status, a.notes,
                           e.full_name AS employee_name
                    FROM logistics.vehicle_daily_assignments a
                    LEFT JOIN logistics.vehicle_daily_drivers d
                        ON d.assignment_date = a.assignment_date AND d.vehicle_id = a.vehicle_id
                    LEFT JOIN logistics.employees e ON e.id = d.employee_id
                    WHERE a.vehicle_id = %s AND a.assignment_date BETWEEN %s AND %s
                    ORDER BY a.assignment_date, e.full_name
                    """,
                    (vehicle["id"], date_from, date_to),
                ).fetchall()
            legacy_requests = db.execute(
                """
                SELECT * FROM logistics.vehicle_usage_requests
                WHERE request_date BETWEEN %s AND %s AND parsed_json IS NOT NULL
                ORDER BY request_date
                """,
                (date_from, date_to),
            ).fetchall()
        days = (
            _group_vehicle_period_rows(list(rows))
            if rows
            else _legacy_vehicle_period_rows(vehicle_name, [_parse_row(row) for row in legacy_requests])
        )
        return {
            "subject": "vehicle",
            "vehicle_name": vehicle["brand_model"] if vehicle else vehicle_name,
            "date_from": date_from,
            "date_to": date_to,
            "source": "normalized_tables" if rows else "vehicle_usage_requests.parsed_json",
            "days": days,
            "summary": _status_summary(days, "status"),
        }

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
                    reminder_count, last_reminder_at, created_by_user_id, updated_by_user_id, source
                )
                VALUES (%s, %s, %s, 'sent', %s, %s, %s, %s, %s, %s, %s)
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
                    last_reminder_at = EXCLUDED.last_reminder_at,
                    updated_by_user_id = EXCLUDED.updated_by_user_id,
                    source = COALESCE(NULLIF(EXCLUDED.source, ''), logistics.vehicle_usage_requests.source)
                """,
                (
                    data.request_date,
                    data.user_id,
                    data.dialog_id,
                    data.message,
                    data.sent_at,
                    data.reminder_count,
                    data.sent_at,
                    data.user_id,
                    data.user_id,
                    str(getattr(data, "source", "") or ""),
                ),
            )
            row = db.execute(
                """
                SELECT id FROM logistics.vehicle_usage_requests
                WHERE request_date = %s AND (user_id = %s OR (%s IS NULL AND user_id IS NULL))
                """,
                (data.request_date, data.user_id, data.user_id),
            ).fetchone()
            db.execute(
                """
                INSERT INTO logistics.vehicle_usage_revisions
                    (report_date, actor_user_id, action, payload_json, created_at)
                VALUES (%s, %s, 'start_day', %s, %s)
                """,
                (
                    data.request_date,
                    data.user_id,
                    json.dumps(
                        {
                            "dialog_id": data.dialog_id,
                            "message": data.message,
                            "sent_at": data.sent_at,
                            "reminder_count": data.reminder_count,
                            "source": str(getattr(data, "source", "") or ""),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    _now(),
                ),
            )
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
                    request_date, user_id, dialog_id, status, response_text, responded_at, parsed_json,
                    created_by_user_id, updated_by_user_id, finalized_by_user_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (request_date, user_id) DO UPDATE SET
                    dialog_id = EXCLUDED.dialog_id,
                    status = EXCLUDED.status,
                    response_text = EXCLUDED.response_text,
                    responded_at = EXCLUDED.responded_at,
                    parsed_json = EXCLUDED.parsed_json,
                    updated_by_user_id = EXCLUDED.updated_by_user_id,
                    finalized_by_user_id = EXCLUDED.finalized_by_user_id
                """,
                (
                    request_date,
                    user_id,
                    dialog_id,
                    status,
                    response_text,
                    now,
                    json.dumps(parsed, ensure_ascii=False, sort_keys=True),
                    user_id,
                    user_id,
                    user_id if status == "answered" else None,
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
        vehicle_assignments: list[tuple[int, int | None, str] | tuple[int, int | None, str, str]],
        actor_user_id: int | None = None,
    ) -> None:
        now = _now()
        with self._sync_connect() as db:
            db.execute(
                "DELETE FROM logistics.employee_daily_statuses WHERE status_date = %s",
                (status_date,),
            )
            db.execute(
                "DELETE FROM logistics.vehicle_daily_assignments WHERE assignment_date = %s",
                (status_date,),
            )
            db.execute(
                "DELETE FROM logistics.vehicle_daily_drivers WHERE assignment_date = %s",
                (status_date,),
            )
            for employee_id, status, notes in employee_statuses:
                db.execute(
                    """
                    INSERT INTO logistics.employee_daily_statuses
                        (status_date, employee_id, status, notes, updated_by_user_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (status_date, employee_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        notes = EXCLUDED.notes,
                        updated_by_user_id = EXCLUDED.updated_by_user_id
                    """,
                    (status_date, employee_id, status, notes, actor_user_id),
                )
            seen_vehicle_ids: set[int] = set()
            for raw_assignment in vehicle_assignments:
                if len(raw_assignment) == 4:
                    vehicle_id, employee_id, assignment_status, notes = raw_assignment
                    assignment_status = str(assignment_status or ("in_use" if employee_id is not None else "idle"))
                else:
                    vehicle_id, employee_id, notes = raw_assignment
                    assignment_status = "in_use" if employee_id is not None else "idle"
                notes = str(notes or "")
                if not isinstance(vehicle_id, int):
                    continue
                if vehicle_id not in seen_vehicle_ids:
                    seen_vehicle_ids.add(vehicle_id)
                    db.execute(
                        """
                        INSERT INTO logistics.vehicle_daily_assignments
                            (assignment_date, vehicle_id, employee_id, notes, assignment_status, updated_by_user_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (assignment_date, vehicle_id) DO UPDATE SET
                            employee_id = COALESCE(logistics.vehicle_daily_assignments.employee_id, EXCLUDED.employee_id),
                            notes = EXCLUDED.notes,
                            assignment_status = EXCLUDED.assignment_status,
                            updated_by_user_id = EXCLUDED.updated_by_user_id
                        """,
                        (status_date, vehicle_id, employee_id, notes, assignment_status, actor_user_id),
                    )
                if employee_id is None:
                    continue
                db.execute(
                    """
                    INSERT INTO logistics.vehicle_daily_drivers
                        (assignment_date, vehicle_id, employee_id, notes, updated_by_user_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (assignment_date, vehicle_id, employee_id) DO UPDATE SET
                        notes = EXCLUDED.notes,
                        updated_by_user_id = EXCLUDED.updated_by_user_id
                    """,
                    (status_date, vehicle_id, employee_id, notes, actor_user_id),
                )
            db.execute(
                """
                INSERT INTO logistics.vehicle_usage_revisions
                    (report_date, actor_user_id, action, payload_json, created_at)
                VALUES (%s, %s, 'replace_day_report', %s, %s)
                """,
                (
                    status_date,
                    actor_user_id,
                    json.dumps(
                        {
                            "employee_statuses": employee_statuses,
                            "vehicle_assignments": vehicle_assignments,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    now,
                ),
            )

    def update_day_report(
        self,
        *,
        report_date: str,
        people: list[dict[str, Any]],
        vehicles: list[dict[str, Any]],
        actor_user_id: int | None = None,
        change_summary: str = "",
    ) -> dict[str, Any]:
        now = _now()
        employee_updates = 0
        vehicle_updates = 0
        with self._sync_connect() as db:
            employees = {
                _norm_name(row.get("full_name")): row
                for row in db.execute("SELECT id, full_name FROM logistics.employees").fetchall()
            }
            vehicles_by_name = {}
            for row in db.execute("SELECT id, brand_model, registration_number FROM logistics.vehicles").fetchall():
                vehicles_by_name[_norm_name(row.get("brand_model"))] = row
                vehicles_by_name[_norm_name(row.get("registration_number"))] = row

            for item in people:
                employee_id = _optional_int(item.get("employee_id") or item.get("staff_order"))
                if employee_id is None:
                    employee = employees.get(_norm_name(item.get("full_name") or item.get("name")))
                    employee_id = _optional_int(employee.get("id")) if employee else None
                if employee_id is None:
                    continue
                db.execute(
                    """
                    INSERT INTO logistics.employee_daily_statuses
                        (status_date, employee_id, status, notes, updated_by_user_id)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (status_date, employee_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        notes = EXCLUDED.notes,
                        updated_by_user_id = EXCLUDED.updated_by_user_id
                    """,
                    (
                        report_date,
                        employee_id,
                        str(item.get("status") or "unknown"),
                        str(item.get("notes") or ""),
                        actor_user_id,
                    ),
                )
                employee_updates += 1

            for item in vehicles:
                vehicle_id = _optional_int(item.get("vehicle_id"))
                if vehicle_id is None:
                    vehicle = vehicles_by_name.get(
                        _norm_name(item.get("vehicle_name") or item.get("name") or item.get("vehicle"))
                    )
                    vehicle_id = _optional_int(vehicle.get("id")) if vehicle else None
                if vehicle_id is None:
                    continue
                raw_status = str(item.get("status") or item.get("assignment_status") or "").strip()
                notes = str(item.get("notes") or "")
                drivers = item.get("drivers")
                if not isinstance(drivers, list):
                    drivers = [item.get("driver") or item.get("employee_name")]
                driver_ids: list[int] = []
                for raw_driver in drivers:
                    employee = employees.get(_norm_name(raw_driver))
                    employee_id = _optional_int(employee.get("id")) if employee else None
                    if employee_id is not None and employee_id not in driver_ids:
                        driver_ids.append(employee_id)
                existing_assignment = db.execute(
                    """
                    SELECT assignment_status, notes
                    FROM logistics.vehicle_daily_assignments
                    WHERE assignment_date = %s AND vehicle_id = %s
                    """,
                    (report_date, vehicle_id),
                ).fetchone()
                existing_driver_rows = db.execute(
                    """
                    SELECT employee_id
                    FROM logistics.vehicle_daily_drivers
                    WHERE assignment_date = %s AND vehicle_id = %s
                    ORDER BY employee_id
                    """,
                    (report_date, vehicle_id),
                ).fetchall()
                existing_driver_ids = [
                    int(row["employee_id"]) for row in existing_driver_rows if row.get("employee_id") is not None
                ]
                replace_drivers = bool(item.get("replace_drivers"))
                merged_driver_ids = list(driver_ids if replace_drivers else existing_driver_ids)
                for employee_id in driver_ids:
                    if employee_id not in merged_driver_ids:
                        merged_driver_ids.append(employee_id)
                for employee_id in driver_ids:
                    other_rows = db.execute(
                        """
                        SELECT vehicle_id
                        FROM logistics.vehicle_daily_drivers
                        WHERE assignment_date = %s AND employee_id = %s AND vehicle_id <> %s
                        """,
                        (report_date, employee_id, vehicle_id),
                    ).fetchall()
                    db.execute(
                        """
                        DELETE FROM logistics.vehicle_daily_drivers
                        WHERE assignment_date = %s AND employee_id = %s AND vehicle_id <> %s
                        """,
                        (report_date, employee_id, vehicle_id),
                    )
                    for other in other_rows:
                        other_vehicle_id = _optional_int(other.get("vehicle_id"))
                        if other_vehicle_id is None:
                            continue
                        remaining = db.execute(
                            """
                            SELECT employee_id
                            FROM logistics.vehicle_daily_drivers
                            WHERE assignment_date = %s AND vehicle_id = %s
                            ORDER BY employee_id
                            """,
                            (report_date, other_vehicle_id),
                        ).fetchall()
                        first_remaining = _optional_int(remaining[0].get("employee_id")) if remaining else None
                        db.execute(
                            """
                            UPDATE logistics.vehicle_daily_assignments
                            SET employee_id = %s,
                                assignment_status = CASE WHEN %s IS NULL THEN 'idle' ELSE assignment_status END,
                                updated_by_user_id = %s
                            WHERE assignment_date = %s AND vehicle_id = %s
                            """,
                            (first_remaining, first_remaining, actor_user_id, report_date, other_vehicle_id),
                        )
                status = raw_status or (
                    str(existing_assignment.get("assignment_status") or "") if existing_assignment else ""
                )
                if not status or status == "unknown":
                    status = "in_use" if merged_driver_ids else "idle"
                first_driver = merged_driver_ids[0] if merged_driver_ids else None
                db.execute(
                    """
                    INSERT INTO logistics.vehicle_daily_assignments
                        (assignment_date, vehicle_id, employee_id, notes, assignment_status, updated_by_user_id)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (assignment_date, vehicle_id) DO UPDATE SET
                        employee_id = EXCLUDED.employee_id,
                        notes = EXCLUDED.notes,
                        assignment_status = EXCLUDED.assignment_status,
                        updated_by_user_id = EXCLUDED.updated_by_user_id
                    """,
                    (report_date, vehicle_id, first_driver, notes, status, actor_user_id),
                )
                db.execute(
                    "DELETE FROM logistics.vehicle_daily_drivers WHERE assignment_date = %s AND vehicle_id = %s",
                    (report_date, vehicle_id),
                )
                for employee_id in merged_driver_ids:
                    db.execute(
                        """
                        INSERT INTO logistics.vehicle_daily_drivers
                            (assignment_date, vehicle_id, employee_id, notes, updated_by_user_id)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (assignment_date, vehicle_id, employee_id) DO UPDATE SET
                            notes = EXCLUDED.notes,
                            updated_by_user_id = EXCLUDED.updated_by_user_id
                        """,
                        (report_date, vehicle_id, employee_id, notes, actor_user_id),
                    )
                    db.execute(
                        """
                        UPDATE logistics.employee_daily_statuses
                        SET status = 'on_car', updated_by_user_id = %s
                        WHERE status_date = %s
                          AND employee_id = %s
                          AND status IN ('unknown', 'office', 'work', 'working')
                        """,
                        (actor_user_id, report_date, employee_id),
                    )
                vehicle_updates += 1

            db.execute(
                """
                INSERT INTO logistics.vehicle_usage_revisions
                    (report_date, actor_user_id, action, payload_json, created_at)
                VALUES (%s, %s, 'update_day_report', %s, %s)
                """,
                (
                    report_date,
                    actor_user_id,
                    json.dumps(
                        {
                            "people": people,
                            "vehicles": vehicles,
                            "change_summary": change_summary,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    now,
                ),
            )
        return {
            "report_date": report_date,
            "employee_updates": employee_updates,
            "vehicle_updates": vehicle_updates,
            "change_summary": change_summary,
        }

    def finalize_pending_unknowns(
        self,
        *,
        report_date: str,
        actor_user_id: int | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        note = reason or "Auto-filled missing vehicle usage data as unknown."
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT * FROM logistics.vehicle_usage_requests
                WHERE request_date = %s
                  AND status IN ('pending_clarification', 'pending_confirmation')
                  AND parsed_json IS NOT NULL
                ORDER BY responded_at DESC NULLS LAST, id DESC
                LIMIT 1
                """,
                (report_date,),
            ).fetchone()
        request = _parse_row(row)
        if not request or not isinstance(request.get("parsed"), dict):
            return {"status": "skipped", "reason": "no_pending_draft", "report_date": report_date}

        employees = self.staff_roster()
        vehicles = self.vehicles()
        completed = _complete_pending_report_with_unknowns(
            request["parsed"],
            report_date=report_date,
            employees=employees,
            vehicles=vehicles,
            note=note,
        )
        employee_statuses, vehicle_assignments = _report_rows_from_completed(
            completed,
            employees=employees,
            vehicles=vehicles,
        )
        request_id = self.save_draft(
            request_date=report_date,
            user_id=_optional_int(request.get("user_id")),
            dialog_id=str(request.get("dialog_id") or ""),
            response_text=str(request.get("response_text") or note),
            parsed=completed,
            status="answered",
        )
        self.replace_day_report(
            status_date=report_date,
            employee_statuses=employee_statuses,
            vehicle_assignments=vehicle_assignments,
            actor_user_id=actor_user_id or _optional_int(request.get("user_id")),
        )
        return {
            "status": "finalized_unknown",
            "report_date": report_date,
            "request_id": request_id,
            "employee_statuses_saved": len(employee_statuses),
            "vehicle_assignments_saved": len(vehicle_assignments),
            "unknown_employees": completed.get("unknown_employees", []),
            "unknown_vehicles": completed.get("unknown_vehicles", []),
        }

    def auto_close_unanswered_day(
        self,
        *,
        report_date: str,
        reason: str,
    ) -> dict[str, Any]:
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT * FROM logistics.vehicle_usage_requests
                WHERE request_date = %s
                ORDER BY id DESC
                LIMIT 1
                """,
                (report_date,),
            ).fetchone()
        request = _parse_row(row)
        if _is_manual_pending_vehicle_usage_draft(request):
            result = self.finalize_pending_unknowns(
                report_date=report_date,
                actor_user_id=_optional_int(request.get("user_id")) if request else None,
                reason="Auto-filled missing vehicle usage data as unknown at day close.",
            )
            if result.get("status") == "finalized_unknown":
                result["reason"] = "pending_draft_finalized_at_day_close"
                return result
        if _has_useful_vehicle_usage_response(request):
            return {
                "status": "skipped",
                "reason": "useful_response_exists",
                "report_date": report_date,
                "request_status": request.get("status") if request else None,
            }

        operator_ids = sorted(self.vehicle_usage_operator_ids())
        user_id = operator_ids[0] if operator_ids else _optional_int(request.get("user_id")) if request else None
        dialog_id = str(request.get("dialog_id") or user_id or "") if request else str(user_id or "")
        request_id = self.cancel_day_report(
            report_date=report_date,
            user_id=user_id,
            dialog_id=dialog_id,
            reason=reason,
        )
        return {
            "status": "closed_day_off",
            "report_date": report_date,
            "request_id": request_id,
            "user_id": user_id,
        }

    def cancel_day_report(
        self,
        *,
        report_date: str,
        user_id: int | None,
        dialog_id: str,
        reason: str,
    ) -> int:
        now = _now()
        employees = self.staff_roster()
        vehicles = self.vehicles()
        parsed = {
            "date": report_date,
            "status": "day_off",
            "reason": reason,
            "people": [
                {
                    "staff_order": row.get("display_order"),
                    "full_name": row.get("full_name"),
                    "status": "day_off",
                    "notes": reason,
                }
                for row in employees
            ],
            "vehicles": [
                {
                    "vehicle_id": row.get("id"),
                    "vehicle_name": row.get("brand_model"),
                    "status": "not_required",
                    "notes": reason,
                }
                for row in vehicles
            ],
        }
        request_id = self.save_draft(
            request_date=report_date,
            user_id=user_id,
            dialog_id=dialog_id,
            response_text=reason,
            parsed=parsed,
            status="cancelled_day_off",
        )
        self.replace_day_report(
            status_date=report_date,
            employee_statuses=[
                (int(row["display_order"]), "day_off", reason)
                for row in employees
                if row.get("display_order") is not None
            ],
            vehicle_assignments=[
                (int(row["id"]), None, "not_required", reason) for row in vehicles if row.get("id") is not None
            ],
            actor_user_id=user_id,
        )
        with self._sync_connect() as db:
            db.execute(
                """
                UPDATE logistics.vehicle_usage_requests
                SET cancelled_at = %s, cancel_reason = %s, updated_by_user_id = %s
                WHERE id = %s
                """,
                (now, reason, user_id, request_id),
            )
        return request_id


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


def _complete_pending_report_with_unknowns(
    parsed: dict[str, Any],
    *,
    report_date: str,
    employees: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    note: str,
) -> dict[str, Any]:
    employee_by_id = {_employee_row_id(row): row for row in employees if _employee_row_id(row) is not None}
    vehicle_by_id = {_vehicle_row_id(row): row for row in vehicles if _vehicle_row_id(row) is not None}
    employees_by_name = {
        _norm_name(row.get("full_name") or row.get("name")): employee_id
        for employee_id, row in employee_by_id.items()
        if _norm_name(row.get("full_name") or row.get("name"))
    }
    vehicles_by_name = {}
    for vehicle_id, row in vehicle_by_id.items():
        for value in (row.get("brand_model"), row.get("registration_number"), row.get("vehicle_name"), row.get("name")):
            name = _norm_name(value)
            if name:
                vehicles_by_name[name] = vehicle_id

    known_people: dict[int, dict[str, Any]] = {}
    vehicle_ids_by_employee: dict[int, int] = {}
    for entry in _raw_people_entries(parsed):
        employee_id = _entry_employee_id(entry, employees_by_name)
        if employee_id is None:
            continue
        known_people[employee_id] = dict(entry)
        vehicle_id = _entry_vehicle_id(entry, vehicles_by_name)
        if vehicle_id is not None:
            vehicle_ids_by_employee[employee_id] = vehicle_id

    known_vehicles: dict[int, dict[str, Any]] = {}
    for entry in _raw_vehicle_entries(parsed):
        vehicle_id = _entry_vehicle_id(entry, vehicles_by_name)
        if vehicle_id is None:
            continue
        item = dict(entry)
        item["vehicle_id"] = vehicle_id
        item["drivers"] = _driver_names(entry, employees_by_name, employee_by_id)
        known_vehicles[vehicle_id] = item
        for driver_name in item["drivers"]:
            employee_id = employees_by_name.get(_norm_name(driver_name))
            if employee_id is not None:
                vehicle_ids_by_employee[employee_id] = vehicle_id

    for employee_id, vehicle_id in vehicle_ids_by_employee.items():
        vehicle = vehicle_by_id.get(vehicle_id)
        employee = employee_by_id.get(employee_id)
        if not vehicle or not employee:
            continue
        vehicle_entry = known_vehicles.setdefault(
            vehicle_id,
            {
                "vehicle_id": vehicle_id,
                "vehicle_name": _vehicle_name(vehicle),
                "status": "in_use",
                "drivers": [],
                "notes": "",
            },
        )
        drivers = vehicle_entry.setdefault("drivers", [])
        employee_name = _employee_name(employee)
        if employee_name and employee_name not in drivers:
            drivers.append(employee_name)

    people: list[dict[str, Any]] = []
    unknown_employees: list[str] = []
    assigned_employee_ids = set(vehicle_ids_by_employee)
    for employee_id in sorted(employee_by_id):
        employee = employee_by_id[employee_id]
        known = known_people.get(employee_id, {})
        has_vehicle = employee_id in assigned_employee_ids
        status = _final_employee_status(known.get("status"), has_vehicle=has_vehicle)
        notes = str(known.get("notes") or "")
        if status == "unknown":
            notes = notes or note
            unknown_employees.append(_employee_name(employee))
        item = dict(known)
        item.update(
            {
                "staff_order": employee_id,
                "full_name": _employee_name(employee),
                "status": status,
                "notes": notes,
            }
        )
        vehicle_id = vehicle_ids_by_employee.get(employee_id)
        if vehicle_id is not None and vehicle_id in vehicle_by_id:
            item["vehicle"] = _vehicle_name(vehicle_by_id[vehicle_id])
        people.append(item)

    completed_vehicles: list[dict[str, Any]] = []
    unknown_vehicles: list[str] = []
    for vehicle_id in sorted(vehicle_by_id):
        vehicle = vehicle_by_id[vehicle_id]
        known = known_vehicles.get(vehicle_id)
        if known is None:
            completed_vehicles.append(
                {
                    "vehicle_id": vehicle_id,
                    "vehicle_name": _vehicle_name(vehicle),
                    "status": "unknown",
                    "drivers": [],
                    "notes": note,
                }
            )
            unknown_vehicles.append(_vehicle_name(vehicle))
            continue
        drivers = [str(item).strip() for item in known.get("drivers", []) if str(item or "").strip()]
        status = _final_vehicle_status(known.get("status") or known.get("assignment_status"), has_drivers=bool(drivers))
        notes = str(known.get("notes") or "")
        if status == "unknown":
            notes = notes or note
            unknown_vehicles.append(_vehicle_name(vehicle))
        item = dict(known)
        item.update(
            {
                "vehicle_id": vehicle_id,
                "vehicle_name": _vehicle_name(vehicle),
                "status": status,
                "drivers": drivers,
                "notes": notes,
            }
        )
        completed_vehicles.append(item)

    completed = dict(parsed)
    completed.update(
        {
            "date": report_date,
            "people": people,
            "vehicles": completed_vehicles,
            "auto_completed_unknown": True,
            "unknown_fill_note": note,
            "unknown_employees": unknown_employees,
            "unknown_vehicles": unknown_vehicles,
        }
    )
    return completed


def _report_rows_from_completed(
    completed: dict[str, Any],
    *,
    employees: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
) -> tuple[list[tuple[int, str, str]], list[tuple[int, int | None, str, str]]]:
    employee_ids = {_employee_row_id(row) for row in employees}
    vehicle_ids = {_vehicle_row_id(row) for row in vehicles}
    employees_by_name = {
        _norm_name(row.get("full_name") or row.get("name")): _employee_row_id(row)
        for row in employees
        if _employee_row_id(row) is not None and _norm_name(row.get("full_name") or row.get("name"))
    }
    employee_statuses: list[tuple[int, str, str]] = []
    for entry in _raw_people_entries(completed):
        employee_id = _entry_employee_id(entry, employees_by_name)
        if employee_id is None or employee_id not in employee_ids:
            continue
        employee_statuses.append((employee_id, str(entry.get("status") or "unknown"), str(entry.get("notes") or "")))

    vehicle_assignments: list[tuple[int, int | None, str, str]] = []
    vehicles_by_name = {}
    for row in vehicles:
        vehicle_id = _vehicle_row_id(row)
        if vehicle_id is None:
            continue
        for value in (row.get("brand_model"), row.get("registration_number"), row.get("vehicle_name"), row.get("name")):
            name = _norm_name(value)
            if name:
                vehicles_by_name[name] = vehicle_id
    for entry in _raw_vehicle_entries(completed):
        vehicle_id = _entry_vehicle_id(entry, vehicles_by_name)
        if vehicle_id is None or vehicle_id not in vehicle_ids:
            continue
        status = str(entry.get("status") or entry.get("assignment_status") or "unknown")
        notes = str(entry.get("notes") or "")
        raw_drivers = entry.get("drivers")
        if not isinstance(raw_drivers, list):
            raw_drivers = [entry.get("driver") or entry.get("employee_name") or entry.get("assigned_to")]
        driver_ids = []
        for driver_name in raw_drivers:
            employee_id = employees_by_name.get(_norm_name(driver_name))
            if employee_id is not None and employee_id not in driver_ids:
                driver_ids.append(employee_id)
        if not driver_ids:
            vehicle_assignments.append((vehicle_id, None, status, notes))
            continue
        for employee_id in driver_ids:
            vehicle_assignments.append((vehicle_id, employee_id, status, notes))
    return employee_statuses, vehicle_assignments


def _is_manual_pending_vehicle_usage_draft(request: dict[str, Any] | None) -> bool:
    if not request:
        return False
    status = str(request.get("status") or "").strip()
    source = str(request.get("source") or "").strip()
    return (
        source == "manual"
        and status in {"pending_clarification", "pending_confirmation"}
        and isinstance(request.get("parsed"), dict)
    )


def _has_useful_vehicle_usage_response(request: dict[str, Any] | None) -> bool:
    if not request:
        return False
    status = str(request.get("status") or "").strip()
    if status in {"answered", "cancelled_day_off"}:
        return True
    return status in {"pending_clarification", "pending_confirmation"} and isinstance(request.get("parsed"), dict)


def _raw_people_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw = parsed.get("people") or parsed.get("staff_entries") or parsed.get("staff") or parsed.get("employees")
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _raw_vehicle_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw = parsed.get("vehicles") or parsed.get("vehicle_entries") or parsed.get("vehicle_assignments")
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _entry_employee_id(entry: dict[str, Any], employees_by_name: dict[str, int]) -> int | None:
    direct = _optional_int(entry.get("staff_order") or entry.get("employee_id") or entry.get("id"))
    if direct is not None:
        return direct
    return employees_by_name.get(_norm_name(entry.get("full_name") or entry.get("name") or entry.get("employee_name")))


def _entry_vehicle_id(entry: dict[str, Any], vehicles_by_name: dict[str, int]) -> int | None:
    direct = _optional_int(entry.get("vehicle_id") or entry.get("id"))
    if direct is not None:
        return direct
    return vehicles_by_name.get(
        _norm_name(entry.get("vehicle_name") or entry.get("vehicle") or entry.get("name") or entry.get("brand_model"))
    )


def _driver_names(
    entry: dict[str, Any],
    employees_by_name: dict[str, int],
    employees_by_id: dict[int, dict[str, Any]],
) -> list[str]:
    raw_drivers = entry.get("drivers")
    if not isinstance(raw_drivers, list):
        raw_drivers = [entry.get("driver") or entry.get("employee_name") or entry.get("assigned_to")]
    result: list[str] = []
    for raw_name in raw_drivers:
        employee_id = employees_by_name.get(_norm_name(raw_name))
        if employee_id is not None:
            name = _employee_name(employees_by_id.get(employee_id, {}))
        else:
            name = str(raw_name or "").strip()
        if name and name not in result:
            result.append(name)
    return result


def _employee_row_id(row: dict[str, Any]) -> int | None:
    return _optional_int(row.get("display_order") or row.get("id") or row.get("employee_id"))


def _vehicle_row_id(row: dict[str, Any]) -> int | None:
    return _optional_int(row.get("id") or row.get("vehicle_id"))


def _employee_name(row: dict[str, Any]) -> str:
    return str(row.get("full_name") or row.get("name") or row.get("employee_name") or "").strip()


def _vehicle_name(row: dict[str, Any]) -> str:
    return str(row.get("brand_model") or row.get("vehicle_name") or row.get("name") or "").strip()


def _final_employee_status(value: Any, *, has_vehicle: bool) -> str:
    raw = str(value or "").strip()
    lowered = raw.casefold()
    if not lowered or lowered == "unknown":
        return "worked" if has_vehicle else "unknown"
    if lowered in {"vacation", "on_leave", "leave", "holiday"}:
        return "vacation"
    if lowered in {"sick"}:
        return "sick"
    if lowered in {"day_off"}:
        return "day_off"
    if lowered in {"not_required"}:
        return "not_required"
    if has_vehicle or lowered in _WORK_STATUS_VALUES:
        return "worked"
    return raw or "unknown"


def _final_vehicle_status(value: Any, *, has_drivers: bool) -> str:
    raw = str(value or "").strip()
    lowered = raw.casefold()
    if lowered in {"idle", "repair", "not_working", "not_required"}:
        return lowered
    if has_drivers or lowered in {"in_use", "working", "work", "worked", "car", "on_car"}:
        return "in_use"
    return raw or "unknown"


_WORK_STATUS_VALUES = {
    "",
    "unknown",
    "office",
    "in_office",
    "at_office",
    "work",
    "worked",
    "working",
    "car",
    "on_car",
    "auto",
    "vehicle",
    "shift",
    "on_shift",
    "object",
    "on_object",
    "site",
    "on_site",
    "field",
    "trip",
}


def _legacy_day_report(report_date: str, request: dict[str, Any] | None) -> dict[str, Any] | None:
    if not request or not isinstance(request.get("parsed"), dict):
        return None
    parsed = request["parsed"]
    employees = _legacy_staff_entries(parsed)
    vehicles = _legacy_vehicle_entries(parsed)
    if not employees and not vehicles:
        return None
    return {
        "report_date": report_date,
        "source": "vehicle_usage_requests.parsed_json",
        "request": {
            "id": request.get("id"),
            "status": request.get("status"),
            "user_id": request.get("user_id"),
            "dialog_id": request.get("dialog_id"),
            "responded_at": request.get("responded_at"),
            "response_text": request.get("response_text"),
        },
        "employee_statuses": employees,
        "vehicle_assignments": vehicles,
        "vehicle_drivers": _legacy_vehicle_drivers(vehicles),
    }


def _legacy_staff_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = parsed.get("people") or parsed.get("staff_entries") or parsed.get("staff") or parsed.get("employees")
    if not isinstance(raw_entries, list):
        return []
    result: list[dict[str, Any]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        name = item.get("full_name") or item.get("name")
        result.append(
            {
                "full_name": str(name or "").strip(),
                "status": str(item.get("status") or "").strip(),
                "vehicle": item.get("vehicle")
                or item.get("vehicle_name")
                or item.get("car_assigned")
                or item.get("car"),
                "notes": str(item.get("notes") or "").strip(),
            }
        )
    return result


def _legacy_vehicle_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entries = parsed.get("vehicles") or parsed.get("vehicle_entries") or parsed.get("vehicle_assignments")
    if not isinstance(raw_entries, list):
        return []
    result: list[dict[str, Any]] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        vehicle = item.get("vehicle_name") or item.get("vehicle") or item.get("name") or item.get("brand_model")
        result.append(
            {
                "vehicle_name": str(vehicle or "").strip(),
                "status": str(item.get("status") or item.get("assignment_status") or "").strip(),
                "drivers": _legacy_drivers(item),
                "notes": str(item.get("notes") or "").strip(),
            }
        )
    return result


def _legacy_drivers(item: dict[str, Any]) -> list[str]:
    raw = item.get("drivers")
    if raw is None:
        raw = item.get("driver") or item.get("employee_name") or item.get("assigned_to")
    if isinstance(raw, list):
        return [str(value).strip() for value in raw if str(value or "").strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    for separator in (";", ","):
        if separator in text:
            return [value.strip() for value in text.split(separator) if value.strip()]
    return [text]


def _legacy_vehicle_drivers(vehicles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for vehicle in vehicles:
        for driver in vehicle.get("drivers") or []:
            result.append({"vehicle_name": vehicle.get("vehicle_name"), "full_name": driver})
    return result


def _merge_legacy_employee_vehicles(
    employees: list[Any],
    legacy_employees: list[dict[str, Any]],
    legacy_vehicles: list[dict[str, Any]],
) -> list[Any]:
    vehicle_by_employee: dict[str, str] = {}
    for item in legacy_employees:
        name = _norm_name(item.get("full_name") or item.get("employee_name") or item.get("name"))
        vehicle = str(item.get("vehicle") or item.get("vehicle_name") or "").strip()
        if name and vehicle:
            vehicle_by_employee[name] = vehicle
    for item in legacy_vehicles:
        vehicle = str(item.get("vehicle_name") or item.get("vehicle") or item.get("name") or "").strip()
        if not vehicle:
            continue
        drivers = item.get("drivers") if isinstance(item.get("drivers"), list) else []
        for driver in drivers:
            name = _norm_name(driver)
            if name:
                vehicle_by_employee.setdefault(name, vehicle)
    if not vehicle_by_employee:
        return employees
    result: list[Any] = []
    for row in employees:
        if not isinstance(row, dict):
            result.append(row)
            continue
        item = dict(row)
        if not item.get("vehicle_name"):
            vehicle = vehicle_by_employee.get(
                _norm_name(item.get("full_name") or item.get("employee_name") or item.get("name"))
            )
            if vehicle:
                item["vehicle_name"] = vehicle
        result.append(item)
    return result


def _legacy_employee_period_rows(employee_name: str, requests: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    wanted = _norm_name(employee_name)
    result: list[dict[str, Any]] = []
    for request in requests:
        if not request or not isinstance(request.get("parsed"), dict):
            continue
        report_date = str(request.get("request_date") or "")
        for item in _legacy_staff_entries(request["parsed"]):
            if wanted and wanted not in _norm_name(item.get("full_name")):
                continue
            result.append(
                {
                    "status_date": report_date,
                    "status": item.get("status") or "unknown",
                    "notes": item.get("notes") or "",
                    "vehicle_name": item.get("vehicle") or "",
                    "source": "vehicle_usage_requests.parsed_json",
                }
            )
    return result


def _legacy_vehicle_period_rows(vehicle_name: str, requests: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    wanted = _norm_name(vehicle_name)
    result: list[dict[str, Any]] = []
    for request in requests:
        if not request or not isinstance(request.get("parsed"), dict):
            continue
        report_date = str(request.get("request_date") or "")
        for item in _legacy_vehicle_entries(request["parsed"]):
            if wanted and wanted not in _norm_name(item.get("vehicle_name")):
                continue
            result.append(
                {
                    "assignment_date": report_date,
                    "status": item.get("status") or "unknown",
                    "drivers": item.get("drivers") or [],
                    "notes": item.get("notes") or "",
                    "source": "vehicle_usage_requests.parsed_json",
                }
            )
    return result


def _group_vehicle_period_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        report_date = str(row.get("assignment_date") or "")
        item = grouped.setdefault(
            report_date,
            {
                "assignment_date": report_date,
                "status": row.get("status") or "unknown",
                "notes": row.get("notes") or "",
                "drivers": [],
            },
        )
        employee_name = str(row.get("employee_name") or "").strip()
        if employee_name and employee_name not in item["drivers"]:
            item["drivers"].append(employee_name)
    return [grouped[key] for key in sorted(grouped)]


def _status_summary(rows: list[dict[str, Any]], status_key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        status = str(row.get(status_key) or "unknown")
        result[status] = result.get(status, 0) + 1
    return result


def _norm_name(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _optional_int(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
