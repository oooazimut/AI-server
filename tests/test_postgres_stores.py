"""Tests for PostgreSQL agent stores (business logic with mocked DB connections)."""

from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from ai_server.integrations.bitrix.portal_search.types import CONTENT_INDEX_VERSION, CONTENT_TERMINAL_STATUSES
from ai_server.integrations.postgres.bitrix_agent import PostgresBitrixAgentStore
from ai_server.integrations.postgres.orchestrator_agent import PostgresOrchestratorStore
from ai_server.integrations.postgres.vehicle_usage import PostgresVehicleUsageStore, _parse_row
from ai_server.tools.vehicle_usage import StaffMember

_real_bitrix_ensure_schema = PostgresBitrixAgentStore.ensure_schema

# ---------------------------------------------------------------------------
# Fake psycopg sync connection
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, rows: list[dict] | None = None):
        self._rows = rows or []
        self.rowcount = len(self._rows)

    def fetchone(self) -> dict | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[dict]:
        return list(self._rows)


class _FakeSyncConn:
    """Fake psycopg sync connection used as a context manager."""

    def __init__(self, rows: list[dict] | None = None):
        self._rows = rows or []
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple = ()) -> _FakeCursor:
        self.calls.append((str(sql), params))
        return _FakeCursor(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _SequencedSyncConn:
    """Fake sync connection that returns a different row set for each execute call."""

    def __init__(self, row_sets: list[list[dict] | None]):
        self._row_sets = list(row_sets)
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple = ()) -> _FakeCursor:
        self.calls.append((str(sql), params))
        rows = self._row_sets.pop(0) if self._row_sets else []
        return _FakeCursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _sync_conn_factory(rows=None):
    conn = _FakeSyncConn(rows=rows)
    return lambda: conn, conn


def _sequenced_sync_conn_factory(row_sets):
    conn = _SequencedSyncConn(row_sets=row_sets)
    return lambda: conn, conn


# ---------------------------------------------------------------------------
# _parse_row (pure function)
# ---------------------------------------------------------------------------


def test_parse_row_none():
    assert _parse_row(None) is None


def test_parse_row_no_parsed_json():
    row = {"id": 1, "status": "sent", "parsed_json": None}
    result = _parse_row(row)
    assert result["status"] == "sent"
    assert "parsed" not in result


def test_parse_row_valid_json():
    payload = {"date": "2026-06-05", "people": []}
    row = {"id": 1, "status": "pending_confirmation", "parsed_json": json.dumps(payload)}
    result = _parse_row(row)
    assert result["parsed"] == payload


def test_parse_row_invalid_json():
    row = {"id": 1, "status": "sent", "parsed_json": "{not valid}"}
    result = _parse_row(row)
    assert result["parsed"] is None


# ---------------------------------------------------------------------------
# PostgresVehicleUsageStore — sync methods
# ---------------------------------------------------------------------------


def test_vehicle_store_upsert_employees(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    store.upsert_employees(
        [
            StaffMember(order=1, user_id=15, name="Иван Петров"),
            StaffMember(order=2, user_id=16, name="Олег Сидоров"),
        ]
    )

    insert_calls = [sql for sql, _ in conn.calls if "INSERT INTO logistics.employees" in sql]
    assert len(insert_calls) == 2
    assert any("SET active = FALSE" in sql for sql, _ in conn.calls)


def test_vehicle_store_seed_default_vehicles_is_idempotent():
    store = PostgresVehicleUsageStore("postgresql://fake")
    conn = _FakeSyncConn()

    store._seed_default_vehicles(conn)

    insert_calls = [(sql, params) for sql, params in conn.calls if "INSERT INTO logistics.vehicles" in sql]
    assert len(insert_calls) == 6
    assert [params for _, params in insert_calls] == [
        (1, "Авто 1"),
        (2, "Авто 2"),
        (3, "Авто 3"),
        (4, "Авто 4"),
        (5, "Авто 5"),
        (6, "Авто 6"),
    ]
    assert all("ON CONFLICT (id) DO NOTHING" in sql for sql, _ in insert_calls)


def test_vehicle_store_save_draft(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    factory, conn = _sync_conn_factory(rows=[{"id": 42}])
    monkeypatch.setattr(store, "_sync_connect", factory)

    result_id = store.save_draft(
        request_date="2026-06-05",
        user_id=9,
        dialog_id="chat9",
        response_text="Иван на Ларгусе",
        parsed={"date": "2026-06-05"},
    )

    assert result_id == 42
    assert any("INSERT INTO logistics.vehicle_usage_requests" in sql for sql, _ in conn.calls)


def test_vehicle_store_save_draft_serializes_parsed(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    factory, conn = _sync_conn_factory(rows=[{"id": 1}])
    monkeypatch.setattr(store, "_sync_connect", factory)

    parsed = {"date": "2026-06-05", "people": [{"staff_order": 1}]}
    store.save_draft(request_date="2026-06-05", user_id=9, dialog_id="", response_text="ok", parsed=parsed)

    insert_params = [p for sql, p in conn.calls if "INSERT INTO logistics.vehicle_usage_requests" in sql]
    assert insert_params
    # parsed_json is the 7th param
    parsed_json_str = insert_params[0][6]
    assert json.loads(parsed_json_str) == parsed


def test_vehicle_store_mark_escalated_returns_true(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    factory, conn = _sync_conn_factory(rows=[{"id": 1}])
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.mark_escalated(request_date="2026-06-05", user_id=9, escalated_at="2026-06-05T09:00:00+03:00")
    assert result is True


def test_vehicle_store_mark_escalated_returns_false_when_no_rows(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    factory, conn = _sync_conn_factory(rows=[])
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.mark_escalated(request_date="2099-01-01", user_id=None, escalated_at="2099-01-01T00:00:00+03:00")
    assert result is False


def test_vehicle_store_latest_request_by_dialog_id(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    row = {"id": 5, "status": "sent", "parsed_json": None, "request_date": "2026-06-05"}
    factory, conn = _sync_conn_factory(rows=[row])
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.latest_request(user_id=None, dialog_id="chat9")
    assert result is not None
    assert result["status"] == "sent"
    assert any("dialog_id = %s" in sql for sql, _ in conn.calls)


def test_vehicle_store_latest_request_none_when_no_user_or_dialog(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    result = store.latest_request(user_id=None, dialog_id="")
    assert result is None


def test_vehicle_store_replace_day_report(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    store.replace_day_report(
        status_date="2026-06-05",
        employee_statuses=[(1, "shift", ""), (2, "day_off", "")],
        vehicle_assignments=[(1, 1, ""), (2, None, "")],
    )

    delete_calls = [sql for sql, _ in conn.calls if "DELETE FROM logistics" in sql]
    assert any("DELETE FROM logistics.employee_daily_statuses" in sql for sql in delete_calls)
    assert any("DELETE FROM logistics.vehicle_daily_assignments" in sql for sql in delete_calls)
    assert any("DELETE FROM logistics.vehicle_daily_drivers" in sql for sql in delete_calls)


def test_vehicle_store_get_day_report_falls_back_to_legacy_parsed_json(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    parsed = {
        "staff_entries": [
            {"full_name": "Борисов Андрей", "status": "на авто", "vehicle": "Авто 2"},
        ],
        "vehicle_entries": [
            {"vehicle": "Авто 2", "status": "в работе", "driver": ["Борисов Андрей", "Карасев Алексей"]},
        ],
    }
    row_sets = [
        [],
        [],
        [],
        [
            {
                "id": 97,
                "request_date": "2026-07-02",
                "status": "answered",
                "parsed_json": json.dumps(parsed),
                "response_text": "Авто 2 — Борисов Андрей, Карасев Алексей",
            }
        ],
    ]
    factory, conn = _sequenced_sync_conn_factory(row_sets)
    monkeypatch.setattr(store, "_sync_connect", factory)

    report = store.get_day_report(report_date="2026-07-02")

    assert report["source"] == "vehicle_usage_requests.parsed_json"
    assert report["employee_statuses"][0]["full_name"] == "Борисов Андрей"
    assert report["vehicle_assignments"][0]["vehicle_name"] == "Авто 2"
    assert report["vehicle_assignments"][0]["drivers"] == ["Борисов Андрей", "Карасев Алексей"]
    assert any("vehicle_usage_requests" in sql for sql, _ in conn.calls)


# ---------------------------------------------------------------------------
# PostgresBitrixAgentStore
# ---------------------------------------------------------------------------


def test_vehicle_store_get_day_report_supplements_missing_vehicles_from_legacy(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    parsed = {
        "staff_entries": [
            {"full_name": "Borisov Andrey", "status": "worked", "vehicle": "Auto 2"},
        ],
        "vehicle_entries": [
            {"vehicle": "Auto 2", "status": "in_use", "drivers": ["Borisov Andrey", "Karasev Alexey"]},
        ],
    }
    row_sets = [
        [
            {
                "status_date": "2026-07-06",
                "employee_id": 1,
                "full_name": "Borisov Andrey",
                "status": "worked",
                "vehicle_name": None,
                "notes": "",
            }
        ],
        [],
        [],
        [
            {
                "id": 98,
                "request_date": "2026-07-06",
                "status": "answered",
                "parsed_json": json.dumps(parsed),
                "response_text": "Auto 2 - Borisov Andrey, Karasev Alexey / in_use",
            }
        ],
    ]
    factory, conn = _sequenced_sync_conn_factory(row_sets)
    monkeypatch.setattr(store, "_sync_connect", factory)

    report = store.get_day_report(report_date="2026-07-06")

    assert report["source"] == "normalized_tables+vehicle_usage_requests.parsed_json"
    assert report["employee_statuses"][0]["vehicle_name"] == "Auto 2"
    assert report["vehicle_assignments"][0]["vehicle_name"] == "Auto 2"
    assert report["vehicle_drivers"][0] == {"vehicle_name": "Auto 2", "full_name": "Borisov Andrey"}
    assert any("vehicle_usage_requests" in sql for sql, _ in conn.calls)


def test_vehicle_store_finalize_pending_unknowns_completes_draft(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    parsed = {
        "people": [{"staff_order": 1, "full_name": "Borisov Andrey", "status": "worked"}],
        "vehicles": [{"vehicle_id": 2, "vehicle_name": "Auto 2", "status": "in_use", "drivers": ["Borisov Andrey"]}],
    }
    factory, _conn = _sync_conn_factory(
        rows=[
            {
                "id": 10,
                "request_date": "2026-07-16",
                "user_id": 13,
                "dialog_id": "13",
                "status": "pending_clarification",
                "response_text": "Borisov Auto 2",
                "parsed_json": json.dumps(parsed),
            }
        ]
    )
    monkeypatch.setattr(store, "_sync_connect", factory)
    monkeypatch.setattr(
        store,
        "staff_roster",
        lambda: [
            {"display_order": 1, "full_name": "Borisov Andrey", "user_id": 27},
            {"display_order": 2, "full_name": "Karasev Alexey", "user_id": 28},
        ],
    )
    monkeypatch.setattr(
        store,
        "vehicles",
        lambda: [
            {"id": 2, "brand_model": "Auto 2", "registration_number": ""},
            {"id": 5, "brand_model": "Auto 5", "registration_number": ""},
        ],
    )
    saved: dict = {}
    replaced: dict = {}

    def _save_draft(**kwargs):
        saved.update(kwargs)
        return 55

    def _replace_day_report(**kwargs):
        replaced.update(kwargs)

    monkeypatch.setattr(store, "save_draft", _save_draft)
    monkeypatch.setattr(store, "replace_day_report", _replace_day_report)

    result = store.finalize_pending_unknowns(report_date="2026-07-16", reason="cutoff")

    assert result["status"] == "finalized_unknown"
    assert saved["status"] == "answered"
    assert saved["parsed"]["auto_completed_unknown"] is True
    assert saved["parsed"]["people"][1]["full_name"] == "Karasev Alexey"
    assert saved["parsed"]["people"][1]["status"] == "unknown"
    assert saved["parsed"]["vehicles"][1]["vehicle_name"] == "Auto 5"
    assert saved["parsed"]["vehicles"][1]["status"] == "unknown"
    assert replaced["employee_statuses"] == [(1, "worked", ""), (2, "unknown", "cutoff")]
    assert replaced["vehicle_assignments"] == [(2, 1, "in_use", ""), (5, None, "unknown", "cutoff")]


def test_vehicle_store_auto_close_unanswered_day_uses_operator(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    factory, _conn = _sync_conn_factory(
        rows=[
            {
                "id": 10,
                "request_date": "2026-07-16",
                "user_id": 13,
                "dialog_id": "13",
                "status": "sent",
                "parsed_json": None,
            }
        ]
    )
    monkeypatch.setattr(store, "_sync_connect", factory)
    monkeypatch.setattr(store, "vehicle_usage_operator_ids", lambda: {13})
    cancelled: dict = {}

    def _cancel_day_report(**kwargs):
        cancelled.update(kwargs)
        return 77

    monkeypatch.setattr(store, "cancel_day_report", _cancel_day_report)

    result = store.auto_close_unanswered_day(report_date="2026-07-16", reason="no response")

    assert result == {"status": "closed_day_off", "report_date": "2026-07-16", "request_id": 77, "user_id": 13}
    assert cancelled == {"report_date": "2026-07-16", "user_id": 13, "dialog_id": "13", "reason": "no response"}


def test_vehicle_store_auto_close_unanswered_day_keeps_scheduled_pending_draft(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    factory, _conn = _sync_conn_factory(
        rows=[
            {
                "id": 10,
                "request_date": "2026-07-16",
                "user_id": 13,
                "dialog_id": "13",
                "status": "pending_clarification",
                "source": "scheduled",
                "parsed_json": json.dumps({"people": []}),
            }
        ]
    )
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.auto_close_unanswered_day(report_date="2026-07-16", reason="no response")

    assert result["status"] == "skipped"
    assert result["reason"] == "useful_response_exists"
    assert result["request_status"] == "pending_clarification"


def test_vehicle_store_auto_close_manual_pending_draft_finalizes_unknowns(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    pending = {
        "id": 10,
        "request_date": "2026-07-16",
        "user_id": 13,
        "dialog_id": "13",
        "status": "pending_clarification",
        "source": "manual",
        "parsed_json": json.dumps({"people": []}),
    }
    factory, _conn = _sync_conn_factory(rows=[pending])
    monkeypatch.setattr(store, "_sync_connect", factory)
    called = {}

    def fake_finalize(**kwargs):
        called.update(kwargs)
        return {"status": "finalized_unknown", "report_date": kwargs["report_date"]}

    monkeypatch.setattr(store, "finalize_pending_unknowns", fake_finalize)

    result = store.auto_close_unanswered_day(report_date="2026-07-16", reason="no response")

    assert result["status"] == "finalized_unknown"
    assert result["reason"] == "pending_draft_finalized_at_day_close"
    assert called["report_date"] == "2026-07-16"
    assert called["actor_user_id"] == 13


def test_bitrix_store_upserts_task_close_controlled_user_with_stable_controlled_from(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    store.upsert_task_close_controlled_user(user_id=231, active=True, updated_by=1)

    sql, params = conn.calls[0]
    assert "task_close_controlled_users" in sql
    assert "controlled_from" in sql
    assert "controlled_from IS NULL" in sql
    assert params == (231, True, True, 1)


def test_bitrix_store_schema_backfills_date_only_control_cutoff():
    source = inspect.getsource(_real_bitrix_ensure_schema)
    assert "length(setting.value) = 10" in source
    assert "setting.value::date::timestamp AT TIME ZONE 'Europe/Moscow'" in source
    assert "($|[T ])" in source


def test_bitrix_store_updates_one_task_close_operator_without_replacing_list(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    store.upsert_task_close_operator(user_id=231, active=True, updated_by=1)

    sql, params = conn.calls[0]
    assert "INSERT INTO bitrix24.task_close_control_operators" in sql
    assert "UPDATE bitrix24.task_close_control_operators\n                SET active = FALSE" not in sql
    assert params == (231, True, 1)


def test_bitrix_store_set_task_close_setting_records_revision(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    store.set_task_close_control_setting(key="auto_close_time", value="19:30", updated_by=1)

    assert any("INSERT INTO bitrix24.task_close_control_settings" in sql for sql, _ in conn.calls)
    revision_calls = [params for sql, params in conn.calls if "task_close_control_revisions" in sql]
    assert revision_calls
    assert revision_calls[0][0] == 1
    assert revision_calls[0][1] == "set_setting"
    assert json.loads(revision_calls[0][2]) == {"key": "auto_close_time", "value": "19:30"}


def test_bitrix_store_task_close_operator_ids(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory(rows=[{"user_id": 13}, {"user_id": 15}])
    monkeypatch.setattr(store, "_sync_connect", factory)

    assert store.task_close_operator_ids() == {13, 15}
    assert "task_close_control_operators" in conn.calls[0][0]


def test_bitrix_store_set_task_close_operators_does_not_control_operators(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    saved = store.set_task_close_operators(operator_user_ids=[15, 13, 13], actor_user_id=1)

    assert saved == [13, 15]
    assert any("UPDATE bitrix24.task_close_control_operators" in sql for sql, _ in conn.calls)
    assert sum("INSERT INTO bitrix24.task_close_control_operators" in sql for sql, _ in conn.calls) == 2
    assert sum("INSERT INTO bitrix24.task_close_controlled_users" in sql for sql, _ in conn.calls) == 0
    revision_calls = [params for sql, params in conn.calls if "task_close_control_revisions" in sql]
    assert revision_calls
    assert revision_calls[0][1] == "set_operators"
    assert json.loads(revision_calls[0][2]) == {"operator_user_ids": [13, 15]}


def test_bitrix_store_task_close_controlled_user_ids(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory(rows=[{"user_id": 13}, {"user_id": 15}])
    monkeypatch.setattr(store, "_sync_connect", factory)

    assert store.task_close_controlled_user_ids() == {13, 15}
    assert "task_close_controlled_users" in conn.calls[0][0]


def test_bitrix_store_get_task_close_control_event_parses_payload(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    row = {
        "task_id": 8875,
        "close_event_key": "closed_at:2026-07-12T12:00:00+03:00",
        "responsible_id": 231,
        "closed_by_user_id": 231,
        "closed_at": "2026-07-12T12:00:00+03:00",
        "decision": "controlled",
        "reason": "controlled_at_close_time",
        "payload_json": json.dumps({"source": "sync"}),
        "seen_at": "2026-07-12T12:01:00+03:00",
        "updated_at": "2026-07-12T12:01:00+03:00",
    }
    factory, conn = _sync_conn_factory(rows=[row])
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.get_task_close_control_event(
        task_id=8875,
        close_event_key="closed_at:2026-07-12T12:00:00+03:00",
    )

    assert result is not None
    assert result["decision"] == "controlled"
    assert result["payload"] == {"source": "sync"}
    assert conn.calls[0][1] == (8875, "closed_at:2026-07-12T12:00:00+03:00")


def test_bitrix_store_upserts_task_close_control_event(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    store.upsert_task_close_control_event(
        task_id=8875,
        close_event_key="closed_at:2026-07-12T12:00:00+03:00",
        decision="ignored_user_not_controlled_at_close",
        reason="user_not_controlled_at_close_time",
        closed_at="2026-07-12T12:00:00+03:00",
        responsible_id=231,
        closed_by_user_id=231,
        payload={"status": 5},
    )

    sql, params = conn.calls[0]
    assert "task_close_control_events" in sql
    assert params[:7] == (
        8875,
        "closed_at:2026-07-12T12:00:00+03:00",
        231,
        231,
        "2026-07-12T12:00:00+03:00",
        "ignored_user_not_controlled_at_close",
        "user_not_controlled_at_close_time",
    )
    assert json.loads(params[7]) == {"status": 5}


def test_bitrix_store_lists_task_close_processing_states(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    rows = [
        {
            "task_id": 8875,
            "state_key": "direct_close:event-a",
            "status": "pending_direct_close",
            "payload_json": json.dumps({"responsible_id": 231, "dialog_key": "chat231"}),
            "actor_user_id": None,
            "created_at": "2026-07-12T12:00:00+03:00",
            "updated_at": "2026-07-12T12:00:00+03:00",
        }
    ]
    factory, conn = _sync_conn_factory(rows=rows)
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.list_task_close_processing_states(
        statuses=["pending_direct_close"],
        state_key_prefix="direct_close:",
        responsible_id=231,
        dialog_key="chat231",
        limit=50,
    )

    assert result[0]["payload"] == {"responsible_id": 231, "dialog_key": "chat231"}
    sql, params = conn.calls[0]
    assert "task_close_processing_state" in sql
    assert "status = ANY" in sql
    assert "state_key LIKE" in sql
    assert "payload_json::jsonb ->> 'responsible_id'" in sql
    assert "payload_json::jsonb ->> 'dialog_key'" in sql
    assert params == (["pending_direct_close"], "direct\\_close:%", "231", "chat231", 50)


def test_bitrix_store_lists_task_close_drafts(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    rows = [
        {
            "dialog_key": "dialog:231:user:231",
            "params_json": json.dumps({"_draft_type": "task_close", "task_id": 8875}),
            "status": "active",
            "created_at": "2026-07-12T19:00:00+03:00",
            "updated_at": "2026-07-12T19:00:00+03:00",
        },
        {
            "dialog_key": "dialog:231:user:231:task-create",
            "params_json": json.dumps({"_draft_type": "task_create", "title": "skip"}),
            "status": "active",
            "created_at": "2026-07-12T19:01:00+03:00",
            "updated_at": "2026-07-12T19:01:00+03:00",
        },
    ]
    factory, conn = _sync_conn_factory(rows=rows)
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.list_task_drafts(draft_type="task_close", limit=50)

    assert result == [
        {
            "dialog_key": "dialog:231:user:231",
            "params": {"_draft_type": "task_close", "task_id": 8875},
            "status": "active",
            "created_at": "2026-07-12T19:00:00+03:00",
            "updated_at": "2026-07-12T19:00:00+03:00",
        }
    ]
    sql, params = conn.calls[0]
    assert "interactive_drafts" in sql
    assert params == (50,)


def test_bitrix_store_content_candidates_filters_completed_items(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    rows = [
        {
            "entity_type": "disk_file",
            "entity_id": "501",
            "title": "pending.txt",
            "body": "",
            "url": "",
            "metadata_json": "{}",
        }
    ]
    factory, conn = _sync_conn_factory(rows=rows)
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.content_candidates(limit=80)

    assert result[0].entity_id == "501"
    sql, params = conn.calls[0]
    assert "COALESCE" in sql
    assert "content_index_status" in sql
    assert "content_index_version" in sql
    assert "ANY" in sql
    assert params == (
        CONTENT_INDEX_VERSION,
        CONTENT_INDEX_VERSION,
        sorted({*CONTENT_TERMINAL_STATUSES, "unsupported"}),
        80,
    )


# ---------------------------------------------------------------------------
# PostgresAgentSchema — async dialog_history methods
# ---------------------------------------------------------------------------


def anyio_run(coro):
    import anyio

    async def _runner():
        return await coro

    return anyio.run(_runner)


def _make_async_conn(rows=None):
    """Return an async context manager mock that returns a fake async connection."""
    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    cur = AsyncMock()
    cur.fetchall = AsyncMock(return_value=rows or [])
    conn.execute = AsyncMock(return_value=cur)
    return conn


def test_orchestrator_store_load_turns_empty():
    store = PostgresOrchestratorStore("postgresql://fake")
    async_conn = _make_async_conn(rows=[])

    async def run():
        with patch.object(store, "_connect", AsyncMock(return_value=async_conn)):
            return await store.load_turns("dialog:9")

    rows = anyio_run(run())
    assert rows == []


def test_orchestrator_store_load_turns_returns_history():
    store = PostgresOrchestratorStore("postgresql://fake")
    fake_rows = [{"role": "user", "content": "привет"}, {"role": "assistant", "content": "здравствуй"}]
    async_conn = _make_async_conn(rows=fake_rows)

    async def run():
        with patch.object(store, "_connect", AsyncMock(return_value=async_conn)):
            return await store.load_turns("dialog:9")

    turns = anyio_run(run())
    assert turns == [{"role": "user", "content": "привет"}, {"role": "assistant", "content": "здравствуй"}]


def test_orchestrator_store_schema_name():
    store = PostgresOrchestratorStore("postgresql://fake")
    assert store._SCHEMA == "internal_orchestrator"


def test_bitrix_store_schema_name():
    store = PostgresBitrixAgentStore("postgresql://fake")
    assert store._SCHEMA == "bitrix24"


def test_bitrix_store_task_draft_edit_preserves_identity_and_original_expiry():
    store = PostgresBitrixAgentStore("postgresql://fake")
    created_at = datetime.now(UTC)
    expires_at = created_at + timedelta(minutes=15)
    active_cursor = AsyncMock()
    active_cursor.fetchone = AsyncMock(
        return_value={
            "draft_id": "draft-1",
            "draft_type": "task_create",
            "original_request": "исходный текст",
            "params_json": json.dumps({"_draft_type": "task_create"}),
            "status": "active",
            "version": 2,
            "created_at": created_at,
            "expires_at": expires_at,
        }
    )
    default_cursor = AsyncMock()
    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock(side_effect=[active_cursor, default_cursor, default_cursor, default_cursor])

    async def run():
        with patch.object(store, "_connect", AsyncMock(return_value=conn)):
            await store.save_task_draft(
                "dialog:13",
                {
                    "_draft_type": "task_create",
                    "_original_request": "изменённый текст",
                    "_draft_user_id": 13,
                    "fields": {"TITLE": "Тест"},
                },
            )

    anyio_run(run())
    insert_call = next(
        call for call in conn.execute.await_args_list if "INSERT INTO bitrix24.interactive_drafts" in call.args[0]
    )
    params = insert_call.args[1]
    payload = json.loads(params[6])
    assert params[0] == "draft-1"
    assert params[7] == 3
    assert params[8] == created_at
    assert params[10] == expires_at
    assert payload["_draft_id"] == "draft-1"
    assert payload["_draft_version"] == 3
    assert payload["_draft_created_at"] == created_at.isoformat()
    assert payload["_draft_expires_at"] == expires_at.isoformat()
    assert payload["_original_request"] == "исходный текст"
    transition_call = next(
        call
        for call in conn.execute.await_args_list
        if "INSERT INTO bitrix24.interactive_draft_transitions" in call.args[0]
    )
    transition_payload = json.loads(transition_call.args[1][3])
    assert transition_payload["_original_request"] == "исходный текст"
    assert transition_payload["_edit_request"] == "изменённый текст"


def test_bitrix_store_does_not_promote_expired_legacy_draft_without_ttl_argument():
    store = PostgresBitrixAgentStore("postgresql://fake")
    no_typed_cursor = AsyncMock()
    no_typed_cursor.fetchone = AsyncMock(return_value=None)
    legacy_cursor = AsyncMock()
    legacy_cursor.fetchone = AsyncMock(
        return_value={
            "params_json": json.dumps({"_draft_type": "task_create", "fields": {"TITLE": "old"}}),
            "created_at": datetime(2020, 1, 1, tzinfo=UTC),
        }
    )
    read_conn = AsyncMock()
    read_conn.__aenter__ = AsyncMock(return_value=read_conn)
    read_conn.__aexit__ = AsyncMock(return_value=False)
    read_conn.execute = AsyncMock(side_effect=[no_typed_cursor, legacy_cursor])
    delete_conn = AsyncMock()
    delete_conn.__aenter__ = AsyncMock(return_value=delete_conn)
    delete_conn.__aexit__ = AsyncMock(return_value=False)
    delete_conn.execute = AsyncMock(return_value=AsyncMock())

    async def run():
        with patch.object(store, "_connect", AsyncMock(side_effect=[read_conn, delete_conn])):
            return await store.get_task_draft("dialog:13")

    result = anyio_run(run())
    assert result is None
    sql, params = delete_conn.execute.await_args.args
    assert "DELETE FROM bitrix24.pending_task_draft" in sql
    assert params == ("dialog:13", datetime(2020, 1, 1, tzinfo=UTC))


def test_bitrix_store_claims_exact_task_draft_once_with_cas():
    store = PostgresBitrixAgentStore("postgresql://fake")
    payload = {"_draft_id": "draft-1", "_draft_version": 3, "_draft_type": "task_create"}
    claimed_cursor = AsyncMock()
    claimed_cursor.fetchone = AsyncMock(
        return_value={"draft_id": "draft-1", "params_json": json.dumps(payload), "user_id": 13}
    )
    default_cursor = AsyncMock()
    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock(side_effect=[claimed_cursor, default_cursor])

    async def run():
        with patch.object(store, "_connect", AsyncMock(return_value=conn)):
            return await store.claim_task_draft(
                "dialog:13",
                expected_draft_id="draft-1",
                expected_version=3,
                expected_type="task_create",
            )

    result = anyio_run(run())
    assert {key: result[key] for key in payload} == payload
    assert result["_draft_claim_token"]
    sql, params = conn.execute.await_args_list[0].args
    assert "status = 'confirming'" in sql
    assert "version = %s" in sql
    assert params[3:7] == ("dialog:13", "draft-1", 3, "task_create")


def test_bitrix_store_confirms_admin_change_in_one_locked_transaction():
    store = PostgresBitrixAgentStore("postgresql://fake")
    draft = {
        "_draft_id": "draft-admin-1",
        "_draft_version": 2,
        "_draft_type": "admin_change",
        "action": "add_operator",
        "field": "operator",
        "target_user_id": 13,
        "old_value": False,
        "new_value": True,
    }
    draft_cursor = AsyncMock()
    draft_cursor.fetchone = AsyncMock(return_value={"params_json": json.dumps(draft)})
    member_cursor = AsyncMock()
    member_cursor.fetchone = AsyncMock(return_value=None)
    default_cursor = AsyncMock()
    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock(
        side_effect=[
            draft_cursor,
            default_cursor,
            member_cursor,
            default_cursor,
            default_cursor,
            default_cursor,
            default_cursor,
            default_cursor,
        ]
    )

    async def run():
        with patch.object(store, "_connect", AsyncMock(return_value=conn)):
            return await store.confirm_admin_change_draft(
                dialog_key="dialog:1",
                draft_id="draft-admin-1",
                draft_version=2,
                actor_user_id=1,
            )

    result = anyio_run(run())
    assert result["status"] == "confirmed"
    sql_calls = [call.args[0] for call in conn.execute.await_args_list]
    assert "FOR UPDATE" in sql_calls[0]
    assert "pg_advisory_xact_lock" in sql_calls[1]
    assert "task_close_control_operators" in sql_calls[2] and "FOR UPDATE" in sql_calls[2]
    assert "INSERT INTO bitrix24.task_close_control_operators" in sql_calls[3]
    assert "INSERT INTO bitrix24.task_close_control_revisions" in sql_calls[4]
    assert "status = 'confirmed'" in sql_calls[5]
    assert "interactive_draft_transitions" in sql_calls[6]
    assert "pending_task_draft" in sql_calls[7]


def test_bitrix_store_reclaims_only_stale_finalizing_draft_with_new_lease_token():
    store = PostgresBitrixAgentStore("postgresql://fake")
    payload = {"_draft_id": "draft-1", "_draft_version": 3, "_draft_type": "task_close"}
    claimed_cursor = AsyncMock()
    claimed_cursor.fetchone = AsyncMock(
        return_value={"draft_id": "draft-1", "params_json": json.dumps(payload), "user_id": 13}
    )
    default_cursor = AsyncMock()
    conn = AsyncMock()
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)
    conn.execute = AsyncMock(side_effect=[claimed_cursor, default_cursor])

    async def run():
        with patch.object(store, "_connect", AsyncMock(return_value=conn)):
            return await store.reclaim_stale_finalizing_task_draft(
                "dialog:13",
                expected_draft_id="draft-1",
                expected_version=3,
                expected_type="task_close",
                lease_seconds=300,
            )

    result = anyio_run(run())
    assert result["_reclaimed_finalizing"] is True
    assert result["_draft_claim_token"]
    sql, params = conn.execute.await_args_list[0].args
    assert "status = 'finalizing'" in sql
    assert "COALESCE(claimed_at, updated_at) <= %s" in sql
    assert params[3:7] == ("dialog:13", "draft-1", 3, "task_close")


def test_vehicle_store_auto_close_pending_draft_finalizes_unknowns(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    pending = {
        "id": 7,
        "request_date": "2026-07-15",
        "user_id": 13,
        "dialog_id": "13",
        "status": "pending_clarification",
        "response_text": "частичный отчет",
        "source": "manual",
        "parsed_json": json.dumps({"date": "2026-07-15", "people": [], "vehicles": []}),
    }
    factory, conn = _sync_conn_factory(rows=[pending])
    monkeypatch.setattr(store, "_sync_connect", factory)
    called = {}

    def fake_finalize(**kwargs):
        called.update(kwargs)
        return {"status": "finalized_unknown", "report_date": kwargs["report_date"]}

    monkeypatch.setattr(store, "finalize_pending_unknowns", fake_finalize)

    result = store.auto_close_unanswered_day(
        report_date="2026-07-15",
        reason="No complete vehicle usage response was received by cutoff.",
    )

    assert result["status"] == "finalized_unknown"
    assert result["reason"] == "pending_draft_finalized_at_day_close"
    assert called["report_date"] == "2026-07-15"
    assert called["actor_user_id"] == 13
    sql, params = conn.calls[0]
    assert "FROM logistics.vehicle_usage_requests" in sql
    assert params == ("2026-07-15",)


def test_vehicle_store_schema_name():
    store = PostgresVehicleUsageStore("postgresql://fake")
    assert store._SCHEMA == "logistics"
