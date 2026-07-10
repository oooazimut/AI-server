"""Tests for PostgreSQL agent stores (business logic with mocked DB connections)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from ai_server.integrations.bitrix.portal_search.types import CONTENT_INDEX_VERSION, CONTENT_TERMINAL_STATUSES
from ai_server.integrations.postgres.bitrix_agent import PostgresBitrixAgentStore
from ai_server.integrations.postgres.orchestrator_agent import PostgresOrchestratorStore
from ai_server.integrations.postgres.pto_agent import PostgresPtoAgentStore
from ai_server.integrations.postgres.vehicle_usage import PostgresVehicleUsageStore, _parse_row
from ai_server.tools.vehicle_usage import StaffMember

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
# PostgresBitrixAgentStore — proposal methods
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


def test_vehicle_store_auto_close_unanswered_day_skips_pending_draft(monkeypatch):
    store = PostgresVehicleUsageStore("postgresql://fake")
    factory, _conn = _sync_conn_factory(
        rows=[
            {
                "id": 10,
                "request_date": "2026-07-16",
                "user_id": 13,
                "dialog_id": "13",
                "status": "pending_clarification",
                "parsed_json": json.dumps({"people": []}),
            }
        ]
    )
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.auto_close_unanswered_day(report_date="2026-07-16", reason="no response")

    assert result["status"] == "skipped"
    assert result["reason"] == "useful_response_exists"
    assert result["request_status"] == "pending_clarification"


def test_bitrix_store_save_proposal(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory(rows=[{"id": 7}])
    monkeypatch.setattr(store, "_sync_connect", factory)

    pid = store.save_proposal(task_id=101, task_title="Задача", missing_parts="исполнитель", responsible_id=9)
    assert pid == 7
    assert any("INSERT INTO bitrix24.incomplete_proposals" in sql for sql, _ in conn.calls)


def test_bitrix_store_get_proposal_by_id(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    row = {"id": 3, "task_id": 101, "status": "awaiting_response"}
    factory, conn = _sync_conn_factory(rows=[row])
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.get_proposal_by_id(3)
    assert result == row


def test_bitrix_store_get_proposal_by_id_missing(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory(rows=[])
    monkeypatch.setattr(store, "_sync_connect", factory)

    assert store.get_proposal_by_id(999) is None


def test_bitrix_store_get_proposals_for_manager(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    rows = [
        {"id": 1, "task_id": 10, "status": "awaiting_response"},
        {"id": 2, "task_id": 11, "status": "proposed"},
    ]
    factory, conn = _sync_conn_factory(rows=rows)
    monkeypatch.setattr(store, "_sync_connect", factory)

    result = store.get_proposals_for_manager()
    assert len(result) == 2
    assert any("awaiting_response" in sql or "proposed" in sql for sql, _ in conn.calls)


def test_bitrix_store_mark_status(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    store.mark_status(3, "proposed")
    assert any("UPDATE bitrix24.incomplete_proposals SET status" in sql for sql, _ in conn.calls)


def test_bitrix_store_delete_proposal(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    store.delete_proposal(3)
    assert any("DELETE FROM bitrix24.incomplete_proposals" in sql for sql, _ in conn.calls)


def test_bitrix_store_update_responsible_response(monkeypatch):
    store = PostgresBitrixAgentStore("postgresql://fake")
    factory, conn = _sync_conn_factory()
    monkeypatch.setattr(store, "_sync_connect", factory)

    store.update_responsible_response(3, "Согласен с дедлайном")
    assert any("responsible_response" in sql for sql, _ in conn.calls)


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
    assert params == (CONTENT_INDEX_VERSION, CONTENT_INDEX_VERSION, list(CONTENT_TERMINAL_STATUSES), 80)


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


def test_pto_store_append_turn_inserts_two_rows():
    store = PostgresPtoAgentStore("postgresql://fake")
    inserted_sqls: list[str] = []

    async def _fake_execute(sql, params=()):
        inserted_sqls.append(str(sql))
        return AsyncMock()

    async_conn = AsyncMock()
    async_conn.__aenter__ = AsyncMock(return_value=async_conn)
    async_conn.__aexit__ = AsyncMock(return_value=False)
    async_conn.execute = _fake_execute

    async def run():
        with patch.object(store, "_connect", AsyncMock(return_value=async_conn)):
            await store.append_turn("dialog:9", "вопрос", "ответ")

    anyio_run(run())

    inserts = [s for s in inserted_sqls if "INSERT INTO" in s]
    assert len(inserts) == 2


def test_pto_store_schema_name():
    store = PostgresPtoAgentStore("postgresql://fake")
    assert store._SCHEMA == "pto"


def test_orchestrator_store_schema_name():
    store = PostgresOrchestratorStore("postgresql://fake")
    assert store._SCHEMA == "internal_orchestrator"


def test_bitrix_store_schema_name():
    store = PostgresBitrixAgentStore("postgresql://fake")
    assert store._SCHEMA == "bitrix24"


def test_vehicle_store_schema_name():
    store = PostgresVehicleUsageStore("postgresql://fake")
    assert store._SCHEMA == "logistics"
