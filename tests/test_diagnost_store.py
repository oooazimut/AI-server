"""Tests for PostgresDiagnostStore — schema creation and query methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import anyio

from ai_server.integrations.postgres.diagnost_agent import PostgresDiagnostStore
from ai_server.models import AgentResult, AgentTask

_real_ensure_schema = PostgresDiagnostStore.ensure_schema


def _run(coro):
    async def _r():
        return await coro

    return anyio.run(_r)


def _store() -> PostgresDiagnostStore:
    return PostgresDiagnostStore("postgresql://test/test")


def _fake_db(executed: list[str] | None = None) -> tuple[AsyncMock, AsyncMock]:
    """Returns (context_manager, db mock)."""
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    if executed is not None:
        db.execute = AsyncMock(side_effect=lambda sql, *a: executed.append(sql.strip()))
    return db, db


# ── ensure_schema ─────────────────────────────────────────────────────────────


def test_ensure_schema_creates_events_and_incidents():
    store = _store()
    executed: list[str] = []
    db, _ = _fake_db(executed)

    with (
        patch.object(PostgresDiagnostStore, "ensure_schema", _real_ensure_schema),
        patch.object(store, "_connect", AsyncMock(return_value=db)),
        patch.object(type(store).__bases__[0], "ensure_schema", new=AsyncMock()),
    ):
        _run(store.ensure_schema())

    joined = " ".join(executed)
    assert "diagnost.events" in joined
    assert "diagnost.incidents" in joined
    assert "idx_diagnost_events_status" in joined
    assert "idx_diagnost_incidents_status" in joined
    assert "diagnost.pending_feedback" in joined
    assert "diagnost.feedback" in joined


# ── save_event ─────────────────────────────────────────────────────────────────


def test_save_event_inserts_row():
    store = _store()
    executed: list[str] = []
    db, _ = _fake_db(executed)

    task = AgentTask(task_id="ev-1", request="запрос")
    result = AgentResult(status="completed", agent_id="internal_orchestrator", confidence=0.85, answer="ответ")

    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        _run(store.save_event(task, result))

    joined = " ".join(executed)
    assert "INSERT INTO diagnost.events" in joined
    assert "ON CONFLICT" in joined


# ── save_incident ──────────────────────────────────────────────────────────────


def test_save_incident_returns_incident_id():
    store = _store()
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.execute = AsyncMock()

    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        incident_id = _run(store.save_incident("ev-1", reason="failed", comment=""))

    assert isinstance(incident_id, str)
    assert len(incident_id) > 0
    db.execute.assert_awaited_once()
    call_sql = db.execute.call_args.args[0]
    assert "INSERT INTO diagnost.incidents" in call_sql


# ── search_events ──────────────────────────────────────────────────────────────


def test_search_events_returns_rows():
    store = _store()
    fake_rows = [{"event_id": "e1", "request": "текст"}]
    cur = AsyncMock()
    cur.fetchall = AsyncMock(return_value=fake_rows)
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.execute = AsyncMock(return_value=cur)

    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        results = _run(store.search_events("текст", limit=5))

    assert results == fake_rows
    call_sql = db.execute.call_args.args[0]
    assert "ILIKE" in call_sql


# ── get_incident ───────────────────────────────────────────────────────────────


def test_get_incident_returns_none_when_not_found():
    store = _store()
    cur = AsyncMock()
    cur.fetchone = AsyncMock(return_value=None)
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.execute = AsyncMock(return_value=cur)

    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        result = _run(store.get_incident("no-such"))

    assert result is None


def test_get_incident_returns_dict():
    store = _store()
    row = {"incident_id": "abc", "reason": "failed", "status": "open"}
    cur = AsyncMock()
    cur.fetchone = AsyncMock(return_value=row)
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.execute = AsyncMock(return_value=cur)

    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        result = _run(store.get_incident("abc"))

    assert result is not None
    assert result["reason"] == "failed"


# ── error_report ───────────────────────────────────────────────────────────────


def test_error_report_returns_aggregated_structure():
    store = _store()
    by_reason_cur = AsyncMock()
    by_reason_cur.fetchall = AsyncMock(return_value=[{"reason": "low_confidence", "cnt": 3}])
    totals_cur = AsyncMock()
    totals_cur.fetchone = AsyncMock(return_value={"total": 3, "open_count": 2})
    recent_cur = AsyncMock()
    recent_cur.fetchall = AsyncMock(return_value=[])
    by_specialist_cur = AsyncMock()
    by_specialist_cur.fetchall = AsyncMock(return_value=[{"agent_id": "bitrix24", "cnt": 2}])
    fb_totals_cur = AsyncMock()
    fb_totals_cur.fetchone = AsyncMock(return_value={"avg_rating": 4.0, "feedback_count": 5})

    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.execute = AsyncMock(side_effect=[by_reason_cur, totals_cur, recent_cur, by_specialist_cur, fb_totals_cur])

    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        report = _run(store.error_report(since_hours=24))

    assert report["since_hours"] == 24
    assert report["total_incidents"] == 3
    assert report["open_incidents"] == 2
    assert report["by_reason"][0]["reason"] == "low_confidence"
    assert report["by_specialist"][0]["agent_id"] == "bitrix24"
    assert report["avg_rating"] == 4.0
    assert report["feedback_count"] == 5
    assert "generated_at" in report


def _fake_db_for_feedback():
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.execute = AsyncMock()
    return db


def test_create_pending_feedback_executes_insert():
    store = _store()
    db = _fake_db_for_feedback()
    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        _run(store.create_pending_feedback("ev-1", "user-1", "dialog-1", channel="bitrix24"))
    sql = db.execute.call_args.args[0]
    assert "INSERT INTO diagnost.pending_feedback" in sql


def test_get_pending_feedback_for_user_returns_row():
    store = _store()
    row = {"id": 1, "event_id": "ev-1", "user_id": "u1", "dialog_key": "dk"}
    cur = AsyncMock()
    cur.fetchone = AsyncMock(return_value=row)
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.execute = AsyncMock(return_value=cur)
    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        result = _run(store.get_pending_feedback_for_user("u1"))
    assert result is not None
    assert result["event_id"] == "ev-1"


def test_get_pending_feedback_for_user_returns_none_when_not_found():
    store = _store()
    cur = AsyncMock()
    cur.fetchone = AsyncMock(return_value=None)
    db = AsyncMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=False)
    db.execute = AsyncMock(return_value=cur)
    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        result = _run(store.get_pending_feedback_for_user("no-user"))
    assert result is None


def test_save_feedback_executes_insert():
    store = _store()
    db = _fake_db_for_feedback()
    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        _run(store.save_feedback("ev-1", "u1", rating=5, raw_text="👍", dialog_key="dk"))
    sql = db.execute.call_args.args[0]
    assert "INSERT INTO diagnost.feedback" in sql


def test_mark_pending_sent_updates_status():
    store = _store()
    db = _fake_db_for_feedback()
    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        _run(store.mark_pending_sent(42))
    sql = db.execute.call_args.args[0]
    assert "status = 'sent'" in sql


def test_mark_pending_received_updates_status():
    store = _store()
    db = _fake_db_for_feedback()
    with patch.object(store, "_connect", AsyncMock(return_value=db)):
        _run(store.mark_pending_received(42))
    sql = db.execute.call_args.args[0]
    assert "status = 'received'" in sql
