"""Tests for PostgresKartotekaStore schema and query methods."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import anyio

from ai_server.integrations.postgres.kartoteka_agent import PostgresKartotekaStore

# Capture the real method before conftest's autouse fixture patches it at the class level.
_real_ensure_schema = PostgresKartotekaStore.ensure_schema


def _run(coro):
    async def _r():
        return await coro

    return anyio.run(_r)


def _make_store():
    return PostgresKartotekaStore("postgresql://test/test")


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------


def test_ensure_schema_calls_super_and_creates_table():
    store = _make_store()
    executed: list[str] = []

    async def _fake_connect():
        db = AsyncMock()
        db.__aenter__ = AsyncMock(return_value=db)
        db.__aexit__ = AsyncMock(return_value=False)
        db.execute = AsyncMock(side_effect=lambda sql, *a: executed.append(sql.strip()))
        return db

    with (
        patch.object(PostgresKartotekaStore, "ensure_schema", _real_ensure_schema),
        patch.object(store, "_connect", _fake_connect),
        patch.object(type(store).__bases__[0], "ensure_schema", new=AsyncMock()),
    ):
        _run(store.ensure_schema())

    # Should have called CREATE TABLE and CREATE INDEX
    joined = " ".join(executed)
    assert "file_index" in joined
    assert "idx_kartoteka_search_text" in joined


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_uses_ilike_pattern():
    store = _make_store()
    captured: list[tuple] = []

    async def _fake_connect():
        cur = MagicMock()
        cur.fetchall = AsyncMock(return_value=[])
        db = AsyncMock()
        db.__aenter__ = AsyncMock(return_value=db)
        db.__aexit__ = AsyncMock(return_value=False)

        async def _execute(sql, params=None):
            captured.append((sql, params))
            return cur

        db.execute = _execute
        return db

    with patch.object(store, "_connect", _fake_connect):
        result = _run(store.search("договор", limit=3))

    assert result == []
    assert len(captured) == 1
    _, params = captured[0]
    assert params == ("%договор%", 3)


def test_search_returns_rows():
    store = _make_store()
    rows = [{"id": 1, "path": "/f.pdf", "filename": "f.pdf", "extension": "pdf", "tags": "", "content_preview": ""}]

    async def _fake_connect():
        cur = MagicMock()
        cur.fetchall = AsyncMock(return_value=rows)
        db = AsyncMock()
        db.__aenter__ = AsyncMock(return_value=db)
        db.__aexit__ = AsyncMock(return_value=False)
        db.execute = AsyncMock(return_value=cur)
        return db

    with patch.object(store, "_connect", _fake_connect):
        result = _run(store.search("f"))

    assert result == rows


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_returns_total():
    store = _make_store()

    async def _fake_connect():
        cur = MagicMock()
        cur.fetchone = AsyncMock(return_value={"total": 7})
        db = AsyncMock()
        db.__aenter__ = AsyncMock(return_value=db)
        db.__aexit__ = AsyncMock(return_value=False)
        db.execute = AsyncMock(return_value=cur)
        return db

    with patch.object(store, "_connect", _fake_connect):
        result = _run(store.stats())

    assert result == {"total_files": 7}


def test_stats_returns_zero_when_no_rows():
    store = _make_store()

    async def _fake_connect():
        cur = MagicMock()
        cur.fetchone = AsyncMock(return_value=None)
        db = AsyncMock()
        db.__aenter__ = AsyncMock(return_value=db)
        db.__aexit__ = AsyncMock(return_value=False)
        db.execute = AsyncMock(return_value=cur)
        return db

    with patch.object(store, "_connect", _fake_connect):
        result = _run(store.stats())

    assert result == {"total_files": 0}
