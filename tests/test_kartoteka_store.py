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


def _make_store(**kwargs):
    return PostgresKartotekaStore("postgresql://test/test", **kwargs)


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

    joined = " ".join(executed)
    assert "file_index" in joined
    assert "chunk_id" in joined
    assert "access_level" in joined
    assert "idx_kartoteka_search_text" in joined
    assert "idx_kartoteka_access_level" in joined


# ---------------------------------------------------------------------------
# access level helpers
# ---------------------------------------------------------------------------


def test_can_read_protected_star():
    store = _make_store(protected_user_ids="*")
    assert store._can_read_protected(None) is True
    assert store._can_read_protected(99) is True


def test_can_read_protected_whitelist():
    store = _make_store(protected_user_ids="1,2,3")
    assert store._can_read_protected(1) is True
    assert store._can_read_protected(99) is False
    assert store._can_read_protected(None) is False


def test_can_read_protected_empty():
    store = _make_store()
    assert store._can_read_protected(1) is False


def test_can_read_secret_whitelist():
    store = _make_store(secret_user_ids="5")
    assert store._can_read_secret(5) is True
    assert store._can_read_secret(1) is False


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_passes_allowed_levels_open_only():
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
        result = _run(store.search("договор", user_id=None, limit=3))

    assert result == []
    assert len(captured) == 1
    _, params = captured[0]
    assert params[0] == "%договор%"
    assert params[1] == ["open"]  # no protected access
    assert params[2] == 9  # limit * 3


def test_search_passes_protected_when_allowed():
    store = _make_store(protected_user_ids="*")
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
        _run(store.search("договор", user_id=1, limit=3))

    _, params = captured[0]
    assert params[1] == ["open", "protected"]


def test_search_returns_scored_rows():
    store = _make_store()
    rows = [
        {
            "chunk_id": "abc:0000",
            "relative_path": "docs\\file.docx",
            "filename": "file.docx",
            "extension": ".docx",
            "text": "Договор аренды помещения",
            "access_level": "open",
            "group_name": "Офис",
            "chunk_index": 0,
        }
    ]

    async def _fake_connect():
        cur = MagicMock()
        cur.fetchall = AsyncMock(return_value=rows)
        db = AsyncMock()
        db.__aenter__ = AsyncMock(return_value=db)
        db.__aexit__ = AsyncMock(return_value=False)
        db.execute = AsyncMock(return_value=cur)
        return db

    with patch.object(store, "_connect", _fake_connect):
        result = _run(store.search("договор"))

    assert len(result) == 1
    assert result[0]["filename"] == "file.docx"
    assert "snippet" in result[0]
    assert "договор" in result[0]["snippet"].lower() or result[0]["snippet"] != ""
    assert result[0]["text"] is None  # text field removed, only snippet returned


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_returns_total():
    store = _make_store()

    async def _fake_connect():
        cur = MagicMock()
        cur.fetchone = AsyncMock(return_value={"total_chunks": 7, "total_documents": 3})
        db = AsyncMock()
        db.__aenter__ = AsyncMock(return_value=db)
        db.__aexit__ = AsyncMock(return_value=False)
        db.execute = AsyncMock(return_value=cur)
        return db

    with patch.object(store, "_connect", _fake_connect):
        result = _run(store.stats())

    assert result == {"total_chunks": 7, "total_documents": 3}


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

    assert result == {"total_chunks": 0, "total_documents": 0}
