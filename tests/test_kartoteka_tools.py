"""Tests for KartotekaSearchTool, KartotekaContextTool and file_ops tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import anyio

from ai_server.agents.kartoteka.tools import (
    FileAddTool,
    FileDeleteTool,
    FileMoveTool,
    KartotekaContextTool,
    KartotekaSearchTool,
)
from ai_server.models import ToolStatus


def _exec(tool, args, *, user_id=None):
    async def _run():
        return await tool.execute(args, user_id=user_id)

    return anyio.run(_run)


def _make_store(*, search_results=None, stats=None):
    store = MagicMock()
    store.search = AsyncMock(return_value=search_results or [])
    store.stats = AsyncMock(return_value=stats or {"total_files": 0})
    return store


# ---------------------------------------------------------------------------
# KartotekaSearchTool
# ---------------------------------------------------------------------------


def test_search_tool_no_store():
    tool = KartotekaSearchTool(store=None)
    result = _exec(tool, {"query": "договор"})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_search_tool_empty_query():
    tool = KartotekaSearchTool(store=_make_store())
    result = _exec(tool, {"query": "  "})
    assert result.status == ToolStatus.INVALID_TOOL_CALL
    assert "query" in result.error


def test_search_tool_missing_query():
    tool = KartotekaSearchTool(store=_make_store())
    result = _exec(tool, {})
    assert result.status == ToolStatus.INVALID_TOOL_CALL


def test_search_tool_ok_no_results():
    tool = KartotekaSearchTool(store=_make_store(search_results=[]))
    result = _exec(tool, {"query": "несуществующий файл"})
    assert result.status == ToolStatus.OK
    assert result.data["count"] == 0
    assert result.data["results"] == []


def test_search_tool_ok_with_results():
    files = [
        {
            "chunk_id": "abc:0000",
            "relative_path": "docs\\contract.pdf",
            "filename": "contract.pdf",
            "extension": ".pdf",
            "snippet": "текст договора",
            "access_level": "open",
            "group_name": "Офис",
        },
    ]
    store = _make_store(search_results=files)
    tool = KartotekaSearchTool(store=store)
    result = _exec(tool, {"query": "договор"})
    assert result.status == ToolStatus.OK
    assert result.data["count"] == 1
    assert result.data["results"][0]["filename"] == "contract.pdf"


def test_search_tool_limit_clamped():
    store = _make_store()
    tool = KartotekaSearchTool(store=store)
    _exec(tool, {"query": "doc", "limit": 999})
    store.search.assert_awaited_once_with("doc", user_id=None, limit=20)


def test_search_tool_default_limit():
    store = _make_store()
    tool = KartotekaSearchTool(store=store)
    _exec(tool, {"query": "doc"})
    store.search.assert_awaited_once_with("doc", user_id=None, limit=5)


# ---------------------------------------------------------------------------
# KartotekaContextTool
# ---------------------------------------------------------------------------


def test_context_tool_no_store():
    tool = KartotekaContextTool(store=None)
    result = _exec(tool, {})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_context_tool_ok():
    store = _make_store(stats={"total_chunks": 42, "total_documents": 10})
    tool = KartotekaContextTool(store=store)
    result = _exec(tool, {})
    assert result.status == ToolStatus.OK
    assert result.data["total_chunks"] == 42
    assert result.data["total_documents"] == 10


# ---------------------------------------------------------------------------
# FileAddTool / FileDeleteTool / FileMoveTool — всегда NOT_CONFIGURED
# ---------------------------------------------------------------------------


def test_file_add_not_configured():
    result = _exec(FileAddTool(store=None), {"path": "/a", "filename": "a.pdf"})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_file_add_not_configured_with_store():
    result = _exec(FileAddTool(store=_make_store()), {"path": "/a", "filename": "a.pdf"})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_file_delete_not_configured():
    result = _exec(FileDeleteTool(store=None), {"path": "/a"})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_file_move_not_configured():
    result = _exec(FileMoveTool(store=None), {"old_path": "/a", "new_path": "/b"})
    assert result.status == ToolStatus.NOT_CONFIGURED
