"""Tests for DiagnostSpecialist AgentTools (5 tools, direct .execute() calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import anyio

from ai_server.agents.diagnost.tools import (
    CreateIncidentTool,
    ErrorReportTool,
    GetIncidentTool,
    ListIncidentsTool,
    SearchEventsTool,
)
from ai_server.models import ToolStatus

# ── SearchEventsTool ─────────────────────────────────────────────────────────


def test_search_events_no_store():
    tool = SearchEventsTool(store=None)
    result = anyio.run(tool.execute, {"query": "камера"})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_search_events_empty_query():
    store = AsyncMock()
    tool = SearchEventsTool(store=store)
    result = anyio.run(tool.execute, {"query": ""})
    assert result.status == ToolStatus.INVALID_TOOL_CALL
    store.search_events.assert_not_called()


def test_search_events_ok():
    store = AsyncMock()
    store.search_events = AsyncMock(return_value=[{"event_id": "t1", "request": "текст"}])
    tool = SearchEventsTool(store=store)
    result = anyio.run(tool.execute, {"query": "текст", "limit": 5})
    assert result.status == ToolStatus.OK
    assert result.data["count"] == 1
    assert result.data["results"][0]["event_id"] == "t1"
    store.search_events.assert_awaited_once_with("текст", limit=5)


def test_search_events_clamps_limit():
    store = AsyncMock()
    store.search_events = AsyncMock(return_value=[])
    tool = SearchEventsTool(store=store)
    anyio.run(tool.execute, {"query": "q", "limit": 999})
    store.search_events.assert_awaited_once_with("q", limit=50)


# ── GetIncidentTool ───────────────────────────────────────────────────────────


def test_get_incident_no_store():
    tool = GetIncidentTool(store=None)
    result = anyio.run(tool.execute, {"incident_id": "abc"})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_get_incident_empty_id():
    store = AsyncMock()
    tool = GetIncidentTool(store=store)
    result = anyio.run(tool.execute, {"incident_id": ""})
    assert result.status == ToolStatus.INVALID_TOOL_CALL
    store.get_incident.assert_not_called()


def test_get_incident_not_found():
    store = AsyncMock()
    store.get_incident = AsyncMock(return_value=None)
    tool = GetIncidentTool(store=store)
    result = anyio.run(tool.execute, {"incident_id": "no-such-id"})
    assert result.status == ToolStatus.NOT_FOUND


def test_get_incident_ok():
    incident = {"incident_id": "abc", "reason": "failed", "status": "open"}
    store = AsyncMock()
    store.get_incident = AsyncMock(return_value=incident)
    tool = GetIncidentTool(store=store)
    result = anyio.run(tool.execute, {"incident_id": "abc"})
    assert result.status == ToolStatus.OK
    assert result.data["reason"] == "failed"


# ── ListIncidentsTool ─────────────────────────────────────────────────────────


def test_list_incidents_no_store():
    tool = ListIncidentsTool(store=None)
    result = anyio.run(tool.execute, {})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_list_incidents_ok_all():
    store = AsyncMock()
    store.list_incidents = AsyncMock(return_value=[{"incident_id": "i1"}, {"incident_id": "i2"}])
    tool = ListIncidentsTool(store=store)
    result = anyio.run(tool.execute, {})
    assert result.status == ToolStatus.OK
    assert result.data["count"] == 2
    store.list_incidents.assert_awaited_once_with(status="", limit=20)


def test_list_incidents_filtered():
    store = AsyncMock()
    store.list_incidents = AsyncMock(return_value=[])
    tool = ListIncidentsTool(store=store)
    anyio.run(tool.execute, {"status": "open", "limit": 5})
    store.list_incidents.assert_awaited_once_with(status="open", limit=5)


# ── CreateIncidentTool ────────────────────────────────────────────────────────


def test_create_incident_no_store():
    tool = CreateIncidentTool(store=None)
    result = anyio.run(tool.execute, {"event_id": "ev1"})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_create_incident_empty_event_id():
    store = AsyncMock()
    tool = CreateIncidentTool(store=store)
    result = anyio.run(tool.execute, {"event_id": ""})
    assert result.status == ToolStatus.INVALID_TOOL_CALL
    store.save_incident.assert_not_called()


def test_create_incident_ok():
    store = AsyncMock()
    store.save_incident = AsyncMock(return_value="incident-uuid-123")
    tool = CreateIncidentTool(store=store)
    result = anyio.run(tool.execute, {"event_id": "task-abc", "comment": "подозрительный ответ"})
    assert result.status == ToolStatus.OK
    assert result.data["incident_id"] == "incident-uuid-123"
    assert result.data["event_id"] == "task-abc"
    store.save_incident.assert_awaited_once_with("task-abc", reason="manual", comment="подозрительный ответ")


# ── ErrorReportTool ───────────────────────────────────────────────────────────


def test_error_report_no_store():
    tool = ErrorReportTool(store=None)
    result = anyio.run(tool.execute, {})
    assert result.status == ToolStatus.NOT_CONFIGURED


def test_error_report_ok():
    report = {
        "since_hours": 24,
        "total_incidents": 3,
        "open_incidents": 2,
        "by_reason": [{"reason": "low_confidence", "count": 2}],
        "recent_incidents": [],
    }
    store = AsyncMock()
    store.error_report = AsyncMock(return_value=report)
    tool = ErrorReportTool(store=store)
    result = anyio.run(tool.execute, {"since_hours": 24})
    assert result.status == ToolStatus.OK
    assert result.data["total_incidents"] == 3
    store.error_report.assert_awaited_once_with(since_hours=24)


def test_error_report_clamps_hours():
    store = AsyncMock()
    store.error_report = AsyncMock(return_value={})
    tool = ErrorReportTool(store=store)
    anyio.run(tool.execute, {"since_hours": 9999})
    store.error_report.assert_awaited_once_with(since_hours=720)
