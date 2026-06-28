import asyncio

from ai_server.models import ToolStatus
from ai_server.orchestrators.tools.manage_suspended import ManageSuspendedTool
from tests.fakes import FakeOrchestratorStore


def _make_tool(store=None) -> ManageSuspendedTool:
    return ManageSuspendedTool(store=store)


def test_list_empty_when_no_suspended():
    store = FakeOrchestratorStore()
    tool = _make_tool(store)

    result = asyncio.run(tool.execute({"action": "list"}, dialog_key="dlg1"))

    assert result.status == ToolStatus.OK
    assert result.data["suspended_specialists"] == []


def test_close_removes_entry():
    import json

    store = FakeOrchestratorStore()
    store._kv[("dlg1", "suspended_specialists")] = json.dumps(["bitrix24", "pto"])
    tool = _make_tool(store)

    result = asyncio.run(tool.execute({"action": "close", "specialist_id": "bitrix24"}, dialog_key="dlg1"))

    assert result.status == ToolStatus.OK
    assert result.data["closed"] == "bitrix24"
    assert result.data["remaining_suspended"] == ["pto"]
    remaining = asyncio.run(store.get_kv("dlg1", "suspended_specialists"))
    assert "bitrix24" not in remaining


def test_close_nonexistent_is_idempotent():
    store = FakeOrchestratorStore()
    tool = _make_tool(store)

    result = asyncio.run(tool.execute({"action": "close", "specialist_id": "bitrix24"}, dialog_key="dlg1"))

    assert result.status == ToolStatus.OK
    assert result.data["closed"] == "bitrix24"


def test_resume_sets_pending():
    import json

    store = FakeOrchestratorStore()
    store._kv[("dlg1", "suspended_specialists")] = json.dumps(["bitrix24"])
    tool = _make_tool(store)

    result = asyncio.run(tool.execute({"action": "resume", "specialist_id": "bitrix24"}, dialog_key="dlg1"))

    assert result.status == ToolStatus.OK
    assert result.data["resumed"] == "bitrix24"
    assert asyncio.run(store.get_kv("dlg1", "pending_specialist")) == "bitrix24"
    assert result.data["remaining_suspended"] == []


def test_list_returns_suspended():
    import json

    store = FakeOrchestratorStore()
    store._kv[("dlg1", "suspended_specialists")] = json.dumps(["bitrix24", "pto"])
    tool = _make_tool(store)

    result = asyncio.run(tool.execute({"action": "list"}, dialog_key="dlg1"))

    assert result.status == ToolStatus.OK
    assert set(result.data["suspended_specialists"]) == {"bitrix24", "pto"}


def test_not_configured_without_store():
    tool = _make_tool(store=None)

    result = asyncio.run(tool.execute({"action": "list"}, dialog_key="dlg1"))

    assert result.status == ToolStatus.NOT_CONFIGURED


def test_invalid_action():
    store = FakeOrchestratorStore()
    tool = _make_tool(store)

    result = asyncio.run(tool.execute({"action": "destroy"}, dialog_key="dlg1"))

    assert result.status == ToolStatus.INVALID_TOOL_CALL
