"""Tests for TaskCreateDraftTool, TaskCreateConfirmTool, TaskDraftDiscardTool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import anyio

from ai_server.agents.bitrix24.tools.project_create import PROJECT_CREATE_DRAFT_TYPE
from ai_server.agents.bitrix24.tools.task_create import (
    TaskCreateConfirmTool,
    TaskCreateDraftTool,
    TaskDraftDiscardTool,
)
from ai_server.models import ToolStatus
from tests.fakes import FakeTaskDraftStore


def _exec(tool, args, *, user_id=None, dialog_key=None, dialog_id=None):
    async def _run():
        return await tool.execute(args, user_id=user_id, dialog_key=dialog_key, dialog_id=dialog_id)

    return anyio.run(_run)


# ---------------------------------------------------------------------------
# TaskCreateDraftTool
# ---------------------------------------------------------------------------


def test_draft_tool_saves_to_store():
    store = FakeTaskDraftStore()
    tool = TaskCreateDraftTool(store=store)
    result = _exec(
        tool,
        {"title": "Тест", "responsible_id": 9, "no_deadline": True},
        user_id=9,
        dialog_key="d:42",
    )
    assert result.status == ToolStatus.OK
    assert "d:42" in store._drafts
    assert store._drafts["d:42"]["fields"]["TITLE"] == "Тест"


def test_draft_tool_returns_plain_preview_without_user_id():
    store = FakeTaskDraftStore()
    tool = TaskCreateDraftTool(store=store)
    result = _exec(
        tool,
        {"title": "Тест", "responsible_self": True, "responsible_name": "Кулинич Валерий"},
        user_id=9,
        dialog_key="d:42",
    )
    preview = result.data["preview"]

    assert result.status == ToolStatus.OK
    assert preview["responsible"] == "Кулинич Валерий"
    assert "текущий пользователь" not in " ".join(preview.values())
    assert "#9" not in " ".join(preview.values())
    assert preview["deadline"].endswith("МСК")


def test_draft_tool_contract_violation():
    store = FakeTaskDraftStore()
    tool = TaskCreateDraftTool(store=store)
    result = _exec(
        tool,
        {"title": "", "responsible_id": 9, "no_deadline": True},
        user_id=9,
        dialog_key="d:42",
    )
    assert result.status == ToolStatus.CONTRACT_VIOLATION
    assert not store._drafts


def test_draft_tool_no_dialog_key_does_not_fail():
    store = FakeTaskDraftStore()
    tool = TaskCreateDraftTool(store=store)
    result = _exec(
        tool,
        {"title": "Задача", "responsible_id": 9, "no_deadline": True},
        user_id=9,
        dialog_key=None,
    )
    assert result.status == ToolStatus.OK
    assert not store._drafts


def test_draft_tool_prepares_personal_project_before_default_self_task():
    class _NoProjectClient:
        async def search_projects(self, query: str, *, limit: int = 10):
            assert query == "Кулинич Валерий"
            return []

    store = FakeTaskDraftStore()
    tool = TaskCreateDraftTool(store=store, project_client=_NoProjectClient())

    result = _exec(
        tool,
        {
            "title": "Тест",
            "responsible_self": True,
            "responsible_name": "Кулинич Валерий",
            "project_name": "Кулинич Валерий",
            "_default_personal_project": True,
            "no_deadline": True,
        },
        user_id=9,
        dialog_key="d:42",
        dialog_id="chat42",
    )

    assert result.status == ToolStatus.OK
    assert result.data["requires_project_creation"] is True
    assert result.data["missing_project_name"] == "Кулинич Валерий"
    stored = store._drafts["d:42"]
    assert stored["_draft_type"] == PROJECT_CREATE_DRAFT_TYPE
    assert stored["params"]["fields"]["NAME"] == "Кулинич Валерий"
    followup = stored["after_project_create_task_draft"]
    assert followup["params"]["fields"]["TITLE"] == "Тест"
    assert followup["params"]["fields"]["RESPONSIBLE_ID"] == 9
    assert "GROUP_ID" not in followup["params"]["fields"]


# ---------------------------------------------------------------------------
# TaskCreateConfirmTool
# ---------------------------------------------------------------------------


def test_confirm_tool_creates_task():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:1", {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9}}))

    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"task": {"id": 777}})

    tool = TaskCreateConfirmTool(
        store=store,
        write_client=write_client,
        dry_run=False,
        oauth_required_for_writes=False,
    )
    result = _exec(tool, {}, dialog_key="d:1")

    assert result.status == ToolStatus.OK
    assert result.data["result"]["task"]["id"] == 777
    assert "d:1" not in store._drafts


def test_confirm_tool_uses_oauth_when_required():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:1", {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9}}))

    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"task": {"id": 111}})
    oauth_client = AsyncMock()
    oauth_client.call = AsyncMock(return_value={"task": {"id": 777}})
    oauth = FakeBitrixOAuth(oauth_client)

    tool = TaskCreateConfirmTool(
        store=store,
        write_client=write_client,
        bitrix_oauth=oauth,
        dry_run=False,
        oauth_required_for_writes=True,
    )
    result = _exec(tool, {}, user_id=9, dialog_key="d:1", dialog_id="chat99")

    assert result.status == ToolStatus.OK
    assert result.data["result"]["task"]["id"] == 777
    assert oauth.user_ids == [9]
    oauth_client.call.assert_awaited_once()
    write_client.call.assert_not_called()
    assert "d:1" not in store._drafts


def test_confirm_tool_required_oauth_blocks_missing_dialog_id():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:1", {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9}}))

    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"task": {"id": 111}})
    oauth = FakeBitrixOAuth(AsyncMock())

    tool = TaskCreateConfirmTool(
        store=store,
        write_client=write_client,
        bitrix_oauth=oauth,
        dry_run=False,
        oauth_required_for_writes=True,
    )
    result = _exec(tool, {}, user_id=9, dialog_key="d:1", dialog_id=None)

    assert result.status == ToolStatus.DENIED
    assert oauth.user_ids == []
    write_client.call.assert_not_called()
    assert "d:1" in store._drafts


def test_confirm_tool_required_oauth_does_not_fallback_to_write_client():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:1", {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9}}))

    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"task": {"id": 111}})

    tool = TaskCreateConfirmTool(
        store=store,
        write_client=write_client,
        bitrix_oauth=None,
        dry_run=False,
        oauth_required_for_writes=True,
    )
    result = _exec(tool, {}, user_id=9, dialog_key="d:1", dialog_id="chat99")

    assert result.status == ToolStatus.NOT_CONFIGURED
    write_client.call.assert_not_called()
    assert "d:1" in store._drafts


def test_confirm_tool_dry_run():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:2", {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9}}))

    tool = TaskCreateConfirmTool(store=store, write_client=None, dry_run=True)
    result = _exec(tool, {}, dialog_key="d:2")

    assert result.status == ToolStatus.DRY_RUN
    assert "d:2" in store._drafts  # не удалён при dry_run


def test_confirm_tool_no_draft():
    store = FakeTaskDraftStore()
    tool = TaskCreateConfirmTool(store=store, write_client=None, dry_run=False)
    result = _exec(tool, {}, dialog_key="d:99")
    assert result.status == ToolStatus.NOT_FOUND


def test_confirm_tool_ignores_task_close_draft():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:1", {"_draft_type": "task_close", "task_id": 139}))

    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"task": {"id": 111}})
    tool = TaskCreateConfirmTool(
        store=store,
        write_client=write_client,
        dry_run=False,
        oauth_required_for_writes=False,
    )
    result = _exec(tool, {}, dialog_key="d:1")

    assert result.status == ToolStatus.NOT_FOUND
    write_client.call.assert_not_called()
    assert "d:1" in store._drafts


def test_confirm_tool_expired_draft_is_not_created():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:1", {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9}}))
    store._expired.add("d:1")

    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"task": {"id": 111}})

    tool = TaskCreateConfirmTool(
        store=store,
        write_client=write_client,
        dry_run=False,
        oauth_required_for_writes=False,
        draft_ttl_minutes=1,
    )
    result = _exec(tool, {}, dialog_key="d:1")

    assert result.status == ToolStatus.NOT_FOUND
    write_client.call.assert_not_called()
    assert "d:1" not in store._drafts


def test_confirm_tool_no_dialog_key():
    store = FakeTaskDraftStore()
    tool = TaskCreateConfirmTool(store=store, write_client=None, dry_run=False)
    result = _exec(tool, {}, dialog_key=None)
    assert result.status == ToolStatus.INVALID_TOOL_CALL


# ---------------------------------------------------------------------------
# TaskDraftDiscardTool
# ---------------------------------------------------------------------------


def test_discard_tool_deletes_draft():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:5", {"fields": {"TITLE": "Задача"}}))

    tool = TaskDraftDiscardTool(store=store)
    result = _exec(tool, {}, dialog_key="d:5")

    assert result.status == ToolStatus.OK
    assert "d:5" not in store._drafts


class FakeBitrixOAuth:
    def __init__(self, client) -> None:
        self.client = client
        self.user_ids = []

    async def client_for_user(self, user_id: int):
        self.user_ids.append(user_id)
        return self.client
