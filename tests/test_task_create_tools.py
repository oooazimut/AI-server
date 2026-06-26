"""Tests for TaskCreateDraftTool, TaskCreateConfirmTool, TaskDraftDiscardTool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import anyio

from ai_server.agents.bitrix24.tools.task_create import (
    TaskCreateConfirmTool,
    TaskCreateDraftTool,
    TaskDraftDiscardTool,
)
from ai_server.models import ToolStatus
from tests.fakes import FakeTaskDraftStore


def _exec(tool, args, *, user_id=None, dialog_key=None):
    async def _run():
        return await tool.execute(args, user_id=user_id, dialog_key=dialog_key)

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


# ---------------------------------------------------------------------------
# TaskCreateConfirmTool
# ---------------------------------------------------------------------------


def test_confirm_tool_creates_task():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:1", {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9}}))

    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"task": {"id": 777}})

    tool = TaskCreateConfirmTool(store=store, write_client=write_client, dry_run=False)
    result = _exec(tool, {}, dialog_key="d:1")

    assert result.status == ToolStatus.OK
    assert result.data["result"]["task"]["id"] == 777
    assert "d:1" not in store._drafts


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
