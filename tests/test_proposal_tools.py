"""Tests for SaveIncompleteProposalTool, DeleteIncompleteProposalTool,
SaveResponsibleResponseTool and proposal_context()."""

from __future__ import annotations

import anyio

from ai_server.agents.bitrix24.tools.proposals import (
    DeleteIncompleteProposalTool,
    SaveIncompleteProposalTool,
    SaveResponsibleResponseTool,
    proposal_context,
)
from ai_server.models import ToolStatus
from tests.fakes import FakeProposalStore


def _exec(tool, args, *, user_id=None):
    async def _run():
        return await tool.execute(args, user_id=user_id)

    return anyio.run(_run)


# ---------------------------------------------------------------------------
# SaveIncompleteProposalTool
# ---------------------------------------------------------------------------


def test_save_proposal_ok():
    store = FakeProposalStore()
    tool = SaveIncompleteProposalTool(store=store)
    result = _exec(tool, {"task_id": 101, "task_title": "Камера", "missing_parts": "не проверен кабель"})
    assert result.status == ToolStatus.OK
    assert result.data["proposal_id"] == 1
    assert result.data["scheduled_for"]
    assert store._proposals[1]["task_id"] == 101
    assert store._proposals[1]["missing_parts"] == "не проверен кабель"


def test_save_proposal_missing_task_id():
    tool = SaveIncompleteProposalTool(store=FakeProposalStore())
    result = _exec(tool, {"missing_parts": "что-то не сделано"})
    assert result.status == ToolStatus.INVALID_TOOL_CALL
    assert "task_id" in result.error


def test_save_proposal_empty_missing_parts():
    tool = SaveIncompleteProposalTool(store=FakeProposalStore())
    result = _exec(tool, {"task_id": 5, "missing_parts": "  "})
    assert result.status == ToolStatus.INVALID_TOOL_CALL
    assert "missing_parts" in result.error


def test_save_proposal_no_store():
    tool = SaveIncompleteProposalTool(store=None)
    result = _exec(tool, {"task_id": 5, "missing_parts": "что-то"})
    assert result.status == ToolStatus.NOT_CONFIGURED


# ---------------------------------------------------------------------------
# DeleteIncompleteProposalTool
# ---------------------------------------------------------------------------


def test_delete_proposal_ok():
    store = FakeProposalStore()
    store.save_proposal(
        task_id=10,
        task_title="",
        missing_parts="x",
        responsible_id=None,
        responsible_dialog_id="",
        scheduled_for="2026-01-01",
    )
    tool = DeleteIncompleteProposalTool(store=store)
    result = _exec(tool, {"proposal_id": 1})
    assert result.status == ToolStatus.OK
    assert 1 not in store._proposals


def test_delete_proposal_missing_id():
    tool = DeleteIncompleteProposalTool(store=FakeProposalStore())
    result = _exec(tool, {})
    assert result.status == ToolStatus.INVALID_TOOL_CALL
    assert "proposal_id" in result.error


# ---------------------------------------------------------------------------
# SaveResponsibleResponseTool
# ---------------------------------------------------------------------------


def test_save_responsible_response_ok():
    store = FakeProposalStore()
    store.save_proposal(
        task_id=10,
        task_title="",
        missing_parts="x",
        responsible_id=9,
        responsible_dialog_id="",
        scheduled_for="2026-01-01",
    )
    tool = SaveResponsibleResponseTool(store=store)
    result = _exec(tool, {"proposal_id": 1, "response_text": "Сделаю завтра"})
    assert result.status == ToolStatus.OK
    assert store._proposals[1]["responsible_response"] == "Сделаю завтра"


def test_save_responsible_response_empty_text():
    tool = SaveResponsibleResponseTool(store=FakeProposalStore())
    result = _exec(tool, {"proposal_id": 1, "response_text": "  "})
    assert result.status == ToolStatus.INVALID_TOOL_CALL
    assert "response_text" in result.error


# ---------------------------------------------------------------------------
# proposal_context()
# ---------------------------------------------------------------------------


def test_proposal_context_for_manager():
    store = FakeProposalStore()
    store.save_proposal(
        task_id=5,
        task_title="Задача",
        missing_parts="не сделано",
        responsible_id=None,
        responsible_dialog_id="",
        scheduled_for="2026-01-01",
    )
    store._proposals[1]["status"] = "proposed"
    ctx = proposal_context(store, user_id=42, manager_id=42)
    assert "pending_manager_proposals" in ctx
    assert ctx["pending_manager_proposals"][0]["task_id"] == 5


def test_proposal_context_for_responsible():
    store = FakeProposalStore()
    store.save_proposal(
        task_id=7,
        task_title="",
        missing_parts="x",
        responsible_id=9,
        responsible_dialog_id="",
        scheduled_for="2026-01-01",
    )
    ctx = proposal_context(store, user_id=9, manager_id=None)
    assert "pending_responsible_question" in ctx
    assert ctx["pending_responsible_question"]["task_id"] == 7


def test_proposal_context_empty_when_no_user():
    store = FakeProposalStore()
    ctx = proposal_context(store, user_id=None, manager_id=None)
    assert ctx == {}


def test_proposal_context_empty_when_no_store():
    ctx = proposal_context(None, user_id=9, manager_id=None)
    assert ctx == {}
