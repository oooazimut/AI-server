from __future__ import annotations

import asyncio

from ai_server.agents.bitrix24.tools.task_close_control import (
    TaskCloseControlGetTool,
    TaskCloseControlUpdateTool,
)
from ai_server.models import ToolStatus
from tests.fakes import FakeTaskDraftStore


def test_task_close_control_admin_adds_operator_and_operator_becomes_controlled() -> None:
    store = FakeTaskDraftStore()
    tool = TaskCloseControlUpdateTool(store=store)

    result = asyncio.run(
        tool.execute(
            {"action": "add_operator", "target_user_id": 13, "_actor_is_admin": True},
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.OK
    assert result.data["operator_user_ids"] == [13]
    assert result.data["controlled_user_ids"] == [13]


def test_task_close_control_operator_can_add_controlled_user() -> None:
    store = FakeTaskDraftStore()
    store.set_task_close_operators(operator_user_ids=[13], actor_user_id=1)
    tool = TaskCloseControlUpdateTool(store=store)

    result = asyncio.run(
        tool.execute({"action": "add_controlled_user", "target_user_id": 15}, user_id=13, dialog_id="13")
    )

    assert result.status == ToolStatus.OK
    assert result.data["operator_user_ids"] == [13]
    assert result.data["controlled_user_ids"] == [13, 15]


def test_task_close_control_operator_cannot_change_auto_close_time() -> None:
    store = FakeTaskDraftStore()
    store.set_task_close_operators(operator_user_ids=[13], actor_user_id=1)
    tool = TaskCloseControlUpdateTool(store=store)

    result = asyncio.run(
        tool.execute({"action": "set_auto_close_time", "auto_close_time": "19:30"}, user_id=13, dialog_id="13")
    )

    assert result.status == ToolStatus.DENIED
    assert store.get_task_close_control_setting("auto_close_time") is None


def test_task_close_control_regular_user_cannot_read_settings() -> None:
    store = FakeTaskDraftStore()
    tool = TaskCloseControlGetTool(store=store)

    result = asyncio.run(tool.execute({}, user_id=44, dialog_id="44"))

    assert result.status == ToolStatus.DENIED


def test_task_close_control_admin_sets_time_and_control_start() -> None:
    store = FakeTaskDraftStore()
    tool = TaskCloseControlUpdateTool(store=store)

    time_result = asyncio.run(
        tool.execute(
            {"action": "set_auto_close_time", "auto_close_time": "19:30", "_actor_is_admin": True},
            user_id=1,
            dialog_id="1",
        )
    )
    start_result = asyncio.run(
        tool.execute(
            {
                "action": "set_control_enabled_from",
                "control_enabled_from": "2026-07-12T00:00:00+03:00",
                "_actor_is_admin": True,
            },
            user_id=1,
            dialog_id="1",
        )
    )

    assert time_result.status == ToolStatus.OK
    assert start_result.status == ToolStatus.OK
    assert start_result.data["auto_close_time"] == "19:30"
    assert start_result.data["control_enabled_from"] == "2026-07-12T00:00:00+03:00"


def test_task_close_control_rejects_invalid_time() -> None:
    store = FakeTaskDraftStore()
    tool = TaskCloseControlUpdateTool(store=store)

    result = asyncio.run(
        tool.execute(
            {"action": "set_auto_close_time", "auto_close_time": "25:99", "_actor_is_admin": True},
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.INVALID_TOOL_CALL
