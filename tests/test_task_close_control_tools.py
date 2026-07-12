from __future__ import annotations

import asyncio

from ai_server.agents.bitrix24.tools.task_close_control import (
    TaskCloseControlGetTool,
    TaskCloseControlUpdateTool,
)
from ai_server.models import ToolStatus
from tests.fakes import FakeTaskDraftStore


class FakeBitrixUserClient:
    def __init__(self, users: list[dict]) -> None:
        self.users = {int(user["ID"]): user for user in users}

    async def list_all_users(
        self, *, filter_: dict | None = None, select: list[str] | None = None, limit: int | None = None
    ):
        users = list(self.users.values())
        return users[:limit] if limit else users

    async def search_users(self, query: str, *, limit: int = 10):
        normalized = query.casefold()
        users = [
            user
            for user in self.users.values()
            if normalized in f"{user.get('LAST_NAME', '')} {user.get('NAME', '')}".casefold()
        ]
        return users[:limit]

    async def get_user(self, user_id: int):
        return self.users.get(int(user_id))


def test_task_close_control_admin_adds_operator_without_controlling_operator() -> None:
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
    assert result.data["controlled_user_ids"] == []


def test_task_close_control_operator_can_add_controlled_user() -> None:
    store = FakeTaskDraftStore()
    store.set_task_close_operators(operator_user_ids=[13], actor_user_id=1)
    tool = TaskCloseControlUpdateTool(store=store)

    result = asyncio.run(
        tool.execute({"action": "add_controlled_user", "target_user_id": 15}, user_id=13, dialog_id="13")
    )

    assert result.status == ToolStatus.OK
    assert result.data["operator_user_ids"] == [13]
    assert result.data["controlled_user_ids"] == [15]


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


def test_task_close_control_get_returns_named_members_and_bitrix_users() -> None:
    store = FakeTaskDraftStore()
    store.set_task_close_operators(operator_user_ids=[13], actor_user_id=1)
    store.upsert_task_close_controlled_user(user_id=15, active=True, updated_by=1)
    client = FakeBitrixUserClient(
        [
            {"ID": 13, "NAME": "Иван", "LAST_NAME": "Петров", "ACTIVE": True},
            {"ID": 15, "NAME": "Анна", "LAST_NAME": "Сидорова", "ACTIVE": True},
            {"ID": 17, "NAME": "Борис", "LAST_NAME": "Борисов", "ACTIVE": True},
        ]
    )
    tool = TaskCloseControlGetTool(store=store, user_client=client)

    result = asyncio.run(
        tool.execute(
            {"_actor_is_admin": True, "available_limit": 10},
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.OK
    members = {item["user_id"]: item for item in result.data["members"]}
    assert members[13]["name"] == "Петров Иван"
    assert members[13]["is_operator"] is True
    assert members[13]["is_controlled"] is False
    assert members[15]["name"] == "Сидорова Анна"
    assert members[15]["is_operator"] is False
    assert members[15]["is_controlled"] is True
    available = {item["user_id"]: item for item in result.data["available_users"]}
    assert available[17]["name"] == "Борисов Борис"
    assert available[17]["can_add_operator"] is True
    assert available[17]["can_add_controlled"] is True


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
