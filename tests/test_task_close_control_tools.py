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
        self.get_user_calls: list[int] = []

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
        self.get_user_calls.append(int(user_id))
        return self.users.get(int(user_id))


def _admin_client(*users: dict) -> FakeBitrixUserClient:
    return FakeBitrixUserClient(
        [
            {"ID": 1, "NAME": "Ada", "LAST_NAME": "Admin", "ACTIVE": True, "IS_ADMIN": True},
            *users,
        ]
    )


class CurrentUserOAuth:
    def __init__(self, client: FakeBitrixUserClient) -> None:
        self.client = client

    async def client_for_user(self, user_id: int):
        return self.client


def _update_tool(store: FakeTaskDraftStore, client: FakeBitrixUserClient) -> TaskCloseControlUpdateTool:
    return TaskCloseControlUpdateTool(store=store, user_client=client, bitrix_oauth=CurrentUserOAuth(client))


def test_task_close_control_admin_adds_operator_without_controlling_operator() -> None:
    store = FakeTaskDraftStore()
    client = _admin_client({"ID": 13, "NAME": "Olga", "LAST_NAME": "Operator", "ACTIVE": True})
    tool = _update_tool(store, client)

    prepared = asyncio.run(
        tool.execute(
            {
                "action": "add_operator",
                "target_user_id": 13,
                "_actor_is_admin": True,
                "_original_request": "Add Olga as an operator",
                "_draft_user_id": 1,
                "_draft_specialist": "bitrix24",
            },
            user_id=1,
            dialog_key="d:1",
            dialog_id="1",
        )
    )

    assert prepared.status == ToolStatus.OK
    assert prepared.data["requires_confirmation"] is True
    assert prepared.data["draft"]["_draft_type"] == "admin_change"
    assert prepared.data["draft"]["old_value"] is False
    assert prepared.data["draft"]["new_value"] is True
    assert prepared.data["draft"]["_original_request"] == "Add Olga as an operator"
    assert prepared.data["draft"]["_draft_user_id"] == 1
    assert prepared.data["draft"]["_draft_specialist"] == "bitrix24"
    assert store.task_close_operator_ids() == set()

    confirmed = asyncio.run(
        tool.execute({"operation": "confirm"}, user_id=1, dialog_key="d:1", dialog_id="1")
    )

    assert confirmed.status == ToolStatus.OK
    assert confirmed.data["operator_user_ids"] == [13]
    assert confirmed.data["controlled_user_ids"] == []
    assert client.get_user_calls.count(1) == 2


def test_task_close_control_operator_cannot_add_controlled_user() -> None:
    store = FakeTaskDraftStore()
    store.set_task_close_operators(operator_user_ids=[13], actor_user_id=1)
    client = FakeBitrixUserClient(
        [
            {"ID": 13, "NAME": "Olga", "LAST_NAME": "Operator", "ACTIVE": True, "IS_ADMIN": False},
            {"ID": 15, "NAME": "Ivan", "LAST_NAME": "Worker", "ACTIVE": True},
        ]
    )
    tool = _update_tool(store, client)

    result = asyncio.run(
        tool.execute(
            {"action": "add_controlled_user", "target_user_id": 15, "_actor_is_admin": True},
            user_id=13,
            dialog_key="d:13",
            dialog_id="13",
        )
    )

    assert result.status == ToolStatus.DENIED
    assert store.task_close_controlled_user_ids() == set()
    assert "d:13" not in store._drafts


def test_task_close_control_operator_cannot_change_auto_close_time() -> None:
    store = FakeTaskDraftStore()
    store.set_task_close_operators(operator_user_ids=[13], actor_user_id=1)
    client = FakeBitrixUserClient(
        [{"ID": 13, "NAME": "Olga", "LAST_NAME": "Operator", "ACTIVE": True, "IS_ADMIN": False}]
    )
    tool = _update_tool(store, client)

    result = asyncio.run(
        tool.execute(
            {"action": "set_auto_close_time", "auto_close_time": "19:30"},
            user_id=13,
            dialog_key="d:13",
            dialog_id="13",
        )
    )

    assert result.status == ToolStatus.DENIED
    assert store.get_task_close_control_setting("auto_close_time") is None


def test_task_close_control_uses_current_user_oauth_for_admin_proof() -> None:
    store = FakeTaskDraftStore()
    fallback = _admin_client()
    oauth_client = FakeBitrixUserClient(
        [{"ID": 1, "NAME": "Not", "LAST_NAME": "Admin", "ACTIVE": True, "IS_ADMIN": False}]
    )

    class FakeOAuth:
        async def client_for_user(self, user_id: int):
            assert user_id == 1
            return oauth_client

    tool = TaskCloseControlUpdateTool(store=store, user_client=fallback, bitrix_oauth=FakeOAuth())

    result = asyncio.run(
        tool.execute(
            {"action": "set_auto_close_time", "auto_close_time": "19:30", "_actor_is_admin": True},
            user_id=1,
            dialog_key="d:1",
        )
    )

    assert result.status == ToolStatus.DENIED
    assert fallback.get_user_calls == []
    assert oauth_client.get_user_calls == [1]
    assert "d:1" not in store._drafts


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
    tool = _update_tool(store, _admin_client())

    time_result = asyncio.run(
        tool.execute(
            {"action": "set_auto_close_time", "auto_close_time": "19:30", "_actor_is_admin": True},
            user_id=1,
            dialog_key="d:1",
            dialog_id="1",
        )
    )
    time_confirm = asyncio.run(tool.execute({"operation": "confirm"}, user_id=1, dialog_key="d:1", dialog_id="1"))
    start_result = asyncio.run(
        tool.execute(
            {
                "action": "set_control_enabled_from",
                "control_enabled_from": "2026-07-12T00:00:00+03:00",
                "_actor_is_admin": True,
            },
            user_id=1,
            dialog_key="d:1",
            dialog_id="1",
        )
    )
    start_confirm = asyncio.run(
        tool.execute({"operation": "confirm"}, user_id=1, dialog_key="d:1", dialog_id="1")
    )

    assert time_result.status == ToolStatus.OK
    assert time_confirm.status == ToolStatus.OK
    assert start_result.status == ToolStatus.OK
    assert start_confirm.status == ToolStatus.OK
    assert start_confirm.data["auto_close_time"] == "19:30"
    assert start_confirm.data["control_enabled_from"] == "2026-07-12T00:00:00+03:00"


def test_task_close_control_rejects_invalid_time() -> None:
    store = FakeTaskDraftStore()
    tool = _update_tool(store, _admin_client())

    result = asyncio.run(
        tool.execute(
            {"action": "set_auto_close_time", "auto_close_time": "25:99", "_actor_is_admin": True},
            user_id=1,
            dialog_key="d:1",
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.INVALID_TOOL_CALL


def test_task_close_control_discard_does_not_mutate_settings() -> None:
    store = FakeTaskDraftStore()
    tool = _update_tool(store, _admin_client())
    prepared = asyncio.run(
        tool.execute(
            {"action": "set_auto_close_time", "auto_close_time": "19:30"},
            user_id=1,
            dialog_key="d:1",
        )
    )

    discarded = asyncio.run(tool.execute({"operation": "discard"}, user_id=1, dialog_key="d:1"))

    assert prepared.status == ToolStatus.OK
    assert discarded.status == ToolStatus.OK
    assert discarded.data["discarded"] is True
    assert store.get_task_close_control_setting("auto_close_time") is None


def test_task_close_control_resolves_exact_name_and_rejects_ambiguity() -> None:
    store = FakeTaskDraftStore()
    client = _admin_client(
        {"ID": 13, "NAME": "Ivan", "LAST_NAME": "Petrov", "ACTIVE": True},
        {"ID": 14, "NAME": "Ivan", "LAST_NAME": "Petrov", "ACTIVE": True},
    )
    tool = _update_tool(store, client)

    result = asyncio.run(
        tool.execute(
            {"action": "add_operator", "target_user_name": "Petrov Ivan"},
            user_id=1,
            dialog_key="d:1",
        )
    )

    assert result.status == ToolStatus.AMBIGUOUS
    assert [item["user_id"] for item in result.data["matches"]] == [13, 14]
    assert store.task_close_operator_ids() == set()


def test_task_close_control_confirm_revalidates_admin_and_fails_closed() -> None:
    store = FakeTaskDraftStore()
    client = _admin_client({"ID": 13, "NAME": "Olga", "LAST_NAME": "Operator", "ACTIVE": True})
    tool = _update_tool(store, client)
    prepared = asyncio.run(
        tool.execute({"action": "add_operator", "target_user_id": 13}, user_id=1, dialog_key="d:1")
    )
    client.users[1]["IS_ADMIN"] = False

    confirmed = asyncio.run(tool.execute({"operation": "confirm"}, user_id=1, dialog_key="d:1"))

    assert prepared.status == ToolStatus.OK
    assert confirmed.status == ToolStatus.DENIED
    assert store.task_close_operator_ids() == set()
    assert "d:1" in store._drafts


def test_task_close_control_confirm_rejects_concurrent_setting_change() -> None:
    store = FakeTaskDraftStore()
    tool = _update_tool(store, _admin_client())
    prepared = asyncio.run(
        tool.execute(
            {"action": "set_auto_close_time", "auto_close_time": "19:30"},
            user_id=1,
            dialog_key="d:1",
        )
    )
    store.set_task_close_control_setting(key="auto_close_time", value="18:00", updated_by=99)

    confirmed = asyncio.run(tool.execute({"operation": "confirm"}, user_id=1, dialog_key="d:1"))

    assert prepared.status == ToolStatus.OK
    assert confirmed.status == ToolStatus.ERROR
    assert store.get_task_close_control_setting("auto_close_time")["value"] == "18:00"
    assert "d:1" in store._drafts
    assert "d:1" not in store._confirming


def test_task_close_control_confirm_refuses_an_already_claimed_draft() -> None:
    store = FakeTaskDraftStore()
    tool = _update_tool(store, _admin_client())
    prepared = asyncio.run(
        tool.execute(
            {"action": "set_auto_close_time", "auto_close_time": "19:30"},
            user_id=1,
            dialog_key="d:1",
        )
    )
    store._confirming.add("d:1")

    confirmed = asyncio.run(tool.execute({"operation": "confirm"}, user_id=1, dialog_key="d:1"))

    assert prepared.status == ToolStatus.OK
    assert confirmed.status == ToolStatus.ERROR
    assert store.get_task_close_control_setting("auto_close_time") is None
