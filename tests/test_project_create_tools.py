"""Tests for Bitrix project creation draft tools."""

from __future__ import annotations

from unittest.mock import AsyncMock

import anyio

from ai_server.agents.bitrix24.tools.project_create import (
    PROJECT_CREATE_DRAFT_TYPE,
    ProjectCreateConfirmTool,
    ProjectCreateDiscardTool,
    ProjectCreateDraftTool,
)
from ai_server.models import ToolStatus
from tests.fakes import FakeTaskDraftStore


def _exec(tool, args, *, user_id=None, dialog_key=None, dialog_id=None):
    async def _run():
        return await tool.execute(args, user_id=user_id, dialog_key=dialog_key, dialog_id=dialog_id)

    return anyio.run(_run)


def test_project_draft_allows_regular_user_personal_project():
    store = FakeTaskDraftStore()
    tool = ProjectCreateDraftTool(store=store)

    result = _exec(
        tool,
        {
            "name": "Кулинич Валерий",
            "personal_for_self": True,
            "_actor_name": "Кулинич Валерий",
        },
        user_id=9,
        dialog_key="d:9",
        dialog_id="chat9",
    )

    assert result.status == ToolStatus.OK
    draft = store._drafts["d:9"]
    assert draft["_draft_type"] == PROJECT_CREATE_DRAFT_TYPE
    assert draft["method"] == "sonet_group.create"
    assert draft["params"]["fields"]["NAME"] == "Кулинич Валерий"
    assert draft["params"]["fields"]["OPENED"] == "Y"
    assert draft["params"]["fields"]["VISIBLE"] == "Y"
    assert draft["params"]["fields"]["PROJECT"] == "Y"
    assert result.data["preview"]["type"] == "личный проект"


def test_project_draft_denies_regular_user_arbitrary_project():
    store = FakeTaskDraftStore()
    tool = ProjectCreateDraftTool(store=store)

    result = _exec(
        tool,
        {
            "name": "Новый общий проект",
            "personal_for_self": False,
            "_actor_name": "Кулинич Валерий",
        },
        user_id=9,
        dialog_key="d:9",
        dialog_id="chat9",
    )

    assert result.status == ToolStatus.CONTRACT_VIOLATION
    assert "d:9" not in store._drafts


def test_project_draft_denies_personal_name_mismatch():
    store = FakeTaskDraftStore()
    tool = ProjectCreateDraftTool(store=store)

    result = _exec(
        tool,
        {
            "name": "Борисов Андрей",
            "personal_for_self": True,
            "_actor_name": "Кулинич Валерий",
        },
        user_id=9,
        dialog_key="d:9",
        dialog_id="chat9",
    )

    assert result.status == ToolStatus.CONTRACT_VIOLATION
    assert "d:9" not in store._drafts


def test_project_draft_allows_personal_project_without_patronymic():
    store = FakeTaskDraftStore()
    tool = ProjectCreateDraftTool(store=store)

    result = _exec(
        tool,
        {
            "name": "Коверга Дмитрий",
            "personal_for_self": True,
            "_actor_name": "Коверга Дмитрий Владимирович",
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert result.status == ToolStatus.OK
    assert store._drafts["d:13"]["params"]["fields"]["NAME"] == "Коверга Дмитрий"


def test_project_draft_allows_admin_arbitrary_project():
    store = FakeTaskDraftStore()
    tool = ProjectCreateDraftTool(store=store)

    result = _exec(
        tool,
        {
            "name": "Новый общий проект",
            "_actor_name": "Администратор",
            "_actor_is_admin": True,
        },
        user_id=1,
        dialog_key="d:1",
        dialog_id="chat1",
    )

    assert result.status == ToolStatus.OK
    assert store._drafts["d:1"]["params"]["fields"]["NAME"] == "Новый общий проект"


def test_project_draft_requires_user_and_dialog_context():
    store = FakeTaskDraftStore()
    tool = ProjectCreateDraftTool(store=store)

    result = _exec(
        tool,
        {"name": "Кулинич Валерий", "personal_for_self": True, "_actor_name": "Кулинич Валерий"},
        user_id=9,
        dialog_key="d:9",
        dialog_id=None,
    )

    assert result.status == ToolStatus.DENIED
    assert "d:9" not in store._drafts


def test_project_confirm_uses_oauth_and_deletes_draft():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:9",
            {
                "_draft_type": PROJECT_CREATE_DRAFT_TYPE,
                "method": "sonet_group.create",
                "params": {"fields": {"NAME": "Кулинич Валерий", "OWNER_ID": 9}},
            },
        )
    )
    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"ID": 111})
    oauth_client = AsyncMock()
    oauth_client.call = AsyncMock(return_value={"ID": 777})
    oauth = FakeBitrixOAuth(oauth_client)

    tool = ProjectCreateConfirmTool(
        store=store,
        write_client=write_client,
        bitrix_oauth=oauth,
        dry_run=False,
        oauth_required_for_writes=True,
    )
    result = _exec(tool, {}, user_id=9, dialog_key="d:9", dialog_id="chat9")

    assert result.status == ToolStatus.OK
    assert result.data["result"]["ID"] == 777
    assert oauth.user_ids == [9]
    oauth_client.call.assert_awaited_once_with(
        "sonet_group.create",
        {"fields": {"NAME": "Кулинич Валерий", "OWNER_ID": 9}},
    )
    write_client.call.assert_not_called()
    assert "d:9" not in store._drafts


def test_project_confirm_replaces_project_draft_with_followup_task_draft():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:9",
            {
                "_draft_type": PROJECT_CREATE_DRAFT_TYPE,
                "method": "sonet_group.create",
                "params": {"fields": {"NAME": "Кулинич Валерий", "OWNER_ID": 9}},
                "after_project_create_task_draft": {
                    "params": {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9}},
                    "preview": {"title": "Задача", "responsible": "Кулинич Валерий"},
                    "project_name": "Кулинич Валерий",
                },
            },
        )
    )
    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"result": 777})
    tool = ProjectCreateConfirmTool(store=store, write_client=write_client, oauth_required_for_writes=False)

    result = _exec(tool, {}, dialog_key="d:9")

    assert result.status == ToolStatus.OK
    assert store._drafts["d:9"]["fields"]["GROUP_ID"] == 777
    assert store._drafts["d:9"]["fields"]["TITLE"] == "Задача"
    assert result.data["followup_task_draft"]["preview"]["project"] == "Кулинич Валерий"


def test_project_confirm_ignores_other_draft_types():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:9", {"fields": {"TITLE": "Задача"}}))
    write_client = AsyncMock()
    write_client.call = AsyncMock(return_value={"ID": 111})
    tool = ProjectCreateConfirmTool(store=store, write_client=write_client, oauth_required_for_writes=False)

    result = _exec(tool, {}, dialog_key="d:9")

    assert result.status == ToolStatus.NOT_FOUND
    write_client.call.assert_not_called()
    assert "d:9" in store._drafts


def test_project_discard_deletes_draft():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:9", {"_draft_type": PROJECT_CREATE_DRAFT_TYPE}))
    tool = ProjectCreateDiscardTool(store=store)

    result = _exec(tool, {}, dialog_key="d:9")

    assert result.status == ToolStatus.OK
    assert result.data["linked_task"] is False
    assert "d:9" not in store._drafts


def test_project_discard_reports_linked_task_from_deleted_draft():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:9",
            {
                "_draft_type": PROJECT_CREATE_DRAFT_TYPE,
                "after_project_create_task_draft": {"params": {"fields": {"TITLE": "Тест"}}},
            },
        )
    )
    result = _exec(ProjectCreateDiscardTool(store=store), {}, dialog_key="d:9")

    assert result.status == ToolStatus.OK
    assert result.data["linked_task"] is True
    assert "d:9" not in store._drafts


class FakeBitrixOAuth:
    def __init__(self, client) -> None:
        self.client = client
        self.user_ids = []

    async def client_for_user(self, user_id: int):
        self.user_ids.append(user_id)
        return self.client
