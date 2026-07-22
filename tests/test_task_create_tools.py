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
from ai_server.integrations.bitrix.oauth import BitrixOAuthTokenMissing
from ai_server.models import ToolStatus
from tests.fakes import FakePortalSearchIndex, FakeTaskDraftStore


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
    assert store._drafts["d:42"]["_draft_type"] == "task_create"
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


def test_draft_tool_prepares_open_personal_project_for_named_responsible():
    class _NoProjectClient:
        async def search_projects(self, query: str, *, limit: int = 10):
            assert query == "Borisov Andrey"
            return []

    store = FakeTaskDraftStore()
    tool = TaskCreateDraftTool(store=store, project_client=_NoProjectClient())

    result = _exec(
        tool,
        {
            "title": "Wash the car",
            "responsible_id": 17,
            "responsible_name": "Borisov Andrey Sergeevich",
            "project_name": "Borisov Andrey",
            "_default_personal_project": True,
            "_default_personal_project_owner_id": 17,
            "no_deadline": True,
        },
        user_id=9,
        dialog_key="d:42",
        dialog_id="chat42",
    )

    assert result.status == ToolStatus.OK
    project_fields = store._drafts["d:42"]["params"]["fields"]
    assert project_fields["NAME"] == "Borisov Andrey"
    assert project_fields["OWNER_ID"] == 17
    assert project_fields["OPENED"] == "Y"
    assert project_fields["VISIBLE"] == "Y"
    assert project_fields["PROJECT"] == "Y"
    followup_fields = store._drafts["d:42"]["after_project_create_task_draft"]["params"]["fields"]
    assert followup_fields["RESPONSIBLE_ID"] == 17
    assert followup_fields["CREATED_BY"] == 9


def test_draft_tool_rejects_unresolved_default_personal_project():
    store = FakeTaskDraftStore()
    result = _exec(
        TaskCreateDraftTool(store=store),
        {
            "title": "Wash the car",
            "responsible_id": 17,
            "_default_personal_project_unresolved": True,
        },
        user_id=9,
        dialog_key="d:42",
    )

    assert result.status == ToolStatus.CONTRACT_VIOLATION
    assert store._drafts == {}


def test_draft_tool_does_not_create_duplicate_for_ambiguous_personal_project():
    class _AmbiguousProjectClient:
        async def search_projects(self, query: str, *, limit: int = 10):
            return [
                {"ID": "77", "NAME": "Кулинич Валерий"},
                {"ID": "78", "NAME": "Кулинич Валерий"},
            ]

    store = FakeTaskDraftStore()
    result = _exec(
        TaskCreateDraftTool(store=store, project_client=_AmbiguousProjectClient()),
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

    assert result.status == ToolStatus.NOT_FOUND
    assert store._drafts == {}


def test_explicit_project_draft_requires_one_exact_numeric_project():
    class _ProjectClient:
        async def search_projects(self, query: str, *, limit: int = 10):
            return [{"ID": "77", "NAME": "Ларгус-2"}]

    store = FakeTaskDraftStore()
    tool = TaskCreateDraftTool(store=store, project_client=_ProjectClient())

    result = _exec(
        tool,
        {"title": "Тест", "responsible_self": True, "project_name": "Ларгус-2"},
        user_id=9,
        dialog_key="d:42",
    )

    assert result.status == ToolStatus.OK
    assert store._drafts["d:42"]["fields"]["GROUP_ID"] == 77


def test_explicit_project_draft_uses_exact_postgres_snapshot_before_live_rest():
    class _ForbiddenLiveClient:
        async def search_projects(self, query: str, *, limit: int = 10):
            raise AssertionError("exact PostgreSQL snapshot must avoid live REST")

    index = FakePortalSearchIndex()
    index.upsert_item(entity_type="project", entity_id="77", title="Ларгус 2")
    store = FakeTaskDraftStore()
    tool = TaskCreateDraftTool(
        store=store,
        project_client=_ForbiddenLiveClient(),
        portal_search=index,
    )

    result = _exec(
        tool,
        {"title": "Тест", "responsible_self": True, "project_name": "Ларгус-2"},
        user_id=9,
        dialog_key="d:42",
    )

    assert result.status == ToolStatus.OK
    assert store._drafts["d:42"]["fields"]["GROUP_ID"] == 77
    assert store._drafts["d:42"]["_resolved_project"] == {"id": 77, "name": "Ларгус 2"}
    assert result.data["preview"]["project"] == "Ларгус 2"


def test_explicit_project_draft_rejects_ambiguous_exact_postgres_snapshot():
    class _ForbiddenLiveClient:
        async def search_projects(self, query: str, *, limit: int = 10):
            raise AssertionError("ambiguous authoritative snapshot must fail closed")

    index = FakePortalSearchIndex()
    index.upsert_item(entity_type="project", entity_id="77", title="Ларгус-2")
    index.upsert_item(entity_type="project", entity_id="78", title="Ларгус-2")
    store = FakeTaskDraftStore()

    result = _exec(
        TaskCreateDraftTool(
            store=store,
            project_client=_ForbiddenLiveClient(),
            portal_search=index,
        ),
        {"title": "Тест", "responsible_self": True, "project_name": "Ларгус-2"},
        user_id=9,
        dialog_key="d:42",
    )

    assert result.status == ToolStatus.NOT_FOUND
    assert store._drafts == {}


def test_explicit_project_draft_falls_back_to_live_for_fuzzy_only_snapshot():
    class _LiveClient:
        def __init__(self):
            self.calls = []

        async def search_projects(self, query: str, *, limit: int = 10):
            self.calls.append((query, limit))
            return [{"ID": "77", "NAME": "Ларгус-2"}]

    index = FakePortalSearchIndex()
    index.upsert_item(entity_type="project", entity_id="78", title="Ларгус-20")
    live = _LiveClient()
    store = FakeTaskDraftStore()

    result = _exec(
        TaskCreateDraftTool(store=store, project_client=live, portal_search=index),
        {"title": "Тест", "responsible_self": True, "project_name": "Ларгус-2"},
        user_id=9,
        dialog_key="d:42",
    )

    assert result.status == ToolStatus.OK
    assert live.calls == [("Ларгус-2", 10)]
    assert store._drafts["d:42"]["fields"]["GROUP_ID"] == 77


def test_explicit_project_snapshot_requires_current_user_oauth_actor():
    class _MissingOAuth:
        async def client_for_user(self, user_id: int):
            raise BitrixOAuthTokenMissing(user_id)

        def authorization_hint(self, user_id: int):
            return {}

    class _ForbiddenFallback:
        async def search_projects(self, query: str, *, limit: int = 10):
            raise AssertionError("missing OAuth actor must stop before fallback")

    index = FakePortalSearchIndex()
    index.upsert_item(entity_type="project", entity_id="77", title="Ларгус-2")
    store = FakeTaskDraftStore()

    result = _exec(
        TaskCreateDraftTool(
            store=store,
            project_client=_ForbiddenFallback(),
            portal_search=index,
            bitrix_oauth=_MissingOAuth(),
        ),
        {"title": "Тест", "responsible_self": True, "project_name": "Ларгус-2"},
        user_id=9,
        dialog_key="d:42",
    )

    assert result.status == ToolStatus.DENIED
    assert store._drafts == {}


def test_explicit_project_draft_without_resolver_fails_closed():
    store = FakeTaskDraftStore()
    tool = TaskCreateDraftTool(store=store, project_client=None)

    result = _exec(
        tool,
        {"title": "Тест", "responsible_self": True, "project_name": "Ларгус-2"},
        user_id=9,
        dialog_key="d:42",
    )

    assert result.status == ToolStatus.NOT_FOUND
    assert store._drafts == {}


def test_explicit_project_draft_with_zero_or_ambiguous_matches_fails_closed():
    class _ProjectClient:
        def __init__(self, projects):
            self.projects = projects

        async def search_projects(self, query: str, *, limit: int = 10):
            return self.projects

    for projects in ([], [{"ID": "77", "NAME": "Ларгус-2"}, {"ID": "78", "NAME": "Ларгус-2"}]):
        store = FakeTaskDraftStore()
        result = _exec(
            TaskCreateDraftTool(store=store, project_client=_ProjectClient(projects)),
            {"title": "Тест", "responsible_self": True, "project_name": "Ларгус-2"},
            user_id=9,
            dialog_key="d:42",
        )
        assert result.status == ToolStatus.NOT_FOUND
        assert store._drafts == {}


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
    write_client.call.assert_awaited_once_with("tasks.task.add", {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9}})
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


def test_confirm_tool_rejects_stale_resolved_project_before_write():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:1",
            {
                "_draft_type": "task_create",
                "_resolved_project": {"id": 77, "name": "Ларгус-2"},
                "fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9, "GROUP_ID": 77},
            },
        )
    )

    class _RenamedProjectClient:
        def __init__(self):
            self.call_count = 0

        async def search_projects(self, query: str, *, limit: int = 10):
            return [{"ID": "77", "NAME": "Ларгус-3"}]

        async def call(self, method: str, params: dict):
            self.call_count += 1
            raise AssertionError("stale project reference must stop before write")

    oauth_client = _RenamedProjectClient()
    tool = TaskCreateConfirmTool(
        store=store,
        bitrix_oauth=FakeBitrixOAuth(oauth_client),
        dry_run=False,
        oauth_required_for_writes=True,
    )

    result = _exec(tool, {}, user_id=9, dialog_key="d:1", dialog_id="chat99")

    assert result.status == ToolStatus.NOT_FOUND
    assert oauth_client.call_count == 0
    assert "d:1" in store._drafts


def test_confirm_tool_revalidates_exact_resolved_project_before_write():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:1",
            {
                "_draft_type": "task_create",
                "_resolved_project": {"id": 77, "name": "Ларгус-2"},
                "fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9, "GROUP_ID": 77},
            },
        )
    )

    class _ExactProjectClient:
        def __init__(self):
            self.calls = []

        async def search_projects(self, query: str, *, limit: int = 10):
            return [{"ID": "77", "NAME": "Ларгус 2"}]

        async def call(self, method: str, params: dict):
            self.calls.append((method, params))
            return {"task": {"id": 777}}

    oauth_client = _ExactProjectClient()
    tool = TaskCreateConfirmTool(
        store=store,
        bitrix_oauth=FakeBitrixOAuth(oauth_client),
        dry_run=False,
        oauth_required_for_writes=True,
    )

    result = _exec(tool, {}, user_id=9, dialog_key="d:1", dialog_id="chat99")

    assert result.status == ToolStatus.OK
    assert oauth_client.calls == [
        (
            "tasks.task.add",
            {"fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9, "GROUP_ID": 77}},
        )
    ]
    assert "d:1" not in store._drafts


def test_confirm_tool_preserves_transient_project_validation_failure_status():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:1",
            {
                "_draft_type": "task_create",
                "_resolved_project": {"id": 77, "name": "Ларгус-2"},
                "fields": {"TITLE": "Задача", "RESPONSIBLE_ID": 9, "GROUP_ID": 77},
            },
        )
    )

    class _TransientFailureClient:
        def __init__(self):
            self.call_count = 0

        async def search_projects(self, query: str, *, limit: int = 10):
            raise RuntimeError("temporary read failure")

        async def call(self, method: str, params: dict):
            self.call_count += 1
            raise AssertionError("validation failure must stop before write")

    oauth_client = _TransientFailureClient()
    tool = TaskCreateConfirmTool(
        store=store,
        bitrix_oauth=FakeBitrixOAuth(oauth_client),
        dry_run=False,
        oauth_required_for_writes=True,
    )

    result = _exec(tool, {}, user_id=9, dialog_key="d:1", dialog_id="chat99")

    assert result.status == ToolStatus.ERROR
    assert oauth_client.call_count == 0
    assert "d:1" in store._drafts


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
