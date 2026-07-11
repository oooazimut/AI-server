"""Tests for Bitrix task closing draft tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, call

import anyio

from ai_server.agents.bitrix24.tools.task_close import (
    INCOMPLETE_CLOSE_MARKER,
    TASK_CLOSE_DRAFT_TYPE,
    TaskCloseConfirmTool,
    TaskCloseDraftTool,
)
from ai_server.models import ToolStatus
from tests.fakes import FakeTaskDraftStore


def _exec(tool, args, *, user_id=None, dialog_key=None, dialog_id=None):
    async def _run():
        return await tool.execute(args, user_id=user_id, dialog_key=dialog_key, dialog_id=dialog_id)

    return anyio.run(_run)


def test_close_draft_saves_incomplete_marker():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    result = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Обучение сотрудников",
            "completion_summary": "Пользователь подтвердил выполнение.",
            "unresolved_items": ["не приложен акт проверки"],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert result.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["_draft_type"] == TASK_CLOSE_DRAFT_TYPE
    assert draft["task_id"] == 139
    assert INCOMPLETE_CLOSE_MARKER in draft["result_text"]
    assert result.data["preview"]["unresolved_items"] == ["не приложен акт проверки"]


def test_close_draft_structures_problem_types_and_mobile_blocks():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    result = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Проверить камеры",
            "completion_summary": "Пользователь сообщил, что часть работ выполнена.",
            "task_points": ["камеры подключены", "архив не проверен"],
            "equipment_consumables": "4 камеры, 30 метров кабеля",
            "overall_status": "partial",
            "not_done_items": ["не проверен архив"],
            "unconfirmed_items": ["нет фото результата"],
            "missing_fields": ["причина, почему архив не проверен"],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert result.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["overall_status"] == "partial"
    assert draft["overall_status_label"] == "выполнена частично"
    assert draft["not_done_items"] == ["не проверен архив"]
    assert draft["unconfirmed_items"] == ["нет фото результата"]
    assert draft["unresolved_items"] == ["не проверен архив", "нет фото результата"]
    assert draft["problem_types"] == ["not_done", "unconfirmed"]
    assert draft["ai_close_incomplete"] is True
    assert draft["ai_close_marker"] == INCOMPLETE_CLOSE_MARKER
    assert "Оборудование, расходники: 4 камеры, 30 метров кабеля" in draft["result_text"]
    assert "Невыполненные пункты:" in draft["result_text"]
    assert "Неподтверждённые пункты:" in draft["result_text"]


def test_close_draft_denies_missing_dialog_id():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    result = _exec(
        tool,
        {"task_id": 139, "completion_summary": "Готово"},
        user_id=13,
        dialog_key="d:13",
        dialog_id=None,
    )

    assert result.status == ToolStatus.DENIED
    assert not store._drafts


def test_close_confirm_uses_oauth_result_then_complete():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:13",
            {
                "_draft_type": TASK_CLOSE_DRAFT_TYPE,
                "task_id": 139,
                "task_title": "Обучение сотрудников",
                "action": "complete",
                "result_text": "Выполнено",
                "unresolved_items": [],
            },
        )
    )
    oauth_client = AsyncMock()
    oauth_client.call = AsyncMock(side_effect=[{"result": 1}, {"result": True}])
    oauth = FakeBitrixOAuth(oauth_client)

    tool = TaskCloseConfirmTool(store=store, bitrix_oauth=oauth, oauth_required_for_writes=True)
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    assert oauth.user_ids == [13]
    assert oauth_client.call.await_args_list == [
        call("tasks.task.result.add", {"taskId": 139, "fields": {"TEXT": "Выполнено"}}),
        call("tasks.task.complete", {"taskId": 139}),
    ]
    assert "d:13" not in store._drafts


def test_close_confirm_can_approve_task():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:13",
            {
                "_draft_type": TASK_CLOSE_DRAFT_TYPE,
                "task_id": 139,
                "task_title": "Обучение сотрудников",
                "action": "approve",
                "result_text": "Результат принят",
                "unresolved_items": [],
            },
        )
    )
    oauth_client = AsyncMock()
    oauth_client.call = AsyncMock(side_effect=[{"result": 1}, {"result": True}])
    oauth = FakeBitrixOAuth(oauth_client)

    tool = TaskCloseConfirmTool(store=store, bitrix_oauth=oauth, oauth_required_for_writes=True)
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    assert oauth_client.call.await_args_list[-1] == call("tasks.task.approve", {"taskId": 139})


def test_close_confirm_missing_close_draft():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:13", {"fields": {"TITLE": "Задача"}}))

    tool = TaskCloseConfirmTool(store=store, bitrix_oauth=FakeBitrixOAuth(AsyncMock()))
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.NOT_FOUND


class FakeBitrixOAuth:
    def __init__(self, client) -> None:
        self.client = client
        self.user_ids = []

    async def client_for_user(self, user_id: int):
        self.user_ids.append(user_id)
        return self.client
