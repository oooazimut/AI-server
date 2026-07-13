"""Tests for Bitrix task closing draft tools."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, call

import anyio

from ai_server.agents.bitrix24.tools.task_close import (
    INCOMPLETE_CLOSE_MARKER,
    TASK_CLOSE_DRAFT_TYPE,
    TaskCloseConfirmTool,
    TaskCloseDiscardTool,
    TaskCloseDraftTool,
    TaskCloseReportIncidentTool,
)
from ai_server.integrations.bitrix.task_close_direct_queue import (
    TASK_CLOSE_DIRECT_STATUS_ACTIVE,
    TASK_CLOSE_DIRECT_STATUS_COMPLETED,
    TASK_CLOSE_DIRECT_STATUS_DISCARDED,
    direct_close_state_key,
)
from ai_server.integrations.bitrix.task_close_reports import (
    task_close_report_problem_types_from_text,
    task_close_report_state_key,
)
from ai_server.models import ToolStatus
from ai_server.settings import get_settings
from tests.fakes import FakePortalSearchIndex, FakeTaskDraftStore


class DirectCloseDraftStore(FakeTaskDraftStore):
    def __init__(self) -> None:
        super().__init__()
        self._task_close_processing_state: dict[tuple[str, str], dict] = {}

    def get_task_close_processing_state(self, *, task_id: object, state_key: str) -> dict | None:
        state = self._task_close_processing_state.get((str(task_id), state_key))
        return dict(state) if state else None

    def upsert_task_close_processing_state(
        self,
        *,
        task_id: object,
        state_key: str,
        status: str,
        payload: dict | None = None,
        actor_user_id: int | None = None,
    ) -> None:
        self._task_close_processing_state[(str(task_id), state_key)] = {
            "task_id": str(task_id),
            "state_key": state_key,
            "status": status,
            "payload": dict(payload or {}),
            "actor_user_id": actor_user_id,
        }


class _TaskDetailClient:
    def __init__(self, task: dict) -> None:
        self.task = task
        self.calls: list[tuple[str, dict]] = []

    async def result(self, method: str, params: dict) -> dict:
        self.calls.append((method, params))
        return {"task": self.task}


def test_ai_close_report_parser_prefers_machine_problem_types_over_human_prompts():
    text = """AI-close report
Задача #8993: Проверка

3. Статус выполнения работ
   (выполнено полностью / [ВЫБРАНО: выполнено частично] / не выполнено; если выполнено частично или не выполнено - укажи причину неполного выполнения работ)
3.1 причина: машинный блок будет проверен после индексации
3.2 Еще причины невыполнения - ... ???

---
Machine metadata
Status: unconfirmed
Problem types: unconfirmed
AI marker: AI_SERVER_TASK_CLOSE_INCOMPLETE
"""

    assert task_close_report_problem_types_from_text(text) == ["unconfirmed"]


def _exec(tool, args, *, user_id=None, dialog_key=None, dialog_id=None):
    async def _run():
        return await tool.execute(args, user_id=user_id, dialog_key=dialog_key, dialog_id=dialog_id)

    return anyio.run(_run)


def test_close_draft_with_only_task_id_loads_task_points_and_missing_fields():
    store = FakeTaskDraftStore()
    client = _TaskDetailClient(
        {
            "ID": "139",
            "TITLE": "Проверить камеры",
            "DESCRIPTION": "1. Проверить камеру на входе\n2. Проверить архив",
            "STATUS": "3",
        }
    )
    tool = TaskCloseDraftTool(store=store, read_client=client)

    result = _exec(tool, {"task_id": 139}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    assert client.calls == [
        ("tasks.task.get", {"taskId": 139, "select": ["ID", "TITLE", "DESCRIPTION", "STATUS", "CLOSED_DATE"]})
    ]
    draft = store._drafts["d:13"]
    assert draft["task_title"] == "Проверить камеры"
    assert draft["task_points"] == ["Проверить камеру на входе", "Проверить архив"]
    assert draft["overall_status"] == "unconfirmed"
    assert draft["unconfirmed_items"] == ["Проверить камеру на входе", "Проверить архив"]
    assert "что сделано по задаче" in draft["missing_fields"]
    assert draft["source_task_description_empty"] is False
    assert result.data["preview"]["task_points"] == ["Проверить камеру на входе", "Проверить архив"]


def test_close_draft_ignores_plain_close_command_summary_for_empty_task():
    store = FakeTaskDraftStore()
    client = _TaskDetailClient(
        {
            "ID": "139",
            "TITLE": "Пустая задача",
            "DESCRIPTION": "",
            "STATUS": "3",
        }
    )
    tool = TaskCloseDraftTool(store=store, read_client=client)

    result = _exec(
        tool,
        {"task_id": 139, "completion_summary": "закрой задачу 139"},
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert result.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["completion_summary"] == ""
    assert draft["task_points"] == []
    assert draft["source_task_description_empty"] is True
    assert draft["overall_status"] == "unconfirmed"
    assert draft["unconfirmed_items"]
    assert "закрой" not in " ".join(draft["unconfirmed_items"]).casefold()
    assert result.data["preview"]["completion_summary"] == ""


def test_close_draft_splits_plain_comma_description_into_task_points():
    store = FakeTaskDraftStore()
    client = _TaskDetailClient(
        {
            "ID": "139",
            "TITLE": "Проверить объект",
            "DESCRIPTION": "Протереть камеру, забрать документы, починить селектор.",
            "STATUS": "3",
        }
    )
    tool = TaskCloseDraftTool(store=store, read_client=client)

    result = _exec(tool, {"task_id": 139}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["task_points"] == ["Протереть камеру", "забрать документы", "починить селектор"]
    assert result.data["preview"]["task_points"] == [
        "Протереть камеру",
        "забрать документы",
        "починить селектор",
    ]


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
    assert draft["ai_close_marker"] == INCOMPLETE_CLOSE_MARKER
    assert INCOMPLETE_CLOSE_MARKER not in draft["result_text"]
    assert "Статус AI-закрытия: неподтвержденная" in draft["result_text"]
    assert result.data["preview"]["unresolved_items"] == ["не приложен акт проверки"]


def test_close_draft_marks_already_closed_task():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    result = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Обучение сотрудников",
            "completion_summary": "Нужно уточнить закрытие уже закрытой задачи.",
            "already_closed": True,
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert result.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["already_closed"] is True
    assert result.data["preview"]["already_closed"] is True


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


def test_close_draft_merges_same_task_update_without_resetting_known_fields():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    first = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Проверить камеры",
            "completion_summary": "Проверены камеры на входе.",
            "task_points": ["камеры проверены", "архив требует уточнения"],
            "overall_status": "unconfirmed",
            "unconfirmed_items": ["архив требует уточнения"],
            "missing_fields": ["что использовано"],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )
    second = _exec(
        tool,
        {
            "task_id": 139,
            "equipment_consumables": "4 камеры, 30 метров кабеля",
            "missing_fields": [],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert first.status == ToolStatus.OK
    assert second.status == ToolStatus.OK
    assert second.data["status"] == "updated_draft"
    draft = store._drafts["d:13"]
    assert draft["task_points"] == ["камеры проверены", "архив требует уточнения"]
    assert draft["completion_summary"] == "Проверены камеры на входе."
    assert draft["equipment_consumables"] == "4 камеры, 30 метров кабеля"
    assert draft["unconfirmed_items"] == ["архив требует уточнения"]
    assert draft["missing_fields"] == []


def test_close_draft_update_rewrites_named_work_point_status():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    first = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Проверить объект",
            "task_points": ["Проверить камеру", "Забрать документы", "Починить селектор"],
            "completion_summary": (
                "Проверить камеру выполнено полностью. "
                "Забрать документы не выполнено. "
                "Починить селектор не подтверждено"
            ),
            "overall_status": "partial",
            "not_done_items": ["Забрать документы"],
            "unconfirmed_items": ["Починить селектор"],
            "missing_fields": [],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )
    second = _exec(
        tool,
        {
            "task_id": 139,
            "completion_summary": "Забрать документы выполнено полностью",
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert first.status == ToolStatus.OK
    assert second.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["not_done_items"] == []
    assert draft["unconfirmed_items"] == ["Починить селектор"]
    assert "Проверить камеру выполнено полностью" in draft["completion_summary"]
    assert "Забрать документы выполнено полностью" in draft["completion_summary"]
    assert "Забрать документы не выполнено" not in draft["completion_summary"]
    assert "Починить селектор не подтверждено" in draft["completion_summary"]


def test_close_draft_update_completed_status_clears_previous_reasons():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    first = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Проверить объект",
            "task_points": ["Проверить камеру"],
            "completion_summary": "Проверить камеру не выполнено",
            "overall_status": "partial",
            "not_done_items": ["Проверить камеру"],
            "status_reasons": ["не было доступа к архиву"],
            "missing_fields": [],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )
    second = _exec(
        tool,
        {
            "task_id": 139,
            "completion_summary": "Проверить камеру выполнено полностью",
            "overall_status": "completed",
            "missing_fields": [],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert first.status == ToolStatus.OK
    assert second.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["overall_status"] == "completed"
    assert draft["not_done_items"] == []
    assert draft["status_reasons"] == []
    assert "Еще причины" not in draft["result_text"]
    assert "не было доступа к архиву" not in draft["result_text"]


def test_close_draft_update_removes_initial_unknown_result_placeholder():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    first = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Проверить камеры",
            "task_points": ["проверить камеру", "проверить архив"],
            "overall_status": "unconfirmed",
            "unconfirmed_items": ["результат выполнения не указан"],
            "missing_fields": ["что сделано по задаче"],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )
    second = _exec(
        tool,
        {
            "task_id": 139,
            "completion_summary": "Проверка выполнена частично, архив не подтвержден.",
            "overall_status": "partial",
            "unconfirmed_items": ["архив не подтвержден"],
            "missing_fields": [],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert first.status == ToolStatus.OK
    assert second.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["unconfirmed_items"] == ["архив не подтвержден"]
    assert draft["unresolved_items"] == ["архив не подтвержден"]
    assert "результат выполнения не указан" not in draft["result_text"]


def test_close_draft_update_recomputes_unresolved_items_from_current_lists():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    first = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Check object",
            "task_points": ["Check camera", "Take documents", "Fix selector"],
            "overall_status": "unconfirmed",
            "unconfirmed_items": ["Check camera", "Take documents", "Fix selector"],
            "missing_fields": ["result"],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )
    second = _exec(
        tool,
        {
            "task_id": 139,
            "completion_summary": "Check camera completed, Take documents not done, Fix selector unconfirmed.",
            "overall_status": "partial",
            "not_done_items": ["Take documents", "No archive access"],
            "unconfirmed_items": ["Fix selector"],
            "missing_fields": [],
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert first.status == ToolStatus.OK
    assert second.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["not_done_items"] == ["Take documents", "No archive access"]
    assert draft["unconfirmed_items"] == ["Fix selector"]
    assert draft["unresolved_items"] == ["Take documents", "No archive access", "Fix selector"]


def test_close_draft_marks_empty_source_description_for_freeform_result():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    result = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Пустая задача",
            "source_task_description_empty": True,
            "completion_summary": "Проверил объект, устранил замечания.",
            "overall_status": "completed",
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert result.status == ToolStatus.OK
    draft = store._drafts["d:13"]
    assert draft["source_task_description_empty"] is True
    assert draft["task_points"] == []
    assert result.data["preview"]["source_task_description_empty"] is True


def test_close_draft_blocks_other_task_when_active_draft_exists():
    store = FakeTaskDraftStore()
    tool = TaskCloseDraftTool(store=store)

    first = _exec(
        tool,
        {
            "task_id": 139,
            "task_title": "Проверить камеры",
            "completion_summary": "Проверены камеры на входе.",
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )
    second = _exec(
        tool,
        {
            "task_id": 140,
            "task_title": "Проверить регистратор",
            "completion_summary": "Регистратор проверен.",
        },
        user_id=13,
        dialog_key="d:13",
        dialog_id="chat4321",
    )

    assert first.status == ToolStatus.OK
    assert second.status == ToolStatus.OK
    assert second.data["status"] == "active_draft_conflict"
    assert second.data["conflict"]["current_task_id"] == 139
    assert second.data["conflict"]["requested_task_id"] == 140
    assert store._drafts["d:13"]["task_id"] == 139
    assert store._drafts["d:13"]["task_title"] == "Проверить камеры"
    assert store._drafts["d:13"]["_task_close_conflict_pending"]["task_id"] == 140
    assert store._drafts["d:13"]["_task_close_conflict_pending"]["task_title"] == "Проверить регистратор"


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


def test_close_confirm_uses_oauth_close_then_system_report_file():
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
    oauth_client.call = AsyncMock(return_value={"result": True})
    oauth = FakeBitrixOAuth(oauth_client)
    system_client = AsyncMock()
    system_client.call = AsyncMock(
        side_effect=[
            {"result": []},
            {"result": {"ATTACHMENT_ID": 5509, "FILE_ID": 62357, "NAME": "AI-close-139.txt"}},
        ]
    )

    tool = TaskCloseConfirmTool(
        store=store,
        write_client=system_client,
        bitrix_oauth=oauth,
        oauth_required_for_writes=True,
    )
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    assert oauth.user_ids == [13]
    assert oauth_client.call.await_args_list == [call("tasks.task.complete", {"taskId": 139})]
    assert system_client.call.await_args_list[0] == call("task.item.getfiles", {"taskId": 139})
    addfile_call = system_client.call.await_args_list[1]
    assert addfile_call.args[0] == "task.item.addfile"
    addfile_payload = addfile_call.args[1]
    assert addfile_payload["taskId"] == 139
    assert addfile_payload["fileParameters"]["NAME"] == "AI-close-139.txt"
    report_text = base64.b64decode(addfile_payload["fileParameters"]["CONTENT"]).decode("utf-8")
    assert "AI-close report" in report_text
    assert "Machine metadata" in report_text
    assert "AI task close report" not in report_text
    assert "Task ID: 139" in report_text
    assert "Status: ok" in report_text
    assert "Выполнено" in report_text
    assert "\n2.1 " not in report_text
    assert "\n4.1 " not in report_text
    assert "Внести изменения" not in report_text
    assert result.data["report_file_name"] == "AI-close-139.txt"
    assert result.data["report_file_owner"] == "system_bitrix_client"
    assert "d:13" not in store._drafts


def test_close_confirm_updates_existing_system_report_file_version():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:13",
            {
                "_draft_type": TASK_CLOSE_DRAFT_TYPE,
                "task_id": 139,
                "task_title": "Task",
                "action": "complete",
                "result_text": "Done with unconfirmed status",
                "overall_status": "unconfirmed",
                "ai_close_incomplete": True,
                "ai_close_marker": INCOMPLETE_CLOSE_MARKER,
                "unresolved_items": ["no photo"],
                "unconfirmed_items": ["no photo"],
            },
        )
    )
    oauth_client = AsyncMock()
    oauth_client.call = AsyncMock(return_value={"result": True})
    oauth = FakeBitrixOAuth(oauth_client)
    system_client = AsyncMock()
    existing = {"ATTACHMENT_ID": 5509, "FILE_ID": 62357, "NAME": "AI-close-139-unconfirmed.txt"}
    system_client.call = AsyncMock(return_value={"result": [existing]})

    tool = TaskCloseConfirmTool(
        store=store,
        write_client=system_client,
        bitrix_oauth=oauth,
        oauth_required_for_writes=True,
    )
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    assert result.data["result_method"] == "disk.file.uploadversion"
    assert result.data["report_file_name"] == "AI-close-139.txt"
    assert oauth_client.call.await_args_list == [
        call("tasks.task.complete", {"taskId": 139}),
    ]
    assert system_client.call.await_args_list[0] == call("task.item.getfiles", {"taskId": 139})
    upload_call = system_client.call.await_args_list[1]
    assert upload_call.args[0] == "disk.file.uploadversion"
    upload_payload = upload_call.args[1]
    assert upload_payload["id"] == 62357
    assert upload_payload["fileContent"][0] == "AI-close-139.txt"
    uploaded_report = base64.b64decode(upload_payload["fileContent"][1]).decode("utf-8")
    assert "Status: unconfirmed" in uploaded_report
    assert "Problem types: unconfirmed" in uploaded_report
    assert "AI marker: AI_SERVER_TASK_CLOSE_INCOMPLETE" in uploaded_report
    assert "d:13" not in store._drafts


def test_close_confirm_already_closed_draft_skips_reclosing_and_adds_report_file():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:13",
            {
                "_draft_type": TASK_CLOSE_DRAFT_TYPE,
                "task_id": 139,
                "task_title": "Task",
                "action": "complete",
                "result_text": "Updated AI close report",
                "already_closed": True,
            },
        )
    )
    oauth_client = AsyncMock()
    oauth_client.call = AsyncMock(return_value={"result": True})
    oauth = FakeBitrixOAuth(oauth_client)
    system_client = AsyncMock()
    system_client.call = AsyncMock(
        side_effect=[
            {"result": []},
            {"result": {"ATTACHMENT_ID": 5509, "FILE_ID": 62357, "NAME": "AI-close-139-ok.txt"}},
        ]
    )

    tool = TaskCloseConfirmTool(
        store=store,
        write_client=system_client,
        bitrix_oauth=oauth,
        oauth_required_for_writes=True,
    )
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    assert oauth.user_ids == [13]
    assert oauth_client.call.await_args_list == []
    assert result.data["close_method"] == "already_closed"
    assert result.data["close_result"] == {"skipped": True, "reason": "task_already_closed"}
    assert system_client.call.await_args_list[0] == call("task.item.getfiles", {"taskId": 139})
    assert system_client.call.await_args_list[1].args[0] == "task.item.addfile"
    assert "d:13" not in store._drafts


def test_close_confirm_direct_close_draft_skips_reclosing_and_completes_queue():
    store = DirectCloseDraftStore()
    close_event_key = "closed_at:2026-07-12T12:00:00+03:00"
    state_key = direct_close_state_key(close_event_key)
    store.upsert_task_close_processing_state(
        task_id=139,
        state_key=state_key,
        status=TASK_CLOSE_DIRECT_STATUS_ACTIVE,
        payload={"close_event_key": close_event_key},
    )
    anyio.run(
        lambda: store.save_task_draft(
            "d:13",
            {
                "_draft_type": TASK_CLOSE_DRAFT_TYPE,
                "task_id": 139,
                "task_title": "Task",
                "action": "complete",
                "result_text": "Done with unconfirmed status",
                "overall_status": "unconfirmed",
                "ai_close_incomplete": True,
                "ai_close_marker": INCOMPLETE_CLOSE_MARKER,
                "unresolved_items": ["no photo"],
                "unconfirmed_items": ["no photo"],
                "_direct_close_already_closed": True,
                "_direct_close_close_event_key": close_event_key,
            },
        )
    )
    oauth_client = AsyncMock()
    oauth_client.call = AsyncMock(return_value={"result": True})
    oauth = FakeBitrixOAuth(oauth_client)
    system_client = AsyncMock()
    system_client.call = AsyncMock(
        side_effect=[
            {"result": []},
            {"result": {"ATTACHMENT_ID": 5509, "FILE_ID": 62357, "NAME": "AI-close-139-unconfirmed.txt"}},
        ]
    )

    tool = TaskCloseConfirmTool(
        store=store,
        write_client=system_client,
        bitrix_oauth=oauth,
        oauth_required_for_writes=True,
    )
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    assert result.data["close_method"] == "already_closed"
    assert result.data["close_result"] == {"skipped": True, "reason": "task_already_closed_directly"}
    assert oauth_client.call.await_args_list == []
    assert system_client.call.await_args_list[0] == call("task.item.getfiles", {"taskId": 139})
    assert system_client.call.await_args_list[1].args[0] == "task.item.addfile"
    state = store.get_task_close_processing_state(task_id=139, state_key=state_key)
    assert state is not None
    assert state["status"] == TASK_CLOSE_DIRECT_STATUS_COMPLETED
    assert "d:13" not in store._drafts


def test_close_discard_direct_close_draft_discards_queue():
    store = DirectCloseDraftStore()
    close_event_key = "closed_at:2026-07-12T12:00:00+03:00"
    state_key = direct_close_state_key(close_event_key)
    store.upsert_task_close_processing_state(
        task_id=139,
        state_key=state_key,
        status=TASK_CLOSE_DIRECT_STATUS_ACTIVE,
        payload={"close_event_key": close_event_key},
    )
    anyio.run(
        lambda: store.save_task_draft(
            "d:13",
            {
                "_draft_type": TASK_CLOSE_DRAFT_TYPE,
                "task_id": 139,
                "_direct_close_close_event_key": close_event_key,
            },
        )
    )

    result = _exec(TaskCloseDiscardTool(store=store), {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    state = store.get_task_close_processing_state(task_id=139, state_key=state_key)
    assert state is not None
    assert state["status"] == TASK_CLOSE_DIRECT_STATUS_DISCARDED
    assert "d:13" not in store._drafts


def test_close_confirm_denies_without_system_report_client():
    store = FakeTaskDraftStore()
    anyio.run(
        lambda: store.save_task_draft(
            "d:13",
            {
                "_draft_type": TASK_CLOSE_DRAFT_TYPE,
                "task_id": 139,
                "action": "complete",
                "result_text": "Выполнено",
            },
        )
    )
    oauth_client = AsyncMock()
    oauth = FakeBitrixOAuth(oauth_client)

    tool = TaskCloseConfirmTool(store=store, bitrix_oauth=oauth, oauth_required_for_writes=True)
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.NOT_CONFIGURED
    assert "write_client" in result.error
    assert oauth_client.call.await_args_list == []
    assert "d:13" in store._drafts


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
    oauth_client.call = AsyncMock(return_value={"result": True})
    oauth = FakeBitrixOAuth(oauth_client)
    system_client = AsyncMock()
    system_client.call = AsyncMock(
        side_effect=[
            {"result": []},
            {"result": {"ATTACHMENT_ID": 5510, "FILE_ID": 62358, "NAME": "AI-close-139-ok.txt"}},
        ]
    )

    tool = TaskCloseConfirmTool(
        store=store,
        write_client=system_client,
        bitrix_oauth=oauth,
        oauth_required_for_writes=True,
    )
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.OK
    assert oauth_client.call.await_args_list == [call("tasks.task.approve", {"taskId": 139})]


def test_close_confirm_missing_close_draft():
    store = FakeTaskDraftStore()
    anyio.run(lambda: store.save_task_draft("d:13", {"fields": {"TITLE": "Задача"}}))

    tool = TaskCloseConfirmTool(store=store, bitrix_oauth=FakeBitrixOAuth(AsyncMock()))
    result = _exec(tool, {}, user_id=13, dialog_key="d:13", dialog_id="chat4321")

    assert result.status == ToolStatus.NOT_FOUND


def test_task_close_report_incident_restore_readds_missing_report(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_TASK_CLOSE_REPORT_ADMIN_USER_IDS", "1")
    index = FakePortalSearchIndex()
    index.upsert_item(
        entity_type="task",
        entity_id=139,
        title="Task",
        body="AI close marker: AI_SERVER_TASK_CLOSE_INCOMPLETE",
        metadata={
            "ai_close_report_files": [
                {
                    "name": "AI-close-139-unconfirmed.txt",
                    "attached_object_id": 5509,
                    "disk_object_id": 62357,
                    "problem_types": ["unconfirmed"],
                }
            ],
            "ai_close_report_missing": True,
            "ai_close_report_missing_files": [
                {
                    "name": "AI-close-139-unconfirmed.txt",
                    "attached_object_id": 5509,
                    "disk_object_id": 62357,
                    "problem_types": ["unconfirmed"],
                }
            ],
            "ai_close_report_incident_status": "pending",
            "ai_close_incomplete": True,
            "ai_close_marker": "AI_SERVER_TASK_CLOSE_INCOMPLETE",
            "ai_close_problem_types": ["unconfirmed"],
            "ai_close_marker_source": "task_attachment",
        },
    )

    class RestoreClient:
        def __init__(self) -> None:
            self.addfile_payloads: list[dict] = []

        async def get_disk_file_download_url(self, file_id: int) -> str:
            return f"fake://disk/{file_id}"

        async def download_file_from_url(self, url: str, destination: Path, *, max_bytes: int):
            data = b"AI task close report\nStatus: unconfirmed\n"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            return len(data)

        async def result(self, method: str, payload: dict):
            assert method == "task.item.addfile"
            self.addfile_payloads.append(payload)
            return {"ATTACHMENT_ID": 901, "FILE_ID": 62358, "NAME": "AI-close-139-unconfirmed (1).txt"}

    client = RestoreClient()
    tool = TaskCloseReportIncidentTool(client=client, portal_search=index, settings=get_settings())
    result = _exec(
        tool,
        {"task_id": 139, "action": "restore"},
        user_id=1,
        dialog_key="d:1",
        dialog_id="1",
    )

    assert result.status == ToolStatus.OK
    assert client.addfile_payloads[0]["fileParameters"]["NAME"] == "AI-close-139-unconfirmed.txt"
    task = index.get_item(entity_type="task", entity_id=139)
    assert task is not None
    assert task.metadata["ai_close_report_missing"] is False
    assert task.metadata["ai_close_report_incident_status"] == "restored"
    assert task.metadata["ai_close_report_files"] == [
        {
            "name": "AI-close-139-unconfirmed (1).txt",
            "attached_object_id": 901,
            "disk_object_id": 62358,
            "problem_types": ["unconfirmed"],
        }
    ]


def test_task_close_report_incident_accept_missing_clears_index_metadata(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_TASK_CLOSE_REPORT_ADMIN_USER_IDS", "1")
    index = FakePortalSearchIndex()
    report_file = {
        "name": "AI-close-139-unconfirmed.txt",
        "attached_object_id": 5509,
        "disk_object_id": 62357,
        "problem_types": ["unconfirmed"],
    }
    index.upsert_item(
        entity_type="task",
        entity_id=139,
        title="Task",
        body=("Task body\nAI close marker: AI_SERVER_TASK_CLOSE_INCOMPLETE\nAI close problem types: unconfirmed"),
        metadata={
            "ai_close_report_files": [report_file],
            "ai_close_report_missing": True,
            "ai_close_report_missing_files": [report_file],
            "ai_close_report_incident_status": "pending",
            "ai_close_incomplete": True,
            "ai_close_marker": "AI_SERVER_TASK_CLOSE_INCOMPLETE",
            "ai_close_problem_types": ["unconfirmed"],
            "ai_close_marker_source": "task_attachment",
        },
    )
    index.upsert_item(
        entity_type="task_attachment",
        entity_id=5509,
        title="AI-close-139-unconfirmed.txt",
        metadata={"ai_close_report": True},
    )

    tool = TaskCloseReportIncidentTool(portal_search=index, settings=get_settings())
    result = _exec(
        tool,
        {"task_id": 139, "action": "accept_missing"},
        user_id=1,
        dialog_key="d:1",
        dialog_id="1",
    )

    assert result.status == ToolStatus.OK
    assert index.get_item(entity_type="task_attachment", entity_id=5509) is None
    state = index.get_task_close_processing_state(task_id=139, state_key=task_close_report_state_key(report_file))
    assert state is not None
    assert state["status"] == "accepted_missing"
    assert state["actor_user_id"] == 1
    task = index.get_item(entity_type="task", entity_id=139)
    assert task is not None
    assert "ai_close_report_missing" not in task.metadata
    assert task.metadata["ai_close_report_files"] == []
    assert task.metadata["ai_close_incomplete"] is False
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" not in task.body


class FakeBitrixOAuth:
    def __init__(self, client) -> None:
        self.client = client
        self.user_ids = []

    async def client_for_user(self, user_id: int):
        self.user_ids.append(user_id)
        return self.client
