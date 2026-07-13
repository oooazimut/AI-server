from __future__ import annotations

import base64
from datetime import datetime

import anyio

from ai_server.integrations.bitrix.client import BitrixApiError
from ai_server.integrations.bitrix.oauth import BitrixOAuthTokenMissing
from ai_server.integrations.bitrix.task_close_direct_queue import (
    TASK_CLOSE_DIRECT_STATUS_ACTIVE,
    TASK_CLOSE_DIRECT_STATUS_AUTO_CLOSED_UNCONFIRMED,
    TASK_CLOSE_DIRECT_STATUS_PENDING,
    activate_next_direct_close_event,
    direct_close_state_key,
    enqueue_direct_close_event,
)
from ai_server.settings import get_settings
from ai_server.utils import MOSCOW_TZ
from ai_server.workers.bitrix.task_close_direct_dispatcher import (
    auto_close_direct_task_close_reports,
    dispatch_direct_task_close_drafts,
    run_task_close_direct_control_once,
)
from tests.fakes import FakePortalSearchIndex


class DraftQueueStore(FakePortalSearchIndex):
    def __init__(self) -> None:
        super().__init__()
        self._drafts: dict[str, dict] = {}

    async def save_task_draft(self, dialog_key: str, params: dict) -> None:
        self._drafts[dialog_key] = dict(params)

    async def get_task_draft(self, dialog_key: str, *, ttl_minutes: int | None = None) -> dict | None:
        return self._drafts.get(dialog_key)

    async def delete_task_draft(self, dialog_key: str) -> None:
        self._drafts.pop(dialog_key, None)

    def list_task_drafts(self, *, draft_type: str = "", limit: int = 100) -> list[dict]:
        result = []
        for dialog_key, params in self._drafts.items():
            if draft_type and params.get("_draft_type") != draft_type:
                continue
            result.append({"dialog_key": dialog_key, "params": dict(params), "created_at": "2026-07-12T19:00:00+03:00"})
        return result[:limit]


class RecordingBitrix:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, int | None]] = []
        self.calls: list[tuple[str, dict]] = []

    async def send_bot_message(self, dialog_id: str, message: str, *, bot_id=None, keyboard=None):
        self.messages.append((dialog_id, message, bot_id))
        return {"message_id": len(self.messages)}

    async def call(self, method: str, payload: dict):
        self.calls.append((method, payload))
        if method == "task.item.getfiles":
            return {"result": []}
        if method == "task.item.addfile":
            return {"result": {"ATTACHMENT_ID": 5523, "FILE_ID": 62373, "NAME": payload["fileParameters"]["NAME"]}}
        if method == "disk.file.uploadversion":
            return {"result": {"ID": payload["id"], "NAME": payload["fileContent"][0]}}
        return {"result": True}


class RecordingOAuthClient:
    def __init__(self, *, fail_close: bool = False) -> None:
        self.fail_close = fail_close
        self.calls: list[tuple[str, dict]] = []

    async def call(self, method: str, payload: dict):
        self.calls.append((method, payload))
        if self.fail_close and method in {"tasks.task.complete", "tasks.task.approve"}:
            raise BitrixApiError(method, "ACCESS_DENIED", "oauth denied")
        return {"result": True}


class RecordingOAuth:
    def __init__(self, client: RecordingOAuthClient | None = None, *, missing: bool = False) -> None:
        self.client = client
        self.missing = missing
        self.user_ids: list[int] = []

    async def client_for_user(self, user_id: int):
        self.user_ids.append(user_id)
        if self.missing:
            raise BitrixOAuthTokenMissing(user_id)
        return self.client


def test_dispatcher_creates_direct_close_draft_and_sends_message(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    enqueue_direct_close_event(
        store,
        task_id=8875,
        close_event_key="closed_at:2026-07-12T12:00:00+03:00",
        responsible_id=231,
        dialog_key="231",
        closed_at="2026-07-12T12:00:00+03:00",
        task_title="Проверить камеры",
        payload={
            "recipient_id": "231",
            "draft_dialog_key": "dialog:231:user:231",
            "task_results": ["Камеры проверены"],
            "task_points": ["Проверить камеры", "Проверить архив"],
            "source_task_description_empty": False,
        },
    )
    activate_next_direct_close_event(store, responsible_id=231, dialog_key="231")

    stats = anyio.run(lambda: dispatch_direct_task_close_drafts(bitrix=bitrix, store=store, settings=get_settings()))

    assert stats.drafts_created == 1
    assert stats.messages_sent == 1
    assert "dialog:231:user:231" in store._drafts
    draft = store._drafts["dialog:231:user:231"]
    assert draft["_draft_type"] == "task_close"
    assert draft["already_closed"] is True
    assert draft["_direct_close_already_closed"] is True
    assert draft["_direct_close_close_event_key"] == "closed_at:2026-07-12T12:00:00+03:00"
    assert draft["completion_summary"] == "Камеры проверены"
    assert bitrix.messages[0][0] == "231"
    assert "Задача закрыта напрямую в Bitrix" in bitrix.messages[0][1]
    state = store.get_task_close_processing_state(
        task_id=8875,
        state_key=direct_close_state_key("closed_at:2026-07-12T12:00:00+03:00"),
    )
    assert state is not None
    assert state["status"] == TASK_CLOSE_DIRECT_STATUS_ACTIVE
    assert state["payload"]["direct_close_draft_sent_at"]


def test_dispatcher_does_not_send_duplicate_message(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    enqueue_direct_close_event(
        store,
        task_id=8875,
        close_event_key="event-a",
        responsible_id=231,
        dialog_key="231",
        payload={"recipient_id": "231", "draft_dialog_key": "dialog:231:user:231"},
    )
    activate_next_direct_close_event(store, responsible_id=231, dialog_key="231")

    anyio.run(lambda: dispatch_direct_task_close_drafts(bitrix=bitrix, store=store, settings=get_settings()))
    stats = anyio.run(lambda: dispatch_direct_task_close_drafts(bitrix=bitrix, store=store, settings=get_settings()))

    assert stats.skipped == 1
    assert len(bitrix.messages) == 1


def test_dispatcher_blocks_when_another_draft_exists(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    store._drafts["dialog:231:user:231"] = {"_draft_type": "task_close", "task_id": 999}
    enqueue_direct_close_event(
        store,
        task_id=8875,
        close_event_key="event-a",
        responsible_id=231,
        dialog_key="231",
        payload={"recipient_id": "231", "draft_dialog_key": "dialog:231:user:231"},
    )
    activate_next_direct_close_event(store, responsible_id=231, dialog_key="231")

    stats = anyio.run(lambda: dispatch_direct_task_close_drafts(bitrix=bitrix, store=store, settings=get_settings()))

    assert stats.blocked == 1
    assert bitrix.messages == []
    state = store.get_task_close_processing_state(task_id=8875, state_key=direct_close_state_key("event-a"))
    assert state is not None
    assert state["payload"]["dispatch_blocked_reason"] == "another_active_draft"


def test_dispatcher_ignores_pending_queue_item(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    enqueue_direct_close_event(
        store,
        task_id=8875,
        close_event_key="event-a",
        responsible_id=231,
        dialog_key="231",
        payload={"recipient_id": "231", "draft_dialog_key": "dialog:231:user:231"},
    )

    stats = anyio.run(lambda: dispatch_direct_task_close_drafts(bitrix=bitrix, store=store, settings=get_settings()))

    assert stats.candidates == 0
    assert bitrix.messages == []
    state = store.get_task_close_processing_state(task_id=8875, state_key=direct_close_state_key("event-a"))
    assert state is not None
    assert state["status"] == TASK_CLOSE_DIRECT_STATUS_PENDING


def test_control_worker_once_dispatches_and_records_status(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    enqueue_direct_close_event(
        store,
        task_id=8875,
        close_event_key="event-a",
        responsible_id=231,
        dialog_key="231",
        payload={"recipient_id": "231", "draft_dialog_key": "dialog:231:user:231"},
    )
    activate_next_direct_close_event(store, responsible_id=231, dialog_key="231")
    status: dict = {}

    result = anyio.run(
        lambda: run_task_close_direct_control_once(
            bitrix=bitrix,
            store=store,
            settings=get_settings(),
            status=status,
            now=datetime(2026, 7, 12, 19, 0, tzinfo=MOSCOW_TZ),
        )
    )

    assert result["dispatch"]["drafts_created"] == 1
    assert result["auto_close"]["due"] is False
    assert status["last_check_at"] == "2026-07-12T19:00:00+03:00"
    assert status["dispatch"]["messages_sent"] == 1
    assert bitrix.messages


def test_auto_close_direct_queue_writes_unknown_report_when_no_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    enqueue_direct_close_event(
        store,
        task_id=8875,
        close_event_key="event-a",
        responsible_id=231,
        dialog_key="231",
        payload={"recipient_id": "231", "draft_dialog_key": "dialog:231:user:231"},
    )
    activate_next_direct_close_event(store, responsible_id=231, dialog_key="231")

    stats = anyio.run(
        lambda: auto_close_direct_task_close_reports(
            bitrix=bitrix,
            store=store,
            settings=get_settings(),
            now=datetime(2026, 7, 12, 20, 1, tzinfo=MOSCOW_TZ),
        )
    )

    assert stats.due is True
    assert stats.candidates == 1
    assert stats.reports_written == 1
    assert [method for method, _ in bitrix.calls] == ["task.item.getfiles", "task.item.addfile"]
    add_payload = bitrix.calls[1][1]
    report_text = base64.b64decode(add_payload["fileParameters"]["CONTENT"]).decode("utf-8")
    assert "Status: unconfirmed" in report_text
    assert "No confirmed draft was received before the auto-close time." in report_text
    state = store.get_task_close_processing_state(task_id=8875, state_key=direct_close_state_key("event-a"))
    assert state is not None
    assert state["status"] == TASK_CLOSE_DIRECT_STATUS_AUTO_CLOSED_UNCONFIRMED


def test_auto_close_direct_queue_preserves_partial_existing_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    store._drafts["dialog:231:user:231"] = {
        "_draft_type": "task_close",
        "task_id": 8875,
        "task_title": "Check cameras",
        "action": "complete",
        "completion_summary": "Two cameras checked, archive still needs verification.",
        "equipment_consumables": "2 cameras, 15m cable",
        "overall_status": "partial",
        "not_done_items": ["Archive was not checked"],
        "missing_fields": ["Need archive screenshot"],
    }
    enqueue_direct_close_event(
        store,
        task_id=8875,
        close_event_key="event-a",
        responsible_id=231,
        dialog_key="231",
        payload={"recipient_id": "231", "draft_dialog_key": "dialog:231:user:231"},
    )
    activate_next_direct_close_event(store, responsible_id=231, dialog_key="231")

    stats = anyio.run(
        lambda: auto_close_direct_task_close_reports(
            bitrix=bitrix,
            store=store,
            settings=get_settings(),
            now=datetime(2026, 7, 12, 20, 1, tzinfo=MOSCOW_TZ),
        )
    )

    assert stats.reports_written == 1
    assert "dialog:231:user:231" not in store._drafts
    add_payload = bitrix.calls[1][1]
    report_text = base64.b64decode(add_payload["fileParameters"]["CONTENT"]).decode("utf-8")
    assert "Two cameras checked, archive still needs verification." in report_text
    assert "2 cameras, 15m cable" in report_text
    assert "Archive was not checked" in report_text
    assert "Need archive screenshot" in report_text
    assert "Draft was not confirmed before the auto-close time." in report_text
    state = store.get_task_close_processing_state(task_id=8875, state_key=direct_close_state_key("event-a"))
    assert state is not None
    assert state["status"] == TASK_CLOSE_DIRECT_STATUS_AUTO_CLOSED_UNCONFIRMED


def test_auto_close_open_draft_uses_oauth_user_first(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    oauth_client = RecordingOAuthClient()
    oauth = RecordingOAuth(oauth_client)
    store._drafts["dialog:231:user:231"] = {
        "_draft_type": "task_close",
        "task_id": 8875,
        "task_title": "Check cameras",
        "action": "complete",
        "completion_summary": "Cameras checked.",
        "overall_status": "completed",
    }

    stats = anyio.run(
        lambda: auto_close_direct_task_close_reports(
            bitrix=bitrix,
            bitrix_oauth=oauth,
            store=store,
            settings=get_settings(),
            now=datetime(2026, 7, 12, 20, 1, tzinfo=MOSCOW_TZ),
        )
    )

    assert stats.open_drafts == 1
    assert stats.oauth_closed == 1
    assert stats.system_fallback_closed == 0
    assert oauth.user_ids == [231]
    assert oauth_client.calls == [("tasks.task.complete", {"taskId": 8875})]
    assert [method for method, _ in bitrix.calls] == ["task.item.getfiles", "task.item.addfile"]
    assert "dialog:231:user:231" not in store._drafts


def test_auto_close_open_draft_falls_back_to_system_webhook_when_oauth_missing(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    oauth = RecordingOAuth(missing=True)
    store._drafts["dialog:231:user:231"] = {
        "_draft_type": "task_close",
        "task_id": 8875,
        "task_title": "Check cameras",
        "action": "complete",
        "completion_summary": "Cameras checked.",
        "overall_status": "completed",
    }

    stats = anyio.run(
        lambda: auto_close_direct_task_close_reports(
            bitrix=bitrix,
            bitrix_oauth=oauth,
            store=store,
            settings=get_settings(),
            now=datetime(2026, 7, 12, 20, 1, tzinfo=MOSCOW_TZ),
        )
    )

    assert stats.open_drafts == 1
    assert stats.oauth_closed == 0
    assert stats.system_fallback_closed == 1
    assert stats.admin_notifications == 1
    assert [method for method, _ in bitrix.calls] == ["tasks.task.complete", "task.item.getfiles", "task.item.addfile"]
    report_text = base64.b64decode(bitrix.calls[2][1]["fileParameters"]["CONTENT"]).decode("utf-8")
    assert "Auto-close used system webhook fallback" in report_text
    assert "OAuth token for Bitrix user #231 is not linked" in report_text
    assert bitrix.messages[0][0] == "1"
    assert "Автозакрытие задачи выполнено системным webhook." in bitrix.messages[0][1]
    assert "OAuth token for Bitrix user #231 is not linked" in bitrix.messages[0][1]
    assert "dialog:231:user:231" not in store._drafts
