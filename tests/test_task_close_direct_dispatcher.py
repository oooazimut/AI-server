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

    async def delete_task_draft(
        self,
        dialog_key: str,
        *,
        status: str = "cancelled",
        expected_draft_id: str = "",
        expected_version: int | None = None,
        expected_claim_token: str = "",
    ) -> None:
        self._drafts.pop(dialog_key, None)

    def list_task_drafts(
        self,
        *,
        draft_type: str = "",
        limit: int = 100,
        expired_only: bool = False,
    ) -> list[dict]:
        result = []
        for dialog_key, params in self._drafts.items():
            if draft_type and params.get("_draft_type") != draft_type:
                continue
            expires_at = str(params.get("_draft_expires_at") or "")
            if expired_only and expires_at:
                parsed_expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                now = datetime.now(parsed_expiry.tzinfo or MOSCOW_TZ)
                if parsed_expiry > now:
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


class RecordingOutbox:
    def __init__(self) -> None:
        self.items: list[dict] = []

    async def enqueue(self, **kwargs):
        self.items.append(dict(kwargs))
        return "delivery-task-close-1", True


class RecordingOAuthClient:
    def __init__(self, *, fail_close: bool = False, task_closed: bool = False) -> None:
        self.fail_close = fail_close
        self.task_closed = task_closed
        self.calls: list[tuple[str, dict]] = []

    async def call(self, method: str, payload: dict):
        self.calls.append((method, payload))
        if method == "tasks.task.get":
            return {
                "result": {
                    "task": {
                        "ID": str(payload["taskId"]),
                        "STATUS": "5" if self.task_closed else "3",
                        "CLOSED_DATE": "2026-07-12T11:00:00+03:00" if self.task_closed else "",
                    }
                }
            }
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
    message = bitrix.messages[0][1]
    assert "Задача закрыта напрямую в Bitrix" in message
    assert "Черновик #8875" in message
    assert "1. Выполняемые работы" in message
    assert "1.1 Проверить камеры" in message
    assert "1.3 Еще работы - ... ???" in message
    assert "2. Использовано материалов, оборудование" in message
    assert "3. Статус выполнения работ" in message
    assert "4. Дополнительная информация" in message
    assert "Пункты задачи:" not in message
    state = store.get_task_close_processing_state(
        task_id=8875,
        state_key=direct_close_state_key("closed_at:2026-07-12T12:00:00+03:00"),
    )
    assert state is not None
    assert state["status"] == TASK_CLOSE_DIRECT_STATUS_ACTIVE
    assert state["payload"]["direct_close_draft_sent_at"]


def test_dispatcher_queues_direct_close_draft_without_contacting_bitrix(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    outbox = RecordingOutbox()
    enqueue_direct_close_event(
        store,
        task_id=8877,
        close_event_key="closed_at:2026-07-12T12:00:00+03:00",
        responsible_id=231,
        dialog_key="231",
        closed_at="2026-07-12T12:00:00+03:00",
        task_title="Queued close",
        payload={
            "recipient_id": "231",
            "draft_dialog_key": "dialog:231:user:231",
            "task_results": ["Done"],
        },
    )
    activate_next_direct_close_event(store, responsible_id=231, dialog_key="231")

    stats = anyio.run(
        lambda: dispatch_direct_task_close_drafts(
            bitrix=bitrix,
            store=store,
            settings=get_settings(),
            outbound_queue=outbox,
        )
    )

    assert stats.drafts_created == 1
    assert stats.messages_sent == 0
    assert stats.messages_queued == 1
    assert bitrix.messages == []
    assert len(outbox.items) == 1
    assert outbox.items[0]["delivery_key"] == (
        "task_close_direct:8877:closed_at:2026-07-12T12:00:00+03:00:draft"
    )
    assert outbox.items[0]["task"]["task_id"].startswith("task-close-direct:8877:")
    state = store.get_task_close_processing_state(
        task_id=8877,
        state_key=direct_close_state_key("closed_at:2026-07-12T12:00:00+03:00"),
    )
    assert state is not None
    assert state["payload"]["direct_close_draft_queued_at"]
    assert state["payload"]["direct_close_draft_delivery_id"] == "delivery-task-close-1"
    assert not state["payload"]["direct_close_draft_sent_at"]


def test_dispatcher_keeps_direct_close_points_unknown_without_specific_result(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    enqueue_direct_close_event(
        store,
        task_id=8876,
        close_event_key="closed_at:2026-07-12T12:00:00+03:00",
        responsible_id=231,
        dialog_key="231",
        closed_at="2026-07-12T12:00:00+03:00",
        task_title="Проверить прямое закрытие",
        payload={
            "recipient_id": "231",
            "draft_dialog_key": "dialog:231:user:231",
            "task_points": ["Проверить прямое закрытие задачи", "Проверить новый черновик"],
            "source_task_description_empty": False,
        },
    )
    activate_next_direct_close_event(store, responsible_id=231, dialog_key="231")

    stats = anyio.run(lambda: dispatch_direct_task_close_drafts(bitrix=bitrix, store=store, settings=get_settings()))

    assert stats.drafts_created == 1
    message = bitrix.messages[0][1]
    assert "1.1 Проверить прямое закрытие задачи - ... ???" in message
    assert "1.2 Проверить новый черновик - ... ???" in message
    assert "1.3 Еще работы - ... ???" in message
    assert "1.1 Проверить прямое закрытие задачи - не подтверждено" not in message


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
    oauth = RecordingOAuth(RecordingOAuthClient())
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
            bitrix_oauth=oauth,
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
    assert oauth_client.calls == [
        ("tasks.task.get", {"taskId": 8875, "select": ["ID", "STATUS", "CLOSED_DATE"]}),
        ("tasks.task.complete", {"taskId": 8875}),
    ]
    assert [method for method, _ in bitrix.calls] == ["task.item.getfiles", "task.item.addfile"]
    assert "dialog:231:user:231" not in store._drafts


def test_expired_started_draft_finalizes_after_15_minutes_without_waiting_for_20(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    oauth_client = RecordingOAuthClient(task_closed=True)
    store._drafts["dialog:231:user:231"] = {
        "_draft_type": "task_close",
        "_draft_expires_at": "2020-01-01T00:00:00+03:00",
        "task_id": 8875,
        "task_title": "Check cameras",
        "already_closed": True,
        "completion_summary": "Two cameras checked.",
        "overall_status": "unconfirmed",
        "missing_fields": ["archive screenshot"],
    }

    stats = anyio.run(
        lambda: auto_close_direct_task_close_reports(
            bitrix=bitrix,
            bitrix_oauth=RecordingOAuth(oauth_client),
            store=store,
            settings=get_settings(),
            now=datetime(2026, 7, 12, 12, 0, tzinfo=MOSCOW_TZ),
        )
    )

    assert stats.due is False
    assert stats.reports_written == 1
    assert oauth_client.calls == [
        ("tasks.task.get", {"taskId": 8875, "select": ["ID", "STATUS", "CLOSED_DATE"]})
    ]
    assert "dialog:231:user:231" not in store._drafts
    report_text = base64.b64decode(bitrix.calls[-1][1]["fileParameters"]["CONTENT"]).decode("utf-8")
    assert "Two cameras checked." in report_text
    assert "unknown: archive screenshot" in report_text


def test_20_hour_does_not_cut_short_an_active_15_minute_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    bitrix = RecordingBitrix()
    store._drafts["dialog:231:user:231"] = {
        "_draft_type": "task_close",
        "_draft_expires_at": "2099-01-01T00:00:00+03:00",
        "task_id": 8875,
        "task_title": "Check cameras",
        "already_closed": True,
        "overall_status": "unconfirmed",
    }

    stats = anyio.run(
        lambda: auto_close_direct_task_close_reports(
            bitrix=bitrix,
            bitrix_oauth=RecordingOAuth(RecordingOAuthClient()),
            store=store,
            settings=get_settings(),
            now=datetime(2026, 7, 12, 20, 1, tzinfo=MOSCOW_TZ),
        )
    )

    assert stats.due is True
    assert stats.reports_written == 0
    assert bitrix.calls == []
    assert "dialog:231:user:231" in store._drafts


def test_auto_close_open_draft_never_falls_back_to_system_webhook_when_oauth_missing(monkeypatch):
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
    assert stats.system_fallback_closed == 0
    assert stats.admin_notifications == 0
    assert stats.errors and "BitrixOAuthTokenMissing" in stats.errors[0]
    assert bitrix.calls == []
    assert bitrix.messages == []
    assert "dialog:231:user:231" in store._drafts
