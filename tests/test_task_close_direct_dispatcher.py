from __future__ import annotations

import anyio

from ai_server.integrations.bitrix.task_close_direct_queue import (
    TASK_CLOSE_DIRECT_STATUS_ACTIVE,
    TASK_CLOSE_DIRECT_STATUS_PENDING,
    activate_next_direct_close_event,
    direct_close_state_key,
    enqueue_direct_close_event,
)
from ai_server.settings import get_settings
from ai_server.workers.bitrix.task_close_direct_dispatcher import dispatch_direct_task_close_drafts
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


class RecordingBitrix:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, int | None]] = []

    async def send_bot_message(self, dialog_id: str, message: str, *, bot_id=None, keyboard=None):
        self.messages.append((dialog_id, message, bot_id))
        return {"message_id": len(self.messages)}


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
