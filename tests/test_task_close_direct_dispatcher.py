from __future__ import annotations

from datetime import datetime

import anyio

from ai_server.integrations.bitrix.task_close_direct_queue import (
    activate_next_direct_close_event,
    direct_close_state_key,
    enqueue_direct_close_event,
)
from ai_server.models import AgentResult, AgentTask
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
        self.draft_rows: list[dict] = []

    def list_task_drafts(self, *, draft_type: str = "", limit: int = 100, expired_only: bool = False):
        return [dict(row) for row in self.draft_rows[:limit]]


class RecordingOrchestrator:
    def __init__(self, status: str = "needs_human") -> None:
        self.status = status
        self.tasks: list[AgentTask] = []

    async def handle(self, task: AgentTask) -> AgentResult:
        self.tasks.append(task)
        return AgentResult(status=self.status, agent_id="internal_orchestrator", answer="ok")


def _active_event(store: DraftQueueStore, *, task_id: int = 8875, responsible_id: int = 231) -> None:
    enqueue_direct_close_event(
        store,
        task_id=task_id,
        close_event_key="event-a",
        responsible_id=responsible_id,
        dialog_key=str(responsible_id),
        closed_at="2026-07-12T12:00:00+03:00",
        task_title="Проверить камеры",
        payload={
            "recipient_id": str(responsible_id),
            "draft_dialog_key": f"dialog:{responsible_id}:user:{responsible_id}",
            "task_results": ["Камеры проверены"],
            "task_points": ["Проверить камеры", "Проверить архив"],
        },
    )
    activate_next_direct_close_event(store, responsible_id=responsible_id, dialog_key=str(responsible_id))


def test_direct_close_worker_delegates_exact_facts_to_orchestrator(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    orchestrator = RecordingOrchestrator()
    _active_event(store)

    stats = anyio.run(
        lambda: dispatch_direct_task_close_drafts(
            store=store,
            settings=get_settings(),
            orchestrator_handler=orchestrator.handle,
        )
    )

    assert stats.drafts_created == 1
    assert stats.messages_sent == 1
    assert len(orchestrator.tasks) == 1
    task = orchestrator.tasks[0]
    assert task.source == "task_close_direct_control"
    assert task.user.id == "231"
    assert task.context["orchestrator_required_tool"] == "task_close_draft"
    assert task.context["task_close_event"] == {
        "task_id": 8875,
        "task_title": "Проверить камеры",
        "close_event_key": "event-a",
        "closed_at": "2026-07-12T12:00:00+03:00",
        "task_results": ["Камеры проверены"],
        "task_points": ["Проверить камеры", "Проверить архив"],
        "source_task_description_empty": False,
    }
    state = store.get_task_close_processing_state(task_id=8875, state_key=direct_close_state_key("event-a"))
    assert state["payload"]["direct_close_draft_orchestrated_at"]


def test_direct_close_worker_fails_closed_without_orchestrator(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    _active_event(store)

    stats = anyio.run(
        lambda: dispatch_direct_task_close_drafts(
            store=store,
            settings=get_settings(),
            orchestrator_handler=None,
        )
    )

    assert stats.blocked == 1
    state = store.get_task_close_processing_state(task_id=8875, state_key=direct_close_state_key("event-a"))
    assert state["payload"]["dispatch_blocked_reason"] == "orchestrator_unavailable"


def test_direct_close_worker_does_not_dispatch_same_event_twice(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    orchestrator = RecordingOrchestrator()
    _active_event(store)

    async def run_twice():
        first = await dispatch_direct_task_close_drafts(
            store=store, settings=get_settings(), orchestrator_handler=orchestrator.handle
        )
        second = await dispatch_direct_task_close_drafts(
            store=store, settings=get_settings(), orchestrator_handler=orchestrator.handle
        )
        return first, second

    first, second = anyio.run(run_twice)
    assert first.drafts_created == 1
    assert second.skipped == 1
    assert len(orchestrator.tasks) == 1


def test_pending_event_is_not_dispatched(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    orchestrator = RecordingOrchestrator()
    enqueue_direct_close_event(
        store,
        task_id=8875,
        close_event_key="event-a",
        responsible_id=231,
        dialog_key="231",
        payload={"recipient_id": "231"},
    )

    stats = anyio.run(
        lambda: dispatch_direct_task_close_drafts(
            store=store, settings=get_settings(), orchestrator_handler=orchestrator.handle
        )
    )
    assert stats.candidates == 0
    assert orchestrator.tasks == []


def test_expired_draft_auto_finalize_is_an_orchestrator_command(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    store.draft_rows = [
        {
            "dialog_key": "dialog:231:user:231:conversation:20260712:101",
            "params": {
                "_draft_type": "task_close",
                "_draft_id": "draft-1",
                "_draft_user_id": 231,
                "task_id": 8875,
            },
        }
    ]
    orchestrator = RecordingOrchestrator(status="completed")

    stats = anyio.run(
        lambda: auto_close_direct_task_close_reports(
            store=store,
            settings=get_settings(),
            orchestrator_handler=orchestrator.handle,
            now=datetime(2026, 7, 12, 20, 1, tzinfo=MOSCOW_TZ),
        )
    )

    assert stats.closed == 1
    task = orchestrator.tasks[0]
    assert task.context["dialog_key"].endswith(":101")
    assert task.context["orchestrator_required_tool"] == "task_close_confirm"
    assert task.context["task_close_confirmation_mode"] == "auto_unconfirmed"


def test_control_once_records_both_orchestrator_dispatches(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = DraftQueueStore()
    _active_event(store)
    orchestrator = RecordingOrchestrator()
    status: dict = {}

    result = anyio.run(
        lambda: run_task_close_direct_control_once(
            store=store,
            settings=get_settings(),
            orchestrator_handler=orchestrator.handle,
            status=status,
            now=datetime(2026, 7, 12, 19, 0, tzinfo=MOSCOW_TZ),
        )
    )

    assert result["dispatch"]["drafts_created"] == 1
    assert result["auto_close"]["due"] is False
    assert status["checked_at"] == "2026-07-12T19:00:00+03:00"
