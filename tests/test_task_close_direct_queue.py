from __future__ import annotations

from ai_server.integrations.bitrix.task_close_direct_queue import (
    TASK_CLOSE_DIRECT_STATUS_ACTIVE,
    TASK_CLOSE_DIRECT_STATUS_AUTO_CLOSED_UNCONFIRMED,
    TASK_CLOSE_DIRECT_STATUS_COMPLETED,
    TASK_CLOSE_DIRECT_STATUS_DISCARDED,
    TASK_CLOSE_DIRECT_STATUS_PENDING,
    activate_next_direct_close_event,
    auto_close_direct_close_queue_as_unconfirmed,
    complete_direct_close_event,
    direct_close_state_key,
    discard_direct_close_event,
    enqueue_direct_close_event,
)
from tests.fakes import FakePortalSearchIndex


def test_direct_close_queue_activates_oldest_event_and_keeps_one_active() -> None:
    index = FakePortalSearchIndex()
    enqueue_direct_close_event(
        index,
        task_id=901,
        close_event_key="event-late",
        responsible_id=231,
        dialog_key="chat231",
        closed_at="2026-07-12T12:20:00+03:00",
        now_iso="2026-07-12T12:21:00+03:00",
    )
    enqueue_direct_close_event(
        index,
        task_id=900,
        close_event_key="event-early",
        responsible_id=231,
        dialog_key="chat231",
        closed_at="2026-07-12T12:10:00+03:00",
        now_iso="2026-07-12T12:22:00+03:00",
    )

    active = activate_next_direct_close_event(
        index,
        responsible_id=231,
        dialog_key="chat231",
        now_iso="2026-07-12T12:23:00+03:00",
    )
    assert active is not None
    assert active.task_id == 900
    assert active.status == TASK_CLOSE_DIRECT_STATUS_ACTIVE

    repeated = activate_next_direct_close_event(
        index,
        responsible_id=231,
        dialog_key="chat231",
        now_iso="2026-07-12T12:24:00+03:00",
    )
    assert repeated is not None
    assert repeated.task_id == 900

    later = index.get_task_close_processing_state(task_id=901, state_key=direct_close_state_key("event-late"))
    assert later is not None
    assert later["status"] == TASK_CLOSE_DIRECT_STATUS_PENDING


def test_direct_close_queue_does_not_downgrade_existing_active_event() -> None:
    index = FakePortalSearchIndex()
    enqueue_direct_close_event(
        index,
        task_id=900,
        close_event_key="event-early",
        responsible_id=231,
        dialog_key="chat231",
        closed_at="2026-07-12T12:10:00+03:00",
        now_iso="2026-07-12T12:11:00+03:00",
    )
    activate_next_direct_close_event(index, responsible_id=231, dialog_key="chat231")

    duplicate = enqueue_direct_close_event(
        index,
        task_id=900,
        close_event_key="event-early",
        responsible_id=231,
        dialog_key="chat231",
        closed_at="2026-07-12T12:10:00+03:00",
        now_iso="2026-07-12T12:30:00+03:00",
    )

    assert duplicate is not None
    assert duplicate.status == TASK_CLOSE_DIRECT_STATUS_ACTIVE
    state = index.get_task_close_processing_state(task_id=900, state_key=direct_close_state_key("event-early"))
    assert state is not None
    assert state["status"] == TASK_CLOSE_DIRECT_STATUS_ACTIVE


def test_direct_close_queue_auto_closes_active_and_pending_for_same_dialog() -> None:
    index = FakePortalSearchIndex()
    enqueue_direct_close_event(
        index,
        task_id=900,
        close_event_key="event-early",
        responsible_id=231,
        dialog_key="chat231",
        closed_at="2026-07-12T12:10:00+03:00",
    )
    enqueue_direct_close_event(
        index,
        task_id=901,
        close_event_key="event-late",
        responsible_id=231,
        dialog_key="chat231",
        closed_at="2026-07-12T12:20:00+03:00",
    )
    activate_next_direct_close_event(index, responsible_id=231, dialog_key="chat231")

    closed = auto_close_direct_close_queue_as_unconfirmed(
        index,
        responsible_id=231,
        dialog_key="chat231",
        now_iso="2026-07-12T20:00:00+03:00",
    )

    assert [event.task_id for event in closed] == [900, 901]
    assert {event.status for event in closed} == {TASK_CLOSE_DIRECT_STATUS_AUTO_CLOSED_UNCONFIRMED}
    assert all(event.payload["problem_types"] == ["unconfirmed"] for event in closed)
    assert all(event.payload["auto_close_reason"] == "control_time_reached" for event in closed)


def test_direct_close_queue_auto_close_excludes_other_dialogs() -> None:
    index = FakePortalSearchIndex()
    enqueue_direct_close_event(index, task_id=900, close_event_key="event-a", responsible_id=231, dialog_key="chat231")
    enqueue_direct_close_event(index, task_id=901, close_event_key="event-b", responsible_id=231, dialog_key="chat999")

    closed = auto_close_direct_close_queue_as_unconfirmed(index, responsible_id=231, dialog_key="chat231")

    assert [event.task_id for event in closed] == [900]
    other = index.get_task_close_processing_state(task_id=901, state_key=direct_close_state_key("event-b"))
    assert other is not None
    assert other["status"] == TASK_CLOSE_DIRECT_STATUS_PENDING


def test_direct_close_queue_completed_event_is_terminal() -> None:
    index = FakePortalSearchIndex()
    enqueue_direct_close_event(index, task_id=900, close_event_key="event-a", responsible_id=231, dialog_key="chat231")

    completed = complete_direct_close_event(
        index,
        task_id=900,
        close_event_key="event-a",
        now_iso="2026-07-12T13:00:00+03:00",
    )

    assert completed is not None
    assert completed.status == TASK_CLOSE_DIRECT_STATUS_COMPLETED
    duplicate = enqueue_direct_close_event(index, task_id=900, close_event_key="event-a", responsible_id=231)
    assert duplicate is not None
    assert duplicate.status == TASK_CLOSE_DIRECT_STATUS_COMPLETED


def test_direct_close_queue_discarded_event_is_terminal() -> None:
    index = FakePortalSearchIndex()
    enqueue_direct_close_event(index, task_id=900, close_event_key="event-a", responsible_id=231, dialog_key="chat231")

    discarded = discard_direct_close_event(
        index,
        task_id=900,
        close_event_key="event-a",
        now_iso="2026-07-12T13:00:00+03:00",
    )

    assert discarded is not None
    assert discarded.status == TASK_CLOSE_DIRECT_STATUS_DISCARDED
    duplicate = enqueue_direct_close_event(index, task_id=900, close_event_key="event-a", responsible_id=231)
    assert duplicate is not None
    assert duplicate.status == TASK_CLOSE_DIRECT_STATUS_DISCARDED
