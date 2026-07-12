from __future__ import annotations

from ai_server.integrations.bitrix.task_close_control import (
    TASK_CLOSE_DECISION_CONTROLLED,
    TASK_CLOSE_DECISION_IGNORED_BEFORE_START,
    TASK_CLOSE_DECISION_IGNORED_USER_NOT_CONTROLLED,
    decide_task_close_control,
    task_close_event_key,
)
from tests.fakes import FakePortalSearchIndex


def test_task_close_control_ignores_closure_before_control_start() -> None:
    decision = decide_task_close_control(
        closed_at="2026-07-10T19:00:00+03:00",
        control_enabled_from="2026-07-12T00:00:00+03:00",
        user_is_controlled=True,
    )

    assert decision.decision == TASK_CLOSE_DECISION_IGNORED_BEFORE_START
    assert decision.reason == "closed_before_control_start"


def test_task_close_control_ignores_user_not_controlled_at_close_time() -> None:
    decision = decide_task_close_control(
        closed_at="2026-07-12T12:00:00+03:00",
        control_enabled_from="2026-07-12T00:00:00+03:00",
        user_is_controlled=False,
    )

    assert decision.decision == TASK_CLOSE_DECISION_IGNORED_USER_NOT_CONTROLLED
    assert decision.reason == "user_not_controlled_at_close_time"


def test_task_close_control_processes_user_controlled_at_close_time() -> None:
    decision = decide_task_close_control(
        closed_at="2026-07-12T12:00:00+03:00",
        control_enabled_from="2026-07-12T00:00:00+03:00",
        user_is_controlled=True,
    )

    assert decision.decision == TASK_CLOSE_DECISION_CONTROLLED
    assert decision.reason == "controlled_at_close_time"


def test_task_close_control_requires_proven_close_time_after_control_start() -> None:
    decision = decide_task_close_control(
        closed_at=None,
        control_enabled_from="2026-07-12T00:00:00+03:00",
        user_is_controlled=True,
    )

    assert decision.decision == TASK_CLOSE_DECISION_IGNORED_BEFORE_START
    assert decision.reason == "close_time_not_proven_after_control_start"


def test_task_close_event_key_prefers_bitrix_event_id() -> None:
    key = task_close_event_key(task_id=8875, event_id="abc-123", closed_at="2026-07-12T12:00:00+03:00")

    assert key == "event:abc-123"


def test_task_close_event_key_uses_close_time_for_reopened_task_events() -> None:
    first = task_close_event_key(task_id=8875, closed_at="2026-07-12T12:00:00+03:00")
    second = task_close_event_key(task_id=8875, closed_at="2026-07-13T12:00:00+03:00")

    assert first == "closed_at:2026-07-12T12:00:00+03:00"
    assert second == "closed_at:2026-07-13T12:00:00+03:00"
    assert first != second


def test_portal_index_stores_control_decision_per_close_event() -> None:
    index = FakePortalSearchIndex()
    old_event = task_close_event_key(task_id=8875, closed_at="2026-07-10T19:00:00+03:00")
    new_event = task_close_event_key(task_id=8875, closed_at="2026-07-13T12:00:00+03:00")

    index.upsert_task_close_control_event(
        task_id=8875,
        close_event_key=old_event,
        decision=TASK_CLOSE_DECISION_IGNORED_USER_NOT_CONTROLLED,
        reason="user_not_controlled_at_close_time",
        closed_at="2026-07-10T19:00:00+03:00",
        responsible_id=231,
    )
    index.upsert_task_close_control_event(
        task_id=8875,
        close_event_key=new_event,
        decision=TASK_CLOSE_DECISION_CONTROLLED,
        reason="controlled_at_close_time",
        closed_at="2026-07-13T12:00:00+03:00",
        responsible_id=231,
    )

    assert index.get_task_close_control_event(task_id=8875, close_event_key=old_event)["decision"] == (
        TASK_CLOSE_DECISION_IGNORED_USER_NOT_CONTROLLED
    )
    assert index.get_task_close_control_event(task_id=8875, close_event_key=new_event)["decision"] == (
        TASK_CLOSE_DECISION_CONTROLLED
    )
