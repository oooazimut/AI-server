from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

TASK_CLOSE_DECISION_CONTROLLED = "controlled"
TASK_CLOSE_DECISION_IGNORED_BEFORE_START = "ignored_closed_before_control_start"
TASK_CLOSE_DECISION_IGNORED_USER_NOT_CONTROLLED = "ignored_user_not_controlled_at_close"
TASK_CLOSE_DECISION_ACCEPTED_MISSING = "accepted_missing"


@dataclass(frozen=True)
class TaskCloseControlDecision:
    decision: str
    reason: str


def task_close_event_key(
    *,
    task_id: object,
    closed_at: object | None = None,
    event_id: object | None = None,
    first_seen_at: object | None = None,
) -> str:
    event_text = _clean(event_id)
    if event_text:
        return f"event:{event_text}"

    closed_text = _clean(closed_at)
    if closed_text:
        return f"closed_at:{closed_text}"

    seen_text = _clean(first_seen_at)
    if seen_text:
        return f"first_seen:{seen_text}"

    return f"task:{_clean(task_id) or 'unknown'}:closed_at:unknown"


def decide_task_close_control(
    *,
    closed_at: object | None,
    control_enabled_from: object | None,
    user_is_controlled: bool,
) -> TaskCloseControlDecision:
    closed_dt = _parse_datetime(closed_at)
    control_start_dt = _parse_datetime(control_enabled_from)

    if control_start_dt is not None:
        if closed_dt is None:
            return TaskCloseControlDecision(
                decision=TASK_CLOSE_DECISION_IGNORED_BEFORE_START,
                reason="close_time_not_proven_after_control_start",
            )
        if _is_before(closed_dt, control_start_dt):
            return TaskCloseControlDecision(
                decision=TASK_CLOSE_DECISION_IGNORED_BEFORE_START,
                reason="closed_before_control_start",
            )

    if not user_is_controlled:
        return TaskCloseControlDecision(
            decision=TASK_CLOSE_DECISION_IGNORED_USER_NOT_CONTROLLED,
            reason="user_not_controlled_at_close_time",
        )

    return TaskCloseControlDecision(decision=TASK_CLOSE_DECISION_CONTROLLED, reason="controlled_at_close_time")


def _clean(value: object | None) -> str:
    text = str(value or "").strip()
    return text


def _parse_datetime(value: object | None) -> datetime | None:
    text = _clean(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _is_before(left: datetime, right: datetime) -> bool:
    if (left.tzinfo is None) != (right.tzinfo is None):
        left = left.replace(tzinfo=None)
        right = right.replace(tzinfo=None)
    return left < right
