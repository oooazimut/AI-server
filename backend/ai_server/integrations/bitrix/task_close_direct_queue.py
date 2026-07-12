from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

TASK_CLOSE_DIRECT_QUEUE_PREFIX = "direct_close:"

TASK_CLOSE_DIRECT_STATUS_PENDING = "pending_direct_close"
TASK_CLOSE_DIRECT_STATUS_ACTIVE = "active_direct_close_draft"
TASK_CLOSE_DIRECT_STATUS_AUTO_CLOSED_UNCONFIRMED = "auto_closed_unconfirmed"
TASK_CLOSE_DIRECT_STATUS_COMPLETED = "completed"
TASK_CLOSE_DIRECT_STATUS_DISCARDED = "discarded"

TASK_CLOSE_DIRECT_TERMINAL_STATUSES = frozenset(
    {
        TASK_CLOSE_DIRECT_STATUS_AUTO_CLOSED_UNCONFIRMED,
        TASK_CLOSE_DIRECT_STATUS_COMPLETED,
        TASK_CLOSE_DIRECT_STATUS_DISCARDED,
    }
)

TASK_CLOSE_DIRECT_OPEN_STATUSES = (
    TASK_CLOSE_DIRECT_STATUS_ACTIVE,
    TASK_CLOSE_DIRECT_STATUS_PENDING,
)


@dataclass(frozen=True)
class TaskCloseDirectQueueEvent:
    task_id: int
    state_key: str
    status: str
    payload: dict[str, Any]
    actor_user_id: int | None = None


def direct_close_state_key(close_event_key: object) -> str:
    return f"{TASK_CLOSE_DIRECT_QUEUE_PREFIX}{_clean(close_event_key) or 'unknown'}"


def enqueue_direct_close_event(
    store: Any,
    *,
    task_id: object,
    close_event_key: object,
    responsible_id: int | None = None,
    dialog_key: str = "",
    closed_at: object | None = None,
    task_title: str = "",
    payload: dict[str, Any] | None = None,
    now_iso: str | None = None,
    actor_user_id: int | None = None,
) -> TaskCloseDirectQueueEvent | None:
    task_id_int = safe_int(task_id)
    if task_id_int is None:
        return None
    state_key = direct_close_state_key(close_event_key)
    existing = _get_state(store, task_id=task_id_int, state_key=state_key)
    if existing and str(existing.get("status") or "") != TASK_CLOSE_DIRECT_STATUS_PENDING:
        return _event_from_state(existing)

    base_payload = dict(payload or {})
    base_payload.update(
        {
            "close_event_key": _clean(close_event_key),
            "responsible_id": responsible_id,
            "dialog_key": dialog_key,
            "closed_at": _clean(closed_at),
            "task_title": task_title,
            "queued_at": (existing.get("payload") or {}).get("queued_at") if existing else now_iso,
        }
    )
    if not base_payload.get("queued_at"):
        base_payload["queued_at"] = now_iso

    _upsert_state(
        store,
        task_id=task_id_int,
        state_key=state_key,
        status=TASK_CLOSE_DIRECT_STATUS_PENDING,
        payload=base_payload,
        actor_user_id=actor_user_id,
    )
    return TaskCloseDirectQueueEvent(
        task_id=task_id_int,
        state_key=state_key,
        status=TASK_CLOSE_DIRECT_STATUS_PENDING,
        payload=base_payload,
        actor_user_id=actor_user_id,
    )


def activate_next_direct_close_event(
    store: Any,
    *,
    responsible_id: int | None = None,
    dialog_key: str = "",
    now_iso: str | None = None,
) -> TaskCloseDirectQueueEvent | None:
    active = _oldest_event(
        _list_states(
            store,
            statuses=[TASK_CLOSE_DIRECT_STATUS_ACTIVE],
            responsible_id=responsible_id,
            dialog_key=dialog_key,
        )
    )
    if active is not None:
        return active

    pending = _oldest_event(
        _list_states(
            store,
            statuses=[TASK_CLOSE_DIRECT_STATUS_PENDING],
            responsible_id=responsible_id,
            dialog_key=dialog_key,
        )
    )
    if pending is None:
        return None

    payload = dict(pending.payload)
    payload["activated_at"] = now_iso
    _upsert_state(
        store,
        task_id=pending.task_id,
        state_key=pending.state_key,
        status=TASK_CLOSE_DIRECT_STATUS_ACTIVE,
        payload=payload,
        actor_user_id=pending.actor_user_id,
    )
    return TaskCloseDirectQueueEvent(
        task_id=pending.task_id,
        state_key=pending.state_key,
        status=TASK_CLOSE_DIRECT_STATUS_ACTIVE,
        payload=payload,
        actor_user_id=pending.actor_user_id,
    )


def auto_close_direct_close_queue_as_unconfirmed(
    store: Any,
    *,
    responsible_id: int | None = None,
    dialog_key: str = "",
    now_iso: str | None = None,
    reason: str = "control_time_reached",
) -> list[TaskCloseDirectQueueEvent]:
    open_events = _sorted_events(
        _list_states(
            store,
            statuses=list(TASK_CLOSE_DIRECT_OPEN_STATUSES),
            responsible_id=responsible_id,
            dialog_key=dialog_key,
        )
    )
    closed: list[TaskCloseDirectQueueEvent] = []
    for event in open_events:
        payload = dict(event.payload)
        payload["auto_closed_at"] = now_iso
        payload["auto_close_reason"] = reason
        payload["problem_types"] = ["unconfirmed"]
        payload.setdefault(
            "unconfirmed_items",
            ["Result was not confirmed before the control time."],
        )
        _upsert_state(
            store,
            task_id=event.task_id,
            state_key=event.state_key,
            status=TASK_CLOSE_DIRECT_STATUS_AUTO_CLOSED_UNCONFIRMED,
            payload=payload,
            actor_user_id=event.actor_user_id,
        )
        closed.append(
            TaskCloseDirectQueueEvent(
                task_id=event.task_id,
                state_key=event.state_key,
                status=TASK_CLOSE_DIRECT_STATUS_AUTO_CLOSED_UNCONFIRMED,
                payload=payload,
                actor_user_id=event.actor_user_id,
            )
        )
    return closed


def complete_direct_close_event(
    store: Any,
    *,
    task_id: object,
    close_event_key: object,
    now_iso: str | None = None,
    actor_user_id: int | None = None,
) -> TaskCloseDirectQueueEvent | None:
    task_id_int = safe_int(task_id)
    if task_id_int is None:
        return None
    state_key = direct_close_state_key(close_event_key)
    state = _get_state(store, task_id=task_id_int, state_key=state_key)
    if not state:
        return None
    event = _event_from_state(state)
    payload = dict(event.payload)
    payload["completed_at"] = now_iso
    _upsert_state(
        store,
        task_id=task_id_int,
        state_key=state_key,
        status=TASK_CLOSE_DIRECT_STATUS_COMPLETED,
        payload=payload,
        actor_user_id=actor_user_id if actor_user_id is not None else event.actor_user_id,
    )
    return TaskCloseDirectQueueEvent(
        task_id=task_id_int,
        state_key=state_key,
        status=TASK_CLOSE_DIRECT_STATUS_COMPLETED,
        payload=payload,
        actor_user_id=actor_user_id if actor_user_id is not None else event.actor_user_id,
    )


def discard_direct_close_event(
    store: Any,
    *,
    task_id: object,
    close_event_key: object,
    now_iso: str | None = None,
    actor_user_id: int | None = None,
) -> TaskCloseDirectQueueEvent | None:
    task_id_int = safe_int(task_id)
    if task_id_int is None:
        return None
    state_key = direct_close_state_key(close_event_key)
    state = _get_state(store, task_id=task_id_int, state_key=state_key)
    if not state:
        return None
    event = _event_from_state(state)
    payload = dict(event.payload)
    payload["discarded_at"] = now_iso
    _upsert_state(
        store,
        task_id=task_id_int,
        state_key=state_key,
        status=TASK_CLOSE_DIRECT_STATUS_DISCARDED,
        payload=payload,
        actor_user_id=actor_user_id if actor_user_id is not None else event.actor_user_id,
    )
    return TaskCloseDirectQueueEvent(
        task_id=task_id_int,
        state_key=state_key,
        status=TASK_CLOSE_DIRECT_STATUS_DISCARDED,
        payload=payload,
        actor_user_id=actor_user_id if actor_user_id is not None else event.actor_user_id,
    )


def _get_state(store: Any, *, task_id: int, state_key: str) -> dict[str, Any] | None:
    getter = getattr(store, "get_task_close_processing_state", None)
    if not callable(getter):
        return None
    state = getter(task_id=task_id, state_key=state_key)
    return dict(state) if state else None


def _upsert_state(
    store: Any,
    *,
    task_id: int,
    state_key: str,
    status: str,
    payload: dict[str, Any],
    actor_user_id: int | None = None,
) -> None:
    upsert = getattr(store, "upsert_task_close_processing_state", None)
    if not callable(upsert):
        return
    upsert(
        task_id=task_id,
        state_key=state_key,
        status=status,
        payload=payload,
        actor_user_id=actor_user_id,
    )


def _list_states(
    store: Any,
    *,
    statuses: list[str],
    responsible_id: int | None,
    dialog_key: str,
) -> list[TaskCloseDirectQueueEvent]:
    lister = getattr(store, "list_task_close_processing_states", None)
    if not callable(lister):
        return []
    states = lister(
        statuses=statuses,
        state_key_prefix=TASK_CLOSE_DIRECT_QUEUE_PREFIX,
        responsible_id=responsible_id,
        dialog_key=dialog_key,
        limit=500,
    )
    return [_event_from_state(state) for state in states]


def _event_from_state(state: dict[str, Any]) -> TaskCloseDirectQueueEvent:
    return TaskCloseDirectQueueEvent(
        task_id=safe_int(state.get("task_id")) or 0,
        state_key=str(state.get("state_key") or ""),
        status=str(state.get("status") or ""),
        payload=dict(state.get("payload") or {}),
        actor_user_id=safe_int(state.get("actor_user_id")),
    )


def _oldest_event(events: list[TaskCloseDirectQueueEvent]) -> TaskCloseDirectQueueEvent | None:
    ordered = _sorted_events(events)
    return ordered[0] if ordered else None


def _sorted_events(events: list[TaskCloseDirectQueueEvent]) -> list[TaskCloseDirectQueueEvent]:
    return sorted(events, key=_event_sort_key)


def _event_sort_key(event: TaskCloseDirectQueueEvent) -> tuple[tuple[int, str], tuple[int, str], int, str]:
    return (
        _datetime_sort_key(event.payload.get("closed_at")),
        _datetime_sort_key(event.payload.get("queued_at")),
        event.task_id,
        event.state_key,
    )


def _datetime_sort_key(value: object | None) -> tuple[int, str]:
    parsed = _parse_datetime(value)
    if parsed is None:
        return (1, _clean(value))
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return (0, parsed.isoformat())


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


def _clean(value: object | None) -> str:
    return str(value or "").strip()


def safe_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
