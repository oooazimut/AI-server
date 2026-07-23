from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

from ai_server.integrations.bitrix.task_close_direct_queue import (
    TASK_CLOSE_DIRECT_QUEUE_PREFIX,
    TASK_CLOSE_DIRECT_STATUS_ACTIVE,
)
from ai_server.models import AgentResult, AgentTask, UserContext
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)

OrchestratorHandler = Callable[[AgentTask], Awaitable[AgentResult]]


@dataclass
class DirectTaskCloseDraftDispatchStats:
    candidates: int = 0
    drafts_created: int = 0
    messages_sent: int = 0
    messages_queued: int = 0
    skipped: int = 0
    blocked: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "drafts_created": self.drafts_created,
            "messages_sent": self.messages_sent,
            "messages_queued": self.messages_queued,
            "skipped": self.skipped,
            "blocked": self.blocked,
            "errors": list(self.errors),
        }


@dataclass
class DirectTaskCloseAutoCloseStats:
    due: bool = False
    auto_close_time: str = "20:00"
    candidates: int = 0
    reports_written: int = 0
    closed: int = 0
    open_drafts: int = 0
    oauth_closed: int = 0
    system_fallback_closed: int = 0
    admin_notifications: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "due": self.due,
            "auto_close_time": self.auto_close_time,
            "candidates": self.candidates,
            "reports_written": self.reports_written,
            "closed": self.closed,
            "open_drafts": self.open_drafts,
            "oauth_closed": self.oauth_closed,
            "system_fallback_closed": self.system_fallback_closed,
            "admin_notifications": self.admin_notifications,
            "skipped": self.skipped,
            "errors": list(self.errors),
        }


async def run_task_close_direct_control_worker(
    *,
    store: Any,
    settings: Settings,
    status: dict[str, Any],
    orchestrator_handler: OrchestratorHandler | None,
    **_: Any,
) -> None:
    """Dispatch task-close facts through the sole semantic owner: the orchestrator."""

    interval_seconds = max(int(settings.bitrix_task_close_control_interval_seconds), 30)
    status.update(
        {
            "enabled": settings.bitrix_task_close_control_worker_enabled,
            "running": True,
            "interval_seconds": interval_seconds,
            "direct_limit": settings.bitrix_task_close_control_direct_limit,
            "auto_close_limit": settings.bitrix_task_close_control_auto_close_limit,
            "last_check_at": None,
            "last_success_at": None,
            "last_error": None,
            "next_check_at": None,
            "runs": int(status.get("runs") or 0),
            "errors": int(status.get("errors") or 0),
        }
    )
    while True:
        try:
            result = await run_task_close_direct_control_once(
                store=store,
                settings=settings,
                status=status,
                orchestrator_handler=orchestrator_handler,
            )
            status["last_success_at"] = _now().isoformat()
            status["last_error"] = None
            status["runs"] = int(status.get("runs") or 0) + 1
            status["last_result"] = result
            await _sleep_until_next(status, interval_seconds)
        except asyncio.CancelledError:
            status["running"] = False
            raise
        except Exception as exc:
            logger.exception("Task-close orchestrator dispatcher tick failed")
            status["last_error"] = f"{type(exc).__name__}: {exc}"
            status["errors"] = int(status.get("errors") or 0) + 1
            await _sleep_until_next(status, min(interval_seconds, 300))


async def run_task_close_direct_control_once(
    *,
    store: Any,
    settings: Settings,
    orchestrator_handler: OrchestratorHandler | None,
    status: dict[str, Any] | None = None,
    now: datetime | None = None,
    **_: Any,
) -> dict[str, Any]:
    now_dt = now or _now()
    if status is not None:
        status["last_check_at"] = now_dt.isoformat()
    dispatch = await dispatch_direct_task_close_drafts(
        store=store,
        settings=settings,
        orchestrator_handler=orchestrator_handler,
        limit=max(int(settings.bitrix_task_close_control_direct_limit), 1),
        now=now_dt,
    )
    auto_close = await auto_close_direct_task_close_reports(
        store=store,
        settings=settings,
        orchestrator_handler=orchestrator_handler,
        limit=max(int(settings.bitrix_task_close_control_auto_close_limit), 1),
        now=now_dt,
    )
    result = {"checked_at": now_dt.isoformat(), "dispatch": dispatch.as_dict(), "auto_close": auto_close.as_dict()}
    if status is not None:
        status.update(result)
    return result


async def dispatch_direct_task_close_drafts(
    *,
    store: Any,
    settings: Settings,
    orchestrator_handler: OrchestratorHandler | None,
    limit: int = 20,
    now: datetime | None = None,
    **_: Any,
) -> DirectTaskCloseDraftDispatchStats:
    stats = DirectTaskCloseDraftDispatchStats()
    states = _list_states(store, statuses=[TASK_CLOSE_DIRECT_STATUS_ACTIVE], limit=limit)
    stats.candidates = len(states)
    now_iso = (now or _now()).isoformat(timespec="seconds")
    for state in states:
        payload = dict(state.get("payload") or {})
        if payload.get("direct_close_draft_orchestrated_at"):
            stats.skipped += 1
            continue
        task = _draft_event_task(state, payload)
        if task is None or orchestrator_handler is None:
            _record_dispatch_failure(store, state, payload, now_iso, "orchestrator_unavailable")
            stats.blocked += 1
            continue
        try:
            result = await orchestrator_handler(task)
        except Exception as exc:
            stats.errors.append(f"{state.get('task_id')}:{type(exc).__name__}: {exc}")
            continue
        if result.status != "needs_human":
            _record_dispatch_failure(store, state, payload, now_iso, f"orchestrator_status:{result.status}")
            stats.blocked += 1
            continue
        _update_state(
            store,
            state,
            {
                **payload,
                "direct_close_draft_orchestrated_at": now_iso,
                "direct_close_orchestrator_task_id": task.task_id,
            },
        )
        stats.drafts_created += 1
        stats.messages_sent += 1
    return stats


async def auto_close_direct_task_close_reports(
    *,
    store: Any,
    settings: Settings,
    orchestrator_handler: OrchestratorHandler | None,
    limit: int = 100,
    now: datetime | None = None,
    **_: Any,
) -> DirectTaskCloseAutoCloseStats:
    now_dt = now or _now()
    auto_close_time = _configured_auto_close_time(store)
    stats = DirectTaskCloseAutoCloseStats(
        due=_is_auto_close_due(now_dt, auto_close_time),
        auto_close_time=auto_close_time,
    )
    # A started numbered draft expires after its own TTL and is finalized on
    # the next control tick.  The configured control time is only telemetry
    # for queue-level policy; it must not prolong or shorten the branch TTL.
    rows = await _list_expired_task_close_drafts(store, limit=limit)
    stats.open_drafts = len(rows)
    stats.candidates = len(rows)
    if settings.agent_dry_run:
        stats.skipped = len(rows)
        return stats
    for row in rows:
        task = _auto_finalize_task(row)
        if task is None or orchestrator_handler is None:
            stats.skipped += 1
            continue
        try:
            result = await orchestrator_handler(task)
        except Exception as exc:
            stats.errors.append(f"draft:{row.get('dialog_key')}:{type(exc).__name__}: {exc}")
            continue
        if result.status != "completed":
            stats.errors.append(f"draft:{row.get('dialog_key')}:orchestrator_status:{result.status}")
            continue
        stats.reports_written += 1
        stats.closed += 1
        stats.oauth_closed += 1
    return stats


def _draft_event_task(state: dict[str, Any], payload: dict[str, Any]) -> AgentTask | None:
    task_id = _safe_int(state.get("task_id") or payload.get("task_id"))
    responsible_id = _safe_int(payload.get("responsible_id"))
    recipient_id = str(payload.get("recipient_id") or responsible_id or "").strip()
    close_event_key = str(payload.get("close_event_key") or "").strip()
    if task_id is None or responsible_id is None or not recipient_id or not close_event_key:
        return None
    title = str(payload.get("task_title") or f"#{task_id}").strip()
    return AgentTask(
        task_id=f"task-close-direct:{task_id}:{close_event_key}",
        source="task_close_direct_control",
        user=UserContext(id=str(responsible_id), channel="bitrix24"),
        request=(
            f"Задача #{task_id} «{title}» уже закрыта напрямую в Bitrix. "
            "Подготовь для ответственного четырёхблочный черновик проверки результата; "
            "не закрывай задачу повторно и не придумывай отсутствующие сведения."
        ),
        context={
            "channel_id": "bitrix24",
            "recipient_id": recipient_id,
            "dialog_id": recipient_id,
            "base_dialog_key": str(payload.get("draft_dialog_key") or f"dialog:{responsible_id}:user:{responsible_id}"),
            "event": "task_close_direct",
            "orchestrator_internal_event": True,
            "orchestrator_required_tool": "task_close_draft",
            "task_close_event": {
                "task_id": task_id,
                "task_title": title,
                "close_event_key": close_event_key,
                "closed_at": payload.get("closed_at"),
                "task_results": list(payload.get("task_results") or []),
                "task_points": list(payload.get("task_points") or []),
                "source_task_description_empty": bool(payload.get("source_task_description_empty")),
            },
        },
    )


def _auto_finalize_task(row: dict[str, Any]) -> AgentTask | None:
    dialog_key = str(row.get("dialog_key") or "").strip()
    draft = row.get("params")
    if not dialog_key or not isinstance(draft, dict):
        return None
    task_id = _safe_int(draft.get("task_id"))
    user_id = _safe_int(draft.get("_draft_user_id")) or _user_id_from_dialog_key(dialog_key)
    if task_id is None or user_id is None:
        return None
    return AgentTask(
        task_id=f"task-close-auto-finalize:{task_id}:{draft.get('_draft_id') or dialog_key}",
        source="task_close_direct_control",
        user=UserContext(id=str(user_id), channel="bitrix24"),
        request=(
            f"Контрольное время черновика закрытия задачи #{task_id} истекло. "
            "Заверши ровно этот черновик как неподтверждённый, сохрани неизвестные сведения неизвестными."
        ),
        context={
            "dialog_key": dialog_key,
            "dialog_id": str(user_id),
            "event": "task_close_auto_finalize",
            "orchestrator_internal_event": True,
            "orchestrator_required_tool": "task_close_confirm",
            "task_close_confirmation_mode": "auto_unconfirmed",
        },
    )


def _list_states(store: Any, *, statuses: list[str], limit: int) -> list[dict[str, Any]]:
    lister = getattr(store, "list_task_close_processing_states", None)
    if not callable(lister):
        return []
    return [
        dict(item)
        for item in lister(statuses=statuses, state_key_prefix=TASK_CLOSE_DIRECT_QUEUE_PREFIX, limit=limit)
        if isinstance(item, dict)
    ]


async def _list_expired_task_close_drafts(store: Any, *, limit: int) -> list[dict[str, Any]]:
    lister = getattr(store, "list_task_drafts", None)
    if not callable(lister):
        return []
    rows = lister(draft_type="task_close", limit=limit, expired_only=True)
    if inspect.isawaitable(rows):
        rows = await rows
    return [dict(row) for row in rows if isinstance(row, dict)]


def _record_dispatch_failure(
    store: Any, state: dict[str, Any], payload: dict[str, Any], now_iso: str, reason: str
) -> None:
    _update_state(
        store,
        state,
        {**payload, "dispatch_blocked_at": now_iso, "dispatch_blocked_reason": reason},
    )


def _update_state(store: Any, state: dict[str, Any], payload: dict[str, Any]) -> None:
    upsert = getattr(store, "upsert_task_close_processing_state", None)
    if callable(upsert):
        upsert(
            task_id=state.get("task_id"),
            state_key=str(state.get("state_key") or ""),
            status=str(state.get("status") or TASK_CLOSE_DIRECT_STATUS_ACTIVE),
            payload=payload,
            actor_user_id=_safe_int(state.get("actor_user_id")),
        )


def _configured_auto_close_time(store: Any) -> str:
    getter = getattr(store, "get_task_close_control_setting", None)
    setting = getter("task_close_auto_close_time") if callable(getter) else None
    value = str((setting or {}).get("value") or "20:00").strip()
    return value if _parse_hhmm(value) is not None else "20:00"


def _is_auto_close_due(now: datetime, value: str) -> bool:
    configured = _parse_hhmm(value) or time(hour=20)
    return now.timetz().replace(tzinfo=None) >= configured


def _parse_hhmm(value: str) -> time | None:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour, minute = int(hour_text), int(minute_text)
    except (TypeError, ValueError):
        return None
    return time(hour=hour, minute=minute) if 0 <= hour <= 23 and 0 <= minute <= 59 else None


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _user_id_from_dialog_key(value: object) -> int | None:
    for part in reversed(str(value or "").split(":")):
        result = _safe_int(part)
        if result is not None and result > 0:
            return result
    return None


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)


async def _sleep_until_next(status: dict[str, Any], seconds: int) -> None:
    status["next_check_at"] = (_now() + timedelta(seconds=seconds)).isoformat()
    await asyncio.sleep(seconds)
