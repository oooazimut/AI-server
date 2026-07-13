from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any

from ai_server.agents.bitrix24.tools.task_close import (
    TASK_CLOSE_DRAFT_TYPE,
    _execute_task_close,
    build_task_close_draft_from_args,
    format_task_close_draft_message,
)
from ai_server.agents.bitrix24.tools.task_close_control import (
    TASK_CLOSE_AUTO_CLOSE_TIME_KEY,
    TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME,
)
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthError
from ai_server.integrations.bitrix.task_close_direct_queue import (
    TASK_CLOSE_DIRECT_OPEN_STATUSES,
    TASK_CLOSE_DIRECT_QUEUE_PREFIX,
    TASK_CLOSE_DIRECT_STATUS_ACTIVE,
    auto_close_direct_close_event_as_unconfirmed,
)
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)


@dataclass
class DirectTaskCloseDraftDispatchStats:
    candidates: int = 0
    drafts_created: int = 0
    messages_sent: int = 0
    skipped: int = 0
    blocked: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "drafts_created": self.drafts_created,
            "messages_sent": self.messages_sent,
            "skipped": self.skipped,
            "blocked": self.blocked,
            "errors": list(self.errors),
        }


@dataclass
class DirectTaskCloseAutoCloseStats:
    due: bool = False
    auto_close_time: str = TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME
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
    bitrix: Any,
    bitrix_oauth: Any | None = None,
    store: Any,
    settings: Settings,
    status: dict[str, Any],
) -> None:
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
                bitrix=bitrix,
                bitrix_oauth=bitrix_oauth,
                store=store,
                settings=settings,
                status=status,
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
            logger.exception("Task close direct control tick failed")
            status["last_error"] = f"{type(exc).__name__}: {exc}"
            status["errors"] = int(status.get("errors") or 0) + 1
            await _sleep_until_next(status, min(interval_seconds, 300))


async def run_task_close_direct_control_once(
    *,
    bitrix: Any,
    bitrix_oauth: Any | None = None,
    store: Any,
    settings: Settings,
    status: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or _now()
    if status is not None:
        status["last_check_at"] = now_dt.isoformat()
    dispatch_stats = await dispatch_direct_task_close_drafts(
        bitrix=bitrix,
        store=store,
        settings=settings,
        limit=max(int(settings.bitrix_task_close_control_direct_limit), 1),
        now=now_dt,
    )
    auto_close_stats = await auto_close_direct_task_close_reports(
        bitrix=bitrix,
        bitrix_oauth=bitrix_oauth,
        store=store,
        settings=settings,
        limit=max(int(settings.bitrix_task_close_control_auto_close_limit), 1),
        now=now_dt,
    )
    result = {
        "checked_at": now_dt.isoformat(),
        "dispatch": dispatch_stats.as_dict(),
        "auto_close": auto_close_stats.as_dict(),
    }
    if status is not None:
        status.update(result)
    return result


async def dispatch_direct_task_close_drafts(
    *,
    bitrix: Any,
    store: Any,
    settings: Settings,
    limit: int = 20,
    now: datetime | None = None,
) -> DirectTaskCloseDraftDispatchStats:
    stats = DirectTaskCloseDraftDispatchStats()
    lister = getattr(store, "list_task_close_processing_states", None)
    if not callable(lister):
        return stats
    states = lister(
        statuses=[TASK_CLOSE_DIRECT_STATUS_ACTIVE],
        state_key_prefix=TASK_CLOSE_DIRECT_QUEUE_PREFIX,
        limit=limit,
    )
    stats.candidates = len(states)
    now_iso = (now or datetime.now(MOSCOW_TZ)).isoformat(timespec="seconds")
    for state in states:
        try:
            result = await _dispatch_one(bitrix=bitrix, store=store, settings=settings, state=state, now_iso=now_iso)
        except Exception as exc:  # pragma: no cover - defensive: one bad task must not stop metadata sync
            stats.errors.append(f"{state.get('task_id')}:{type(exc).__name__}: {exc}")
            continue
        if result == "created":
            stats.drafts_created += 1
            stats.messages_sent += 1
        elif result == "sent_existing":
            stats.messages_sent += 1
        elif result == "blocked":
            stats.blocked += 1
        else:
            stats.skipped += 1
    return stats


async def auto_close_direct_task_close_reports(
    *,
    bitrix: Any,
    bitrix_oauth: Any | None = None,
    store: Any,
    settings: Settings,
    limit: int = 100,
    now: datetime | None = None,
) -> DirectTaskCloseAutoCloseStats:
    now_dt = now or datetime.now(MOSCOW_TZ)
    auto_close_time = _configured_auto_close_time(store)
    stats = DirectTaskCloseAutoCloseStats(
        due=_is_auto_close_due(now_dt, auto_close_time),
        auto_close_time=auto_close_time,
    )
    if not stats.due:
        return stats
    lister = getattr(store, "list_task_close_processing_states", None)
    if not callable(lister):
        return stats

    states = lister(
        statuses=list(TASK_CLOSE_DIRECT_OPEN_STATUSES),
        state_key_prefix=TASK_CLOSE_DIRECT_QUEUE_PREFIX,
        limit=limit,
    )
    stats.candidates = len(states)
    if settings.agent_dry_run:
        stats.skipped = len(states)
        return stats
    now_iso = now_dt.isoformat(timespec="seconds")
    for state in states:
        payload = dict(state.get("payload") or {})
        close_event_key = str(payload.get("close_event_key") or "").strip()
        task_id = safe_int(state.get("task_id") or payload.get("task_id"))
        if task_id is None or not close_event_key:
            stats.skipped += 1
            continue
        try:
            draft_dialog_key = str(
                payload.get("draft_dialog_key") or _private_dialog_key(safe_int(payload.get("responsible_id")))
            )
            existing_draft = await _existing_draft(store, draft_dialog_key)
            draft = _auto_close_draft_from_state(state=state, payload=payload, existing_draft=existing_draft)
            result = await _execute_task_close(close_call=bitrix.call, report_call=bitrix.call, draft=draft)
            if existing_draft and _same_task_close_draft(existing_draft, task_id):
                deleter = getattr(store, "delete_task_draft", None)
                if callable(deleter) and draft_dialog_key:
                    await deleter(draft_dialog_key)
            auto_close_direct_close_event_as_unconfirmed(
                store,
                task_id=task_id,
                close_event_key=close_event_key,
                now_iso=now_iso,
                payload_updates={
                    "auto_close_report_method": result.get("report_method"),
                    "auto_close_report_file_name": result.get("report_file_name"),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive: one bad task must not stop the queue
            stats.errors.append(f"{task_id}:{type(exc).__name__}: {exc}")
            continue
        stats.reports_written += 1
        stats.closed += 1
    draft_rows = await _list_task_close_drafts(store, limit=limit)
    stats.open_drafts = len(draft_rows)
    for row in draft_rows:
        dialog_key = str(row.get("dialog_key") or "")
        draft = row.get("params")
        if not isinstance(draft, dict):
            stats.skipped += 1
            continue
        task_id = safe_int(draft.get("task_id"))
        if task_id is None:
            stats.skipped += 1
            continue
        if _truthy(draft.get("_direct_close_auto_closed")):
            stats.skipped += 1
            continue
        try:
            prepared = _auto_close_open_draft(draft)
            user_id = _user_id_from_dialog_key(dialog_key)
            result, actor_mode = await _execute_auto_close_with_fallback(
                bitrix=bitrix,
                bitrix_oauth=bitrix_oauth,
                draft=prepared,
                user_id=user_id,
            )
            deleter = getattr(store, "delete_task_draft", None)
            if callable(deleter) and dialog_key:
                deleted = deleter(dialog_key)
                if inspect.isawaitable(deleted):
                    await deleted
            if actor_mode == "oauth_user":
                stats.oauth_closed += 1
            elif actor_mode == "system_webhook_fallback":
                stats.system_fallback_closed += 1
                stats.admin_notifications += await _notify_admin_system_fallback(
                    bitrix=bitrix,
                    settings=settings,
                    draft=prepared,
                    result=result,
                )
            payload_updates = {
                "auto_close_report_method": result.get("report_method"),
                "auto_close_report_file_name": result.get("report_file_name"),
                "auto_close_actor_mode": actor_mode,
                "close_event_key": f"open_draft:{dialog_key or task_id}",
            }
            auto_close_direct_close_event_as_unconfirmed(
                store,
                task_id=task_id,
                close_event_key=payload_updates["close_event_key"],
                now_iso=now_iso,
                payload_updates=payload_updates,
            )
        except Exception as exc:  # pragma: no cover - defensive: one bad draft must not stop the batch
            stats.errors.append(f"draft:{task_id}:{type(exc).__name__}: {exc}")
            continue
        stats.reports_written += 1
        stats.closed += 1
    return stats


async def _dispatch_one(
    *,
    bitrix: Any,
    store: Any,
    settings: Settings,
    state: dict[str, Any],
    now_iso: str,
) -> str:
    payload = dict(state.get("payload") or {})
    if payload.get("direct_close_draft_sent_at"):
        return "skipped"
    task_id = safe_int(state.get("task_id") or payload.get("task_id"))
    close_event_key = str(payload.get("close_event_key") or "").strip()
    responsible_id = safe_int(payload.get("responsible_id"))
    recipient_id = str(payload.get("recipient_id") or responsible_id or "").strip()
    draft_dialog_key = str(payload.get("draft_dialog_key") or _private_dialog_key(responsible_id)).strip()
    if task_id is None or not close_event_key or not recipient_id or not draft_dialog_key:
        _update_state(
            store,
            state=state,
            payload={**payload, "dispatch_blocked_at": now_iso, "dispatch_blocked_reason": "missing_routing"},
        )
        return "blocked"

    draft = await _existing_draft(store, draft_dialog_key)
    if draft and not _same_task_close_draft(draft, task_id):
        _update_state(
            store,
            state=state,
            payload={
                **payload,
                "dispatch_blocked_at": now_iso,
                "dispatch_blocked_reason": "another_active_draft",
                "draft_dialog_key": draft_dialog_key,
                "recipient_id": recipient_id,
            },
        )
        return "blocked"

    created = False
    if not draft:
        draft = _build_direct_close_draft(state=state, payload=payload)
        saver = getattr(store, "save_task_draft", None)
        if not callable(saver):
            _update_state(
                store,
                state=state,
                payload={**payload, "dispatch_blocked_at": now_iso, "dispatch_blocked_reason": "draft_store_missing"},
            )
            return "blocked"
        await saver(draft_dialog_key, draft)
        created = True

    message = _direct_close_draft_message(draft)
    if not settings.agent_dry_run:
        await bitrix.send_bot_message(recipient_id, message, bot_id=settings.bitrix_bot_id)
    _update_state(
        store,
        state=state,
        payload={
            **payload,
            "recipient_id": recipient_id,
            "draft_dialog_key": draft_dialog_key,
            "direct_close_draft_sent_at": now_iso,
        },
    )
    return "created" if created else "sent_existing"


def _configured_auto_close_time(store: Any) -> str:
    getter = getattr(store, "get_task_close_control_setting", None)
    if not callable(getter):
        return TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME
    setting = getter(TASK_CLOSE_AUTO_CLOSE_TIME_KEY)
    if not isinstance(setting, dict):
        return TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME
    value = str(setting.get("value") or "").strip()
    return value if _parse_hhmm(value) is not None else TASK_CLOSE_DEFAULT_AUTO_CLOSE_TIME


def _is_auto_close_due(now: datetime, auto_close_time: str) -> bool:
    configured = _parse_hhmm(auto_close_time) or time(hour=20)
    return now.timetz().replace(tzinfo=None) >= configured


def _parse_hhmm(value: str) -> time | None:
    parts = value.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return time(hour=hour, minute=minute)


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)


async def _sleep_until_next(status: dict[str, Any], seconds: int) -> None:
    next_check_at = _now() + timedelta(seconds=seconds)
    status["next_check_at"] = next_check_at.isoformat()
    await asyncio.sleep(seconds)


def _auto_close_draft_from_state(
    *,
    state: dict[str, Any],
    payload: dict[str, Any],
    existing_draft: dict[str, Any] | None,
) -> dict[str, Any]:
    task_id = safe_int(state.get("task_id") or payload.get("task_id"))
    if existing_draft and task_id is not None and _same_task_close_draft(existing_draft, task_id):
        draft_args = dict(existing_draft)
        unconfirmed_items = _unique_strings(
            [
                *_string_list(existing_draft.get("unconfirmed_items")),
                *_string_list(existing_draft.get("unresolved_items")),
                "Draft was not confirmed before the auto-close time.",
            ]
        )
        draft_args.update(
            {
                "already_closed": True,
                "unconfirmed_items": unconfirmed_items,
            }
        )
        draft = build_task_close_draft_from_args(draft_args).payload
    else:
        draft = _build_direct_close_draft(state=state, payload=payload)
        draft["unconfirmed_items"] = _unique_strings(
            [
                *_string_list(draft.get("unconfirmed_items")),
                "No confirmed draft was received before the auto-close time.",
            ]
        )
        draft["problem_types"] = _unique_strings([*_string_list(draft.get("problem_types")), "unconfirmed"])
        draft["ai_close_incomplete"] = True
        draft["ai_close_marker"] = "AI_SERVER_TASK_CLOSE_INCOMPLETE"
    draft.update(
        {
            "already_closed": True,
            "_direct_close_already_closed": True,
            "_direct_close_queue_state_key": state.get("state_key"),
            "_direct_close_close_event_key": payload.get("close_event_key"),
            "_direct_close_closed_at": payload.get("closed_at"),
            "_direct_close_detected_at": payload.get("seen_at"),
            "_direct_close_task_url": payload.get("task_url"),
            "_direct_close_auto_closed": True,
        }
    )
    return draft


def _auto_close_open_draft(draft: dict[str, Any]) -> dict[str, Any]:
    draft_args = dict(draft)
    unconfirmed_items = _unique_strings(
        [
            *_string_list(draft.get("unconfirmed_items")),
            *_string_list(draft.get("unresolved_items")),
            "Draft was not confirmed before the auto-close time.",
        ]
    )
    draft_args.update(
        {
            "already_closed": False,
            "unconfirmed_items": unconfirmed_items,
        }
    )
    prepared = build_task_close_draft_from_args(draft_args).payload
    prepared.update(
        {
            "_direct_close_auto_closed": True,
            "_auto_close_open_draft": True,
        }
    )
    return prepared


async def _execute_auto_close_with_fallback(
    *,
    bitrix: Any,
    bitrix_oauth: Any | None,
    draft: dict[str, Any],
    user_id: int | None,
) -> tuple[dict[str, Any], str]:
    if bitrix_oauth is not None and user_id is not None:
        close_stage_error = True
        try:
            oauth_client = await bitrix_oauth.client_for_user(user_id)
            close_stage_error = True

            async def oauth_close_call(method: str, payload: dict[str, Any]) -> Any:
                nonlocal close_stage_error
                close_stage_error = True
                return await oauth_client.call(method, payload)

            async def system_report_call(method: str, payload: dict[str, Any]) -> Any:
                nonlocal close_stage_error
                close_stage_error = False
                return await bitrix.call(method, payload)

            result = await _execute_task_close(
                close_call=oauth_close_call,
                report_call=system_report_call,
                draft=draft,
            )
            result["auto_close_actor_mode"] = "oauth_user"
            result["auto_close_actor_user_id"] = user_id
            return result, "oauth_user"
        except (BitrixOAuthError, BitrixApiError, BitrixConfigError) as exc:
            if not close_stage_error:
                raise
            fallback_draft = _system_fallback_draft(draft, reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            if not close_stage_error:
                raise
            fallback_draft = _system_fallback_draft(draft, reason=f"{type(exc).__name__}: {exc}")
    else:
        reason = "OAuth user context is not available for auto-close."
        fallback_draft = _system_fallback_draft(draft, reason=reason)

    result = await _execute_task_close(close_call=bitrix.call, report_call=bitrix.call, draft=fallback_draft)
    result["auto_close_actor_mode"] = "system_webhook_fallback"
    result["auto_close_system_webhook_fallback_reason"] = str(
        fallback_draft.get("_auto_close_system_webhook_fallback_reason") or ""
    )
    return result, "system_webhook_fallback"


def _system_fallback_draft(draft: dict[str, Any], *, reason: str) -> dict[str, Any]:
    draft_args = dict(draft)
    unconfirmed_items = _unique_strings(
        [
            *_string_list(draft.get("unconfirmed_items")),
            f"Auto-close used system webhook fallback: {reason}",
        ]
    )
    draft_args.update(
        {
            "unconfirmed_items": unconfirmed_items,
            "_auto_close_system_webhook_fallback": True,
            "_auto_close_system_webhook_fallback_reason": reason,
        }
    )
    prepared = build_task_close_draft_from_args(draft_args).payload
    prepared["_auto_close_system_webhook_fallback"] = True
    prepared["_auto_close_system_webhook_fallback_reason"] = reason
    return prepared


async def _notify_admin_system_fallback(
    *,
    bitrix: Any,
    settings: Settings,
    draft: dict[str, Any],
    result: dict[str, Any],
) -> int:
    sender = getattr(bitrix, "send_bot_message", None)
    if not callable(sender):
        return 0
    admin_ids = sorted(settings.resolved_task_close_report_admin_user_ids)
    if not admin_ids:
        return 0
    task_id = str(draft.get("task_id") or "")
    task_title = str(draft.get("task_title") or f"#{task_id}").strip()
    reason = str(result.get("auto_close_system_webhook_fallback_reason") or "")
    if not reason:
        reason = str(draft.get("_auto_close_system_webhook_fallback_reason") or "")
    report_file_name = str(result.get("report_file_name") or "")
    message = "\n".join(
        [
            "Автозакрытие задачи выполнено системным webhook.",
            f"Задача: {task_title}",
            f"ID: {task_id}",
            f"AI-close файл: {report_file_name or 'создан/обновлен'}",
            f"Причина fallback: {reason or 'OAuth закрытие недоступно'}",
        ]
    )
    sent = 0
    for admin_id in admin_ids:
        await sender(str(admin_id), message, bot_id=settings.bitrix_bot_id)
        sent += 1
    return sent


async def _existing_draft(store: Any, dialog_key: str) -> dict[str, Any] | None:
    getter = getattr(store, "get_task_draft", None)
    if not callable(getter):
        return None
    draft = await getter(dialog_key)
    return dict(draft) if isinstance(draft, dict) else None


async def _list_task_close_drafts(store: Any, *, limit: int) -> list[dict[str, Any]]:
    lister = getattr(store, "list_task_drafts", None)
    if not callable(lister):
        return []
    rows = lister(draft_type=TASK_CLOSE_DRAFT_TYPE, limit=limit)
    if inspect.isawaitable(rows):
        rows = await rows
    return [dict(row) for row in rows if isinstance(row, dict)]


def _same_task_close_draft(draft: dict[str, Any], task_id: int) -> bool:
    return draft.get("_draft_type") == TASK_CLOSE_DRAFT_TYPE and safe_int(draft.get("task_id")) == task_id


def _build_direct_close_draft(*, state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    task_id = safe_int(state.get("task_id") or payload.get("task_id"))
    task_results = _string_list(payload.get("task_results"))
    task_points = _string_list(payload.get("task_points"))
    result_summary = "\n".join(task_results).strip()
    source_empty = bool(payload.get("source_task_description_empty"))
    missing_fields = [
        "Подтвердите, что результат закрытия соответствует задаче.",
        "Укажите оборудование и расходники, если они использовались.",
    ]
    if not result_summary:
        missing_fields.insert(0, "Кратко напишите, что было сделано по задаче.")
    draft = build_task_close_draft_from_args(
        {
            "task_id": task_id,
            "task_title": payload.get("task_title"),
            "completion_summary": result_summary,
            "task_points": task_points,
            "source_task_description_empty": source_empty,
            "equipment_consumables": "",
            "overall_status": "unconfirmed",
            "unconfirmed_items": [
                "Результат закрытия задачи не подтверждён через AI-черновик.",
            ],
            "missing_fields": missing_fields,
            "action": "complete",
        }
    ).payload
    draft.update(
        {
            "already_closed": True,
            "_direct_close_already_closed": True,
            "_direct_close_queue_state_key": state.get("state_key"),
            "_direct_close_close_event_key": payload.get("close_event_key"),
            "_direct_close_closed_at": payload.get("closed_at"),
            "_direct_close_detected_at": payload.get("seen_at"),
            "_direct_close_task_url": payload.get("task_url"),
        }
    )
    return draft


def _direct_close_draft_message(draft: dict[str, Any]) -> str:
    return format_task_close_draft_message(
        draft,
        intro_lines=[
            "Задача закрыта напрямую в Bitrix.",
            "Нужно подтвердить результат для AI-контроля.",
        ],
    )


def _update_state(store: Any, *, state: dict[str, Any], payload: dict[str, Any]) -> None:
    upsert = getattr(store, "upsert_task_close_processing_state", None)
    if not callable(upsert):
        return
    upsert(
        task_id=state.get("task_id"),
        state_key=str(state.get("state_key") or ""),
        status=str(state.get("status") or TASK_CLOSE_DIRECT_STATUS_ACTIVE),
        payload=payload,
        actor_user_id=safe_int(state.get("actor_user_id")),
    )


def _private_dialog_key(user_id: int | None) -> str:
    if user_id is None or user_id <= 0:
        return ""
    return f"dialog:{user_id}:user:{user_id}"


def _string_list(value: object) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        return [str(value).strip()]
    return []


def _user_id_from_dialog_key(value: object) -> int | None:
    parts = str(value or "").split(":")
    for part in reversed(parts):
        user_id = safe_int(part)
        if user_id is not None and user_id > 0:
            return user_id
    return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def safe_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
