from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_server.agents.bitrix24.tools.task_close import TASK_CLOSE_DRAFT_TYPE, build_task_close_draft_from_args
from ai_server.integrations.bitrix.task_close_direct_queue import (
    TASK_CLOSE_DIRECT_QUEUE_PREFIX,
    TASK_CLOSE_DIRECT_STATUS_ACTIVE,
)
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ


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


async def _existing_draft(store: Any, dialog_key: str) -> dict[str, Any] | None:
    getter = getattr(store, "get_task_draft", None)
    if not callable(getter):
        return None
    draft = await getter(dialog_key)
    return dict(draft) if isinstance(draft, dict) else None


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
    task_id = str(draft.get("task_id") or "")
    title = str(draft.get("task_title") or f"#{task_id}").strip()
    result = str(draft.get("completion_summary") or "").strip()
    task_points = _string_list(draft.get("task_points"))
    missing = _string_list(draft.get("missing_fields"))
    unconfirmed = _string_list(draft.get("unconfirmed_items"))
    lines = [
        "Задача закрыта напрямую в Bitrix.",
        "Нужно подтвердить результат для AI-контроля.",
        "",
        "Черновик закрытия задачи:",
        f"Задача: {title}",
        "Действие: сохранить AI-отчёт по закрытию",
    ]
    if task_points:
        lines.append("Пункты задачи:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(task_points, start=1))
    else:
        lines.append("Описание выполненной работы:")
        lines.append(f"- {result or '? кратко напишите, что было сделано'}")
    lines.append("Оборудование, расходники: ? что использовано")
    if result and task_points:
        lines.append(f"Результат: {result}")
    lines.append("Итог: не подтверждено")
    if unconfirmed:
        lines.append("Не подтверждено:")
        lines.extend(f"- {item}" for item in unconfirmed)
    if missing:
        lines.append("Нужно дописать:")
        lines.extend(f"- {item}" for item in missing)
    lines.extend(
        [
            "",
            'Действия: допишите данные или напишите "да, закрывай как есть".',
        ]
    )
    return "\n".join(lines)


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


def safe_int(value: object | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
