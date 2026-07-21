"""Stable, user-visible references for independent dialog branches.

The visible number is deliberately only a hint: a short continuation may use
the most recent live branch when it is unambiguous.  The full date stays in
the durable key, so a new day can safely start its visible sequence again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ai_server.models import AgentTask
from ai_server.utils import MOSCOW_TZ

_PREFIX = re.compile(r"^\s*(?P<number>\d{3,})\s*(?:[,.:;—–-]\s*|\s+)(?P<text>.+)$", re.DOTALL)
_CONTINUATION = re.compile(
    r"\b(?:следующ|продолж|подтверж|отмен|закрыва|да\b|нет\b|покажи\s+ещ[её])",
    re.IGNORECASE,
)
_COUNTER_FIELD = "conversation_reference_counter"
_CURRENT_FIELD = "conversation_reference_current"
_CURRENT_AT_FIELD = "conversation_reference_current_at"
_RECENT_FIELD = "conversation_reference_recent"
_FIELD_PREFIX = "conversation_reference:"
_VISIBLE_START = 100
_AUTO_CONTINUATION_TTL = timedelta(minutes=30)


@dataclass(frozen=True)
class ConversationReferenceResolution:
    task: AgentTask
    error: str = ""


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)


def _reference_field(day: str, number: int) -> str:
    return f"{_FIELD_PREFIX}{day}:{number}"


def _branch_key(base_dialog_key: str, day: str, number: int) -> str:
    return f"{base_dialog_key}:conversation:{day}:{number}"


def _prefixed_reference(request: str) -> tuple[int | None, str]:
    match = _PREFIX.match(request or "")
    if not match:
        return None, request
    return int(match.group("number")), match.group("text").strip()


def _looks_like_continuation(request: str) -> bool:
    return bool(_CONTINUATION.search((request or "").strip()))


def _recent_numbers(value: str | None, *, now: datetime) -> dict[int, datetime]:
    result: dict[int, datetime] = {}
    for item in str(value or "").split(","):
        number, separator, epoch = item.partition("@")
        if not separator or not number.isdigit() or not epoch.isdigit():
            continue
        at = datetime.fromtimestamp(int(epoch), tz=MOSCOW_TZ)
        if now - at <= _AUTO_CONTINUATION_TTL:
            result[int(number)] = at
    return result


async def _touch_recent(store: Any, base_key: str, *, number: int, now: datetime) -> None:
    recent = _recent_numbers(await store.get_kv(base_key, _RECENT_FIELD), now=now)
    recent[number] = now
    compact = sorted(recent.items(), key=lambda item: item[1], reverse=True)[:10]
    await store.set_kv(base_key, _RECENT_FIELD, ",".join(f"{item_number}@{int(at.timestamp())}" for item_number, at in compact))


async def resolve_conversation_reference(task: AgentTask, store: Any) -> ConversationReferenceResolution:
    """Attach a task to an explicit or safely inferred numbered branch.

    This function intentionally does *not* alter queue partitioning.  The
    first rollout adds durable identity and an unambiguous continuation path;
    a later approved rollout can safely use the branch key for parallel reads.
    """
    context = dict(task.context or {})
    # Numbered branches are a Bitrix-chat surface.  Keep direct/internal tasks
    # and old callers untouched until they explicitly provide the base key.
    base_key = str(context.get("base_dialog_key") or "").strip()
    if not base_key or store is None or not hasattr(store, "get_kv") or not hasattr(store, "set_kv"):
        return ConversationReferenceResolution(task=task)

    request = str(task.request or "").strip()
    explicit_number, clean_request = _prefixed_reference(request)
    now = _now()
    day = now.strftime("%Y%m%d")
    branch_key = ""
    number: int | None = None

    if explicit_number is not None:
        mapped = await store.get_kv(base_key, _reference_field(day, explicit_number))
        if not mapped:
            return ConversationReferenceResolution(
                task=task,
                error=f"Диалог №{explicit_number} за сегодня не найден. Начните новый запрос без номера.",
            )
        branch_key, number = str(mapped), explicit_number
        request = clean_request
    elif _looks_like_continuation(request):
        recent = _recent_numbers(await store.get_kv(base_key, _RECENT_FIELD), now=now)
        if len(recent) > 1:
            known = ", ".join(str(item) for item in sorted(recent))
            return ConversationReferenceResolution(
                task=task,
                error=f"Есть несколько активных диалогов: №{known}. Начните сообщение с нужного номера.",
            )
        if len(recent) == 1:
            number = next(iter(recent))
            mapped = await store.get_kv(base_key, _reference_field(day, number))
            if mapped:
                branch_key = str(mapped)

    if not branch_key:
        raw_counter = await store.get_kv(base_key, _COUNTER_FIELD)
        stored_day, separator, stored_number = str(raw_counter or "").partition(":")
        previous = int(stored_number) if separator and stored_day == day and stored_number.isdigit() else _VISIBLE_START
        number = previous + 1
        branch_key = _branch_key(base_key, day, number)
        await store.set_kv(base_key, _COUNTER_FIELD, f"{day}:{number}")
        await store.set_kv(base_key, _reference_field(day, number), branch_key)

    assert number is not None
    await store.set_kv(base_key, _CURRENT_FIELD, branch_key)
    await store.set_kv(base_key, _CURRENT_AT_FIELD, now.isoformat())
    await _touch_recent(store, base_key, number=number, now=now)
    return ConversationReferenceResolution(
        task=task.model_copy(
            update={
                "request": request,
                "context": {
                    **context,
                    "base_dialog_key": base_key,
                    "dialog_key": branch_key,
                    "conversation_number": number,
                    "conversation_day": day,
                    "conversation_original_request": str(task.request or ""),
                },
            }
        )
    )
