"""Stable, user-visible references for independent dialog branches.

The visible number selects one exact live branch.  Continuations never guess
the most recent branch: this keeps concurrent user conversations isolated.
The full date stays in the durable key, so a new day can safely start its
visible sequence again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ai_server.models import AgentTask
from ai_server.utils import MOSCOW_TZ

_PREFIX = re.compile(
    r"^\s*(?P<number>\d{3,})\s*(?:[,.:;—–-]\s*|\s+)(?P<text>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_SUFFIX = re.compile(
    r"^(?P<text>.+?)(?:\s*[,.:;—–-]\s*|\s+)(?P<number>\d{3,})\s*$",
    re.IGNORECASE | re.DOTALL,
)
_CONTINUATION = re.compile(
    r"\b(?:следующ|продолж|подтверж|подтверд|отмен|закрыва|измен|да\b|нет\b|покажи\s+ещ[её])",
    re.IGNORECASE,
)
_LEGACY_PART_REFERENCE = re.compile(r"\b\d{3,}\s*(?:[,.:;—–-]\s*|\s+)част(?:ь|и)\s+\d+\b", re.IGNORECASE)
_COUNTER_FIELD = "conversation_reference_counter"
_CURRENT_FIELD = "conversation_reference_current"
_CURRENT_AT_FIELD = "conversation_reference_current_at"
_RECENT_FIELD = "conversation_reference_recent"
_ACTIVE_DRAFT_BRANCH_FIELD = "conversation_reference_active_draft_branch"
_ACTIVE_DRAFT_NUMBER_FIELD = "conversation_reference_active_draft_number"
_FIELD_PREFIX = "conversation_reference:"
_VISIBLE_START = 100
_AUTO_CONTINUATION_TTL = timedelta(minutes=15)


@dataclass(frozen=True)
class ConversationReferenceResolution:
    task: AgentTask
    error: str = ""


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)


def _reference_field(day: str, number: int) -> str:
    return f"{_FIELD_PREFIX}{day}:{number}"


def _reference_at_field(day: str, number: int) -> str:
    return f"{_FIELD_PREFIX}{day}:{number}:at"


def _branch_key(base_dialog_key: str, day: str, number: int) -> str:
    return f"{base_dialog_key}:conversation:{day}:{number}"


def _explicit_reference(request: str) -> tuple[int | None, str]:
    match = _PREFIX.match(request or "")
    if match and _looks_like_continuation(match.group("text")):
        return int(match.group("number")), match.group("text").strip()
    match = _SUFFIX.match(request or "")
    if match and _looks_like_continuation(match.group("text")):
        return int(match.group("number")), match.group("text").strip()
    return None, request


def _looks_like_continuation(request: str) -> bool:
    return bool(_CONTINUATION.search((request or "").strip()))


def _looks_like_draft_write(request: str) -> bool:
    lowered = str(request or "").casefold()
    return bool(
        re.search(
            r"\b(?:созда(?:й|ть|йте)|напомни|календар|закр(?:ой|ыть|ойте)\s+задач|"
            r"оператор|контрол(?:ь|я)\s+закрытия|время\s+автозакрытия)\b",
            lowered,
        )
    )


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
    await store.set_kv(
        base_key, _RECENT_FIELD, ",".join(f"{item_number}@{int(at.timestamp())}" for item_number, at in compact)
    )


def _is_live_reference(value: str | None, *, now: datetime) -> bool:
    try:
        touched = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    if touched.tzinfo is None:
        return False
    return now - touched.astimezone(MOSCOW_TZ) <= _AUTO_CONTINUATION_TTL


async def _touch_reference(store: Any, base_key: str, *, day: str, number: int, now: datetime) -> None:
    await store.set_kv(base_key, _reference_at_field(day, number), now.isoformat())
    await _touch_recent(store, base_key, number=number, now=now)


def _reference_constraint(task: AgentTask, context: dict[str, Any], message: str) -> ConversationReferenceResolution:
    """Keep the user turn for mandatory Pro planning, but fail closed on dispatch."""
    return ConversationReferenceResolution(
        task=task.model_copy(
            update={
                "context": {
                    **context,
                    "conversation_reference_error": message,
                    "conversation_reference_dispatch_allowed": False,
                    "conversation_original_request": str(task.request or ""),
                }
            }
        ),
        error=message,
    )


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
    # Earlier experimental replies exposed a root number together with a
    # synthetic "part" number.  Parts are no longer a user-facing contract:
    # accepting them would silently reintroduce the ambiguous behaviour that
    # this resolver is meant to remove.  Keep the turn for mandatory planning,
    # but prevent any specialist dispatch.
    if _LEGACY_PART_REFERENCE.search(request):
        return _reference_constraint(
            task,
            context,
            "Части ответа больше не используются. Укажите склад или другой объект явно.",
        )
    explicit_number, clean_request = _explicit_reference(request)
    now = _now()
    day = now.strftime("%Y%m%d")
    branch_key = ""
    number: int | None = None
    reused_active_draft = False

    if explicit_number is not None:
        mapped = await store.get_kv(base_key, _reference_field(day, explicit_number))
        touched = await store.get_kv(base_key, _reference_at_field(day, explicit_number))
        if not mapped or not _is_live_reference(touched, now=now):
            return _reference_constraint(
                task,
                context,
                f"Диалог {explicit_number} не активен. Начните новый запрос без номера.",
            )
        branch_key, number = str(mapped), explicit_number
        request = clean_request
    elif _looks_like_continuation(request):
        return _reference_constraint(
            task,
            context,
            "Для продолжения, подтверждения, отмены или изменения укажите номер диалога: «122 подтвердить».",
        )
    elif _looks_like_draft_write(request):
        active_branch = str(await store.get_kv(base_key, _ACTIVE_DRAFT_BRANCH_FIELD) or "").strip()
        active_number = str(await store.get_kv(base_key, _ACTIVE_DRAFT_NUMBER_FIELD) or "").strip()
        if active_branch and active_number.isdigit():
            candidate = int(active_number)
            mapped = await store.get_kv(base_key, _reference_field(day, candidate))
            touched = await store.get_kv(base_key, _reference_at_field(day, candidate))
            if str(mapped or "") == active_branch and _is_live_reference(touched, now=now):
                branch_key, number = active_branch, candidate
                reused_active_draft = True

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
    await _touch_reference(store, base_key, day=day, number=number, now=now)
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
                    "conversation_reference_explicit": explicit_number is not None,
                    "conversation_reference_reused_active_draft": reused_active_draft,
                    "conversation_original_request": str(task.request or ""),
                },
            }
        )
    )
