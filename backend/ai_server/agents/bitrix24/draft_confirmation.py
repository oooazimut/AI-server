from __future__ import annotations

import re
from typing import Any


_PHRASES = {
    "task_create": "да, подтверждаю создание задачи",
    "task_close": "да, закрываю задачу как есть",
    "calendar_event": "да, подтверждаю создание записи в календаре",
    "project_create": "да, подтверждаю создание проекта",
    "admin_change": "да, подтверждаю изменение настройки",
}


def draft_confirmation_phrase(draft_type: str | None) -> str:
    """The one unambiguous user-facing confirmation phrase for a draft type."""
    return _PHRASES.get(str(draft_type or "").strip(), _PHRASES["task_create"])


def matches_draft_confirmation(request: str, draft: dict[str, Any] | None) -> bool:
    """Accept only the displayed phrase, ignoring harmless casing and final punctuation."""
    if not isinstance(draft, dict) or not draft:
        return False
    expected = _normalize(draft_confirmation_phrase(str(draft.get("_draft_type") or "task_create")))
    return _normalize(request) == expected


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold()).rstrip(".!?")
