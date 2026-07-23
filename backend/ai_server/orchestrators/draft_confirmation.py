"""Orchestrator-owned recognition of explicit Bitrix draft confirmations."""

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

_REQUIRED_TYPE_WORDS = {
    "task_create": ("задачи",),
    "task_close": ("закрываю", "задачу"),
    "calendar_event": ("календаре",),
    "project_create": ("проекта",),
    "admin_change": ("настройки",),
}


def draft_confirmation_phrase(draft_type: str | None) -> str:
    """Return the one unambiguous user-facing confirmation phrase."""
    return _PHRASES.get(str(draft_type or "").strip(), _PHRASES["task_create"])


def matches_draft_confirmation(
    request: str,
    draft: dict[str, Any] | None,
    *,
    allow_short_command: bool = False,
) -> bool:
    """Recognize the displayed confirmation despite harmless voice noise."""
    if not isinstance(draft, dict) or not draft:
        return False
    draft_type = str(draft.get("_draft_type") or "task_create")
    expected = _normalize(draft_confirmation_phrase(draft_type))
    actual = _normalize(request)
    if allow_short_command and actual in {"подтвердить", "подтверждаю", "подтверждение"}:
        return True
    if actual == expected:
        return True
    expected_words = expected.split()
    actual_words = actual.split()
    if len(actual_words) < max(2, len(expected_words) - 1) or len(actual_words) > len(expected_words) + 1:
        return False
    if _word_distance(expected_words, actual_words) > 1:
        return False
    required = _REQUIRED_TYPE_WORDS.get(draft_type, ("задачи",))
    return all(any(_character_distance(word, seen) <= 1 for seen in actual_words) for word in required)


def _normalize(value: str) -> str:
    value = re.sub(r"[^\w\s-]", " ", value.casefold(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", value.strip())


def _word_distance(expected: list[str], actual: list[str]) -> int:
    rows = list(range(len(actual) + 1))
    for index, left in enumerate(expected, start=1):
        previous, rows[0] = rows[0], index
        for inner_index, right in enumerate(actual, start=1):
            old = rows[inner_index]
            same = left == right or _character_distance(left, right) <= 1
            rows[inner_index] = min(
                rows[inner_index] + 1,
                rows[inner_index - 1] + 1,
                previous + (0 if same else 2),
            )
            previous = old
    return rows[-1]


def _character_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return 2
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True))
    longer, shorter = (left, right) if len(left) > len(right) else (right, left)
    for index in range(len(longer)):
        if longer[:index] + longer[index + 1 :] == shorter:
            return 1
    return 2
