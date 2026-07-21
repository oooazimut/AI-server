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
    """The one unambiguous user-facing confirmation phrase for a draft type."""
    return _PHRASES.get(str(draft_type or "").strip(), _PHRASES["task_create"])


def matches_draft_confirmation(
    request: str,
    draft: dict[str, Any] | None,
    *,
    allow_short_command: bool = False,
) -> bool:
    """Accept the displayed confirmation despite harmless voice-recognition noise.

    A plain ``да`` is deliberately insufficient: the draft type must still be
    evident, so a confirmation for a calendar item cannot create a task.
    """
    if not isinstance(draft, dict) or not draft:
        return False
    expected = _normalize(draft_confirmation_phrase(str(draft.get("_draft_type") or "task_create")))
    actual = _normalize(request)
    if allow_short_command and actual in {"подтвердить", "подтверждаю", "подтверждение"}:
        return True
    if actual == expected:
        return True
    expected_words = expected.split()
    actual_words = actual.split()
    # Voice input commonly drops a comma, one short connective, or changes a
    # single final letter.  Accept at most one such harmless omission/typo, but
    # never turn an unrelated "да" into confirmation of a write draft.
    if len(actual_words) < max(2, len(expected_words) - 1) or len(actual_words) > len(expected_words) + 1:
        return False
    if _word_distance(expected_words, actual_words) > 1:
        return False
    # The action type is a safety boundary.  A voice typo in it is acceptable,
    # but omitting it must never turn a task confirmation into a calendar or
    # project confirmation (or vice versa).
    required = _REQUIRED_TYPE_WORDS.get(str(draft.get("_draft_type") or "task_create"), ("задачи",))
    return all(any(_character_distance(word, seen) <= 1 for seen in actual_words) for word in required)


def _normalize(value: str) -> str:
    value = re.sub(r"[^\w\s-]", " ", value.casefold(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", value.strip())


def _word_distance(expected: list[str], actual: list[str]) -> int:
    """Small edit distance over words, with one-character word typos allowed."""
    rows = list(range(len(actual) + 1))
    for i, left in enumerate(expected, start=1):
        previous, rows[0] = rows[0], i
        for j, right in enumerate(actual, start=1):
            old = rows[j]
            same = left == right or _character_distance(left, right) <= 1
            rows[j] = min(rows[j] + 1, rows[j - 1] + 1, previous + (0 if same else 2))
            previous = old
    return rows[-1]


def _character_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return 2
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right))
    longer, shorter = (left, right) if len(left) > len(right) else (right, left)
    for index in range(len(longer)):
        if longer[:index] + longer[index + 1 :] == shorter:
            return 1
    return 2
