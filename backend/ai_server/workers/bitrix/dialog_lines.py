from __future__ import annotations

import re

DEFAULT_AUTO_LINE_MAX = 3


def dialog_line_label(line_id: str | int) -> str:
    return f"Линия {int(line_id)}"


def line_partition_key(base_dialog_key: str, line_id: str | int) -> str:
    return f"dialog:{base_dialog_key}:line:{int(line_id)}"


def base_partition_key(base_dialog_key: str) -> str:
    return f"dialog:{base_dialog_key}"


def active_line_ids(active_partition_keys: set[str], base_dialog_key: str) -> set[int]:
    prefix = f"dialog:{base_dialog_key}:line:"
    result: set[int] = set()
    for partition_key in active_partition_keys:
        if not partition_key.startswith(prefix):
            continue
        raw = partition_key.removeprefix(prefix).split(":", 1)[0]
        try:
            result.add(int(raw))
        except ValueError:
            continue
    return result


def choose_auto_line_id(
    active_partition_keys: set[str],
    base_dialog_key: str,
    *,
    max_lines: int = DEFAULT_AUTO_LINE_MAX,
) -> int | None:
    if not base_dialog_key:
        return None
    max_lines = max(1, int(max_lines or DEFAULT_AUTO_LINE_MAX))
    used = active_line_ids(active_partition_keys, base_dialog_key)
    if base_partition_key(base_dialog_key) in active_partition_keys:
        used.add(1)
    if not used:
        return None
    for line_id in range(1, max_lines + 1):
        if line_id not in used:
            return line_id
    return None


def is_auto_line_candidate(text: str) -> bool:
    """Return True when a message looks like a new independent Bitrix request."""
    normalized = re.sub(r"\s+", " ", str(text or "").casefold()).strip()
    if not normalized:
        return False
    if _is_short_followup(normalized):
        return False
    if normalized.startswith(("битрикс", "bitrix", "bittrex")):
        return True
    has_command = re.search(r"\b(найди|найти|покажи|показать|выведи|дай|список|искать)\b", normalized)
    has_bitrix_object = re.search(
        r"\b(задач|задачи|задачу|проект|проекты|склад|остат|документ|файл|битрикс|bitrix)\b",
        normalized,
    )
    return bool(has_command and has_bitrix_object)


def _is_short_followup(text: str) -> bool:
    cleaned = text.strip(" .,!?:;")
    return cleaned in {
        "да",
        "нет",
        "ок",
        "окей",
        "хорошо",
        "согласен",
        "подтверждаю",
        "отмена",
        "отмени",
        "стоп",
    }
