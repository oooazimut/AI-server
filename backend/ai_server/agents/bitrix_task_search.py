from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ai_server.models import AgentTask, ToolResult
from ai_server.settings import get_settings


MOSCOW_TZ = timezone(timedelta(hours=3))
DEFAULT_TASK_SELECT = [
    "ID",
    "TITLE",
    "STATUS",
    "RESPONSIBLE_ID",
    "CREATED_BY",
    "DEADLINE",
    "CHANGED_DATE",
    "CLOSED_DATE",
    "GROUP_ID",
]
TASK_STATUS_LABELS = {
    "1": "новая",
    "2": "ждет выполнения",
    "3": "выполняется",
    "4": "ждет контроля",
    "5": "завершена",
    "6": "отложена",
    "7": "отклонена",
}
TASK_MARKERS = ("задач", "заявк")
SEARCH_MARKERS = ("найди", "найти", "покажи", "показать", "какие", "список", "посмотри")
CREATE_MARKERS = ("создай", "создать", "поставь задачу", "поставить задачу")


@dataclass(frozen=True)
class BitrixTaskSearchDraft:
    args: dict[str, Any]
    summary: str


def looks_like_task_search_request(text: str) -> bool:
    normalized = _normalize(text)
    if any(marker in normalized for marker in CREATE_MARKERS):
        return False
    if _extract_task_id(text) is not None and any(marker in normalized for marker in TASK_MARKERS):
        return True
    return any(marker in normalized for marker in SEARCH_MARKERS) and any(
        marker in normalized for marker in TASK_MARKERS
    )


def build_task_search_draft(task: AgentTask) -> BitrixTaskSearchDraft:
    request = task.request.strip()
    task_id = _extract_task_id(request)
    if task_id is not None:
        return BitrixTaskSearchDraft(
            args={
                "mode": "get",
                "task_id": task_id,
                "select": DEFAULT_TASK_SELECT,
            },
            summary=f"найти задачу #{task_id}",
        )

    normalized = _normalize(request)
    current_user_id = _optional_int(task.user.id)
    filter_: dict[str, Any] = {}
    notes: list[str] = []

    status_filter = _status_filter(normalized)
    if status_filter:
        filter_.update(status_filter)

    if current_user_id is not None:
        if any(marker in normalized for marker in ("мои", "моя", "мою", "у меня", "на мне", "мне назнач")):
            filter_["RESPONSIBLE_ID"] = current_user_id
            notes.append("только задачи текущего пользователя")
        elif any(marker in normalized for marker in ("поставленные мной", "созданные мной", "я создал")):
            filter_["CREATED_BY"] = current_user_id
            notes.append("только задачи, созданные текущим пользователем")

    title_query = _extract_title_query(request)
    if title_query:
        filter_["%TITLE"] = title_query
        notes.append(f"по названию `{title_query}`")

    limit = _extract_limit(normalized)
    args = {
        "mode": "list",
        "filter": filter_,
        "select": DEFAULT_TASK_SELECT,
        "order": {"CHANGED_DATE": "desc"},
        "limit": limit,
    }
    summary_parts = ["найти задачи"]
    if notes:
        summary_parts.append("(" + "; ".join(notes) + ")")
    return BitrixTaskSearchDraft(args=args, summary=" ".join(summary_parts))


def format_task_search_answer(result: ToolResult) -> str:
    if result.status == "not_configured":
        return "Не смог выполнить поиск задач: Bitrix API или поисковый инструмент пока не настроен."
    if result.status == "error":
        return f"Не смог выполнить поиск задач: {result.error or 'ошибка Bitrix API'}."
    if result.status == "invalid_tool_call":
        return f"Не смог выполнить поиск задач: {result.error or 'некорректный запрос'}."
    if result.status == "not_found":
        return "Задачу не нашёл."
    if result.status != "ok":
        return f"Поиск задач вернул статус `{result.status}`."

    mode = str(result.data.get("mode") or "")
    if mode == "get":
        task = result.data.get("task")
        if not isinstance(task, dict):
            return "Задачу не нашёл."
        return "Нашёл задачу:\n" + _format_task_line(task)

    tasks = result.data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return "По этому запросу задач не нашёл."

    total = result.data.get("total")
    lines = [_format_task_line(task, index=index) for index, task in enumerate(tasks[:10], start=1) if isinstance(task, dict)]
    suffix = ""
    if isinstance(total, int) and total > len(lines):
        suffix = f"\n\nПоказал первые {len(lines)} из {total}."
    return "Нашёл задачи:\n" + "\n".join(lines) + suffix


def _status_filter(normalized: str) -> dict[str, Any]:
    if any(marker in normalized for marker in ("закрыт", "заверш", "выполненн")):
        return {"STATUS": 5}
    if any(marker in normalized for marker in ("контрол", "провер")):
        return {"STATUS": 4}
    if any(marker in normalized for marker in ("в работе", "выполня", "делает")):
        return {"STATUS": 3}
    if any(marker in normalized for marker in ("просроч", "горящ")):
        return {
            "<DEADLINE": datetime.now(MOSCOW_TZ).isoformat(),
            "!STATUS": 5,
        }
    if any(marker in normalized for marker in ("открыт", "активн", "незакрыт", "невыполн")):
        return {"!STATUS": 5}
    return {}


def _extract_task_id(text: str) -> int | None:
    patterns = (
        r"(?:задач[аиуе]?|заявк[аиуе]?)\s*#?\s*(\d{2,10})",
        r"#\s*(\d{2,10})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_title_query(text: str) -> str:
    value = text
    value = re.sub(r"\b(?:найди|найти|покажи|показать|посмотри|список|какие)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:мои|моя|мою|у меня|на мне|открытые|активные|закрытые|завершенные|завершённые|просроченные)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:задачи|задачу|задача|заявки|заявку|заявка|в битриксе|битриксе)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"#?\d{2,10}", "", value)
    value = re.sub(r"\s+", " ", value).strip(" .,:;-")
    return value if len(value) >= 3 else ""


def _extract_limit(normalized: str) -> int:
    match = re.search(r"\b(\d{1,2})\s+(?:задач|заяв)", normalized)
    if not match:
        return 5
    return max(1, min(int(match.group(1)), 10))


def _format_task_line(task: dict[str, Any], *, index: int | None = None) -> str:
    prefix = f"{index}. " if index is not None else ""
    task_id = _first(task, "id", "ID")
    title = str(_first(task, "title", "TITLE") or "Без названия")
    status = format_task_status(_first(task, "status", "STATUS"))
    responsible_id = _first(task, "responsibleId", "responsible_id", "RESPONSIBLE_ID")
    deadline = format_deadline_value(_first(task, "deadline", "DEADLINE"))
    parts = [f"{prefix}{format_task_link(task_id, title)}", status]
    if responsible_id not in (None, ""):
        parts.append(f"ответственный #{responsible_id}")
    parts.append(f"срок: {deadline or 'без срока'}")
    return " - ".join(parts)


def format_task_link(task_id: object, title: str) -> str:
    if task_id in (None, ""):
        return title
    return f"[#{task_id}: {title}]({_task_url(task_id)})"


def format_task_status(value: object) -> str:
    return TASK_STATUS_LABELS.get(str(value), str(value or "статус неизвестен"))


def format_deadline_value(value: object) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(MOSCOW_TZ)
        return parsed.strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return text.replace("T", " ").split("+", 1)[0]


def _task_url(task_id: object) -> str:
    domain = get_settings().bitrix_domain.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if not domain:
        return f"/company/personal/user/0/tasks/task/view/{task_id}/"
    return f"https://{domain}/company/personal/user/0/tasks/task/view/{task_id}/"


def _first(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _optional_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize(text: str) -> str:
    return text.casefold().replace("ё", "е")
