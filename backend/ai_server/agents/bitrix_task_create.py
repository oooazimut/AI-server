from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from ai_server.models import AgentTask


TASK_CREATE_MARKERS = (
    "создай задачу",
    "создать задачу",
    "поставь задачу",
    "сделай задачу",
    "создай заявку",
    "создать заявку",
)


@dataclass(frozen=True)
class BitrixTaskCreateDraft:
    method: str = "tasks.task.add"
    params: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    missing_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    responsible_query: str = ""
    project_query: str = ""

    @property
    def is_ready(self) -> bool:
        return not self.missing_fields and bool(self.params.get("fields"))

    def as_action_details(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "summary": self.summary,
            "missing_fields": self.missing_fields,
            "notes": self.notes,
            "responsible_query": self.responsible_query,
            "project_query": self.project_query,
        }


@dataclass(frozen=True)
class BitrixTaskCreateResolution:
    responsible_id: int | None = None
    responsible_note: str = ""
    group_id: int | None = None
    group_note: str = ""
    notes: list[str] = field(default_factory=list)


def looks_like_task_create_request(text: str) -> bool:
    normalized = _normalize(text)
    return any(marker in normalized for marker in TASK_CREATE_MARKERS)


def build_task_create_draft(
    task: AgentTask,
    *,
    now: datetime | None = None,
    resolution: BitrixTaskCreateResolution | None = None,
) -> BitrixTaskCreateDraft:
    now = now or datetime.now().astimezone()
    resolution = resolution or BitrixTaskCreateResolution()
    request = task.request.strip()
    cleaned = _strip_create_prefix(request)
    responsible_id, responsible_note, responsible_query = _extract_responsible(cleaned, task)
    group_id, group_note, project_query = _extract_group(cleaned)
    if responsible_id is None and resolution.responsible_id is not None:
        responsible_id = resolution.responsible_id
        responsible_note = resolution.responsible_note or f"Ответственный найден по запросу `{responsible_query}`."
    if group_id is None and resolution.group_id is not None:
        group_id = resolution.group_id
        group_note = resolution.group_note or f"Проект найден по запросу `{project_query}`."
    deadline, deadline_note, no_deadline = _extract_deadline(cleaned, now=now)
    title = _extract_title(cleaned, responsible_query=responsible_query, project_query=project_query)

    missing_fields: list[str] = []
    notes = [note for note in (responsible_note, group_note, deadline_note, *resolution.notes) if note]
    fields: dict[str, Any] = {}
    current_user_id = _optional_int(task.user.id)
    if title:
        fields["TITLE"] = title
        fields["DESCRIPTION"] = cleaned
    else:
        missing_fields.append("title")

    if responsible_id is not None:
        fields["RESPONSIBLE_ID"] = responsible_id
    else:
        missing_fields.append("responsible_id")

    if project_query and group_id is None:
        missing_fields.append("group_id")

    if current_user_id is not None:
        fields["CREATED_BY"] = current_user_id
    if group_id is not None:
        fields["GROUP_ID"] = group_id
    if deadline:
        fields["DEADLINE"] = deadline
    elif no_deadline:
        fields["NO_DEADLINE"] = True

    params = {"fields": fields} if fields else {}
    summary = _summary(title=title, responsible_id=responsible_id, deadline=deadline, group_id=group_id)
    return BitrixTaskCreateDraft(
        params=params,
        summary=summary,
        missing_fields=missing_fields,
        notes=notes,
        responsible_query=responsible_query,
        project_query=project_query,
    )


def format_task_create_clarification(draft: BitrixTaskCreateDraft) -> str:
    labels = {
        "title": "название задачи",
        "responsible_id": "ответственный",
        "group_id": "проект/группа",
    }
    missing = ", ".join(labels.get(field, field) for field in draft.missing_fields)
    parts = [f"Не хватает данных, чтобы подготовить задачу в Bitrix24: {missing}."]
    parts.append("Можно указать ответственного как `на меня`, `ответственный #123` или уточнить ФИО точнее.")
    if draft.notes:
        parts.append("Примечания: " + " ".join(draft.notes))
    return " ".join(parts)


def _strip_create_prefix(text: str) -> str:
    value = text.strip()
    patterns = (
        r"^\s*создай\s+(?:мне\s+)?(?:в\s+битриксе\s+)?(?:задачу|заявку)\s*",
        r"^\s*создать\s+(?:в\s+битриксе\s+)?(?:задачу|заявку)\s*",
        r"^\s*поставь\s+(?:мне\s+)?(?:задачу|заявку)\s*",
        r"^\s*сделай\s+(?:мне\s+)?(?:задачу|заявку)\s*",
    )
    for pattern in patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    return _compact(value)


def _extract_responsible(text: str, task: AgentTask) -> tuple[int | None, str, str]:
    normalized = _normalize(text)
    if any(marker in normalized for marker in ("на меня", "мне", "себе", "я сам", "на себя")):
        current_user_id = _optional_int(task.user.id)
        if current_user_id is not None:
            return current_user_id, "Ответственным выбран текущий пользователь.", ""
        return None, "Пользователь не определён в канале, поэтому `на меня` нельзя превратить в Bitrix user id.", ""

    match = re.search(
        r"(?:ответственн\w*|исполнител\w*|на|для)\s*(?:пользовател[яю]\s*)?#?(\d{1,10})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1)), "Ответственный взят из явного Bitrix user id.", ""

    hash_ids = re.findall(r"#(\d{1,10})\b", text)
    if len(hash_ids) == 1:
        return int(hash_ids[0]), "Ответственный взят из единственного `#ID` в запросе.", ""

    query = _extract_responsible_query(text)
    if query:
        return None, f"Ответственного нужно найти в Bitrix по запросу `{query}`.", query
    return None, "", ""


def _extract_group(text: str) -> tuple[int | None, str, str]:
    match = re.search(
        r"(?:проект(?:е|у)?|групп(?:е|у)?)\s*#?(\d{1,10})\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return int(match.group(1)), "Проект/группа взят из явного Bitrix group id.", ""
    query = _extract_project_query(text)
    if query:
        return None, f"Проект нужно найти в Bitrix по запросу `{query}`.", query
    return None, "", ""


def _extract_deadline(text: str, *, now: datetime) -> tuple[str | None, str, bool]:
    normalized = _normalize(text)
    if any(marker in normalized for marker in ("без срока", "бессрочно", "срок не нужен")):
        return None, "Пользователь явно попросил задачу без срока.", True

    date_match = re.search(r"(?:до|к)\s+(\d{1,2})[.\/-](\d{1,2})(?:[.\/-](\d{2,4}))?", text, flags=re.IGNORECASE)
    if date_match:
        day = int(date_match.group(1))
        month = int(date_match.group(2))
        raw_year = date_match.group(3)
        year = int(raw_year) if raw_year else now.year
        if year < 100:
            year += 2000
        try:
            deadline = datetime(year, month, day, 19, 0, tzinfo=_timezone(now))
            return deadline.isoformat(), "Срок взят из явной даты.", False
        except ValueError:
            return None, "Не смог распознать дату срока.", False

    relative_days = {
        "сегодня": 0,
        "завтра": 1,
        "послезавтра": 2,
    }
    for marker, days in relative_days.items():
        if marker in normalized:
            deadline = (now + timedelta(days=days)).replace(hour=19, minute=0, second=0, microsecond=0)
            return deadline.isoformat(), f"Срок распознан как `{marker}`.", False

    weekdays = {
        "понедельник": 0,
        "вторник": 1,
        "среду": 2,
        "среда": 2,
        "четверг": 3,
        "пятницу": 4,
        "пятница": 4,
        "субботу": 5,
        "суббота": 5,
        "воскресенье": 6,
    }
    for marker, weekday in weekdays.items():
        if re.search(rf"(?:до|к)\s+{marker}\b", normalized):
            days_ahead = (weekday - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            deadline = (now + timedelta(days=days_ahead)).replace(hour=19, minute=0, second=0, microsecond=0)
            return deadline.isoformat(), f"Срок распознан как ближайшая `{marker}`.", False

    default_deadline = (now + timedelta(days=3)).replace(hour=19, minute=0, second=0, microsecond=0)
    return default_deadline.isoformat(), "Срок не указан; поставлен дефолт MVP: через 3 дня в 19:00.", False


def _extract_title(text: str, *, responsible_query: str = "", project_query: str = "") -> str:
    value = text
    value = re.sub(r"\b(?:ответственн\w*|исполнител\w*)\s*(?:пользовател[яю]\s*)?#?\d{1,10}\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:на|для)\s*#?\d{1,10}\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:на\s+меня|мне|себе|я\s+сам|на\s+себя)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:в\s+)?(?:проекте|проект|группе|группа)\s*#?\d{1,10}\b", "", value, flags=re.IGNORECASE)
    if responsible_query:
        value = _remove_responsible_query(value, responsible_query)
    if project_query:
        value = _remove_project_query(value, project_query)
    value = re.sub(r"\b(?:до|к)\s+\d{1,2}[.\/-]\d{1,2}(?:[.\/-]\d{2,4})?\b", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\b(?:сегодня|завтра|послезавтра|без\s+срока|бессрочно|срок\s+не\s+нужен)\b", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\b(?:до|к)\s+(?:понедельник|вторник|сред[ау]|четверг|пятниц[ау]|суббот[ау]|воскресенье)\b",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\b(?:в\s+битриксе|битрикс24|битрикс)\b", "", value, flags=re.IGNORECASE)
    return _compact(value).strip(" .,:;-")


def _extract_responsible_query(text: str) -> str:
    explicit = re.search(
        r"(?i:(?:ответственн\w*|исполнител\w*|для|на))\s+(?P<name>[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z-]+(?:\s+[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z-]+){0,2})",
        text,
    )
    if explicit:
        return _compact(explicit.group("name")).strip(" .,:;-")

    first_word = re.match(r"\s*(?P<name>[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z-]{2,})\b", text)
    if first_word:
        return _compact(first_word.group("name")).strip(" .,:;-")
    return ""


def _extract_project_query(text: str) -> str:
    match = re.search(
        r"(?:в\s+)?(?:проекте|проект|группе|группа)\s+(?P<query>(?!#)[^,.;]+?)(?=\s+(?:до|к|сегодня|завтра|послезавтра|ответственн|исполнител|для|на\s+#)|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return _compact(match.group("query")).strip(" .,:;-")


def _remove_responsible_query(text: str, query: str) -> str:
    escaped = re.escape(query)
    value = re.sub(rf"\b(?:ответственн\w*|исполнител\w*|для|на)\s+{escaped}\b", "", text, flags=re.IGNORECASE)
    value = re.sub(rf"^\s*{escaped}\b", "", value, flags=re.IGNORECASE)
    return value


def _remove_project_query(text: str, query: str) -> str:
    escaped = re.escape(query)
    return re.sub(rf"\b(?:в\s+)?(?:проекте|проект|группе|группа)\s+{escaped}\b", "", text, flags=re.IGNORECASE)


def _summary(*, title: str, responsible_id: int | None, deadline: str | None, group_id: int | None) -> str:
    title_part = title or "задача без названия"
    parts = [f"создать задачу `{title_part}`"]
    if responsible_id is not None:
        parts.append(f"ответственный #{responsible_id}")
    if group_id is not None:
        parts.append(f"проект/группа #{group_id}")
    if deadline:
        parts.append(f"срок {deadline}")
    return ", ".join(parts)


def _optional_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize(text: str) -> str:
    return _compact(text.casefold().replace("ё", "е"))


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _timezone(value: datetime) -> timezone:
    if value.tzinfo is None:
        return timezone.utc
    offset = value.utcoffset()
    return timezone(offset or timedelta())
