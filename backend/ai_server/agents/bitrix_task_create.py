from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_server.models import AgentTask


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


def build_task_create_draft_from_args(
    task: AgentTask,
    args: dict[str, Any],
    *,
    resolution: BitrixTaskCreateResolution | None = None,
) -> BitrixTaskCreateDraft:
    resolution = resolution or BitrixTaskCreateResolution()
    args = args or {}
    title = _compact(str(args.get("title") or "")).strip(" .,:;-")
    description = _compact(str(args.get("description") or task.request.strip()))
    responsible_id = _optional_int(args.get("responsible_id"))
    responsible_query = _compact(str(args.get("responsible_query") or ""))
    responsible_note = ""
    group_id = _optional_int(args.get("group_id"))
    project_query = _compact(str(args.get("project_query") or ""))
    group_note = ""

    if responsible_id is None and _truthy(args.get("responsible_self")):
        responsible_id = _optional_int(task.user.id)
        if responsible_id is not None:
            responsible_note = "Ответственным выбран текущий пользователь."
        else:
            responsible_note = "LLM выбрала текущего пользователя, но канал не передал Bitrix user id."

    if responsible_id is None and resolution.responsible_id is not None:
        responsible_id = resolution.responsible_id
        responsible_note = resolution.responsible_note or f"Ответственный найден по запросу `{responsible_query}`."
    if group_id is None and resolution.group_id is not None:
        group_id = resolution.group_id
        group_note = resolution.group_note or f"Проект найден по запросу `{project_query}`."

    deadline, deadline_note, no_deadline, invalid_deadline = _deadline_from_args(args)

    missing_fields: list[str] = []
    notes = [note for note in (responsible_note, group_note, deadline_note, *resolution.notes) if note]
    fields: dict[str, Any] = {}
    current_user_id = _optional_int(task.user.id)
    if title:
        fields["TITLE"] = title
        fields["DESCRIPTION"] = description
    else:
        missing_fields.append("title")

    if responsible_id is not None:
        fields["RESPONSIBLE_ID"] = responsible_id
    else:
        missing_fields.append("responsible_id")
        if responsible_query:
            notes.append(f"Ответственного нужно найти в Bitrix по запросу `{responsible_query}`.")

    if project_query and group_id is None:
        missing_fields.append("group_id")
        notes.append(f"Проект нужно найти в Bitrix по запросу `{project_query}`.")
    if invalid_deadline:
        missing_fields.append("deadline")

    if current_user_id is not None:
        fields["CREATED_BY"] = current_user_id
    if group_id is not None:
        fields["GROUP_ID"] = group_id
    if deadline:
        fields["DEADLINE"] = deadline
    elif no_deadline:
        fields["NO_DEADLINE"] = True

    return BitrixTaskCreateDraft(
        params={"fields": fields} if fields else {},
        summary=_summary(title=title, responsible_id=responsible_id, deadline=deadline, group_id=group_id),
        missing_fields=_unique(missing_fields),
        notes=notes,
        responsible_query=responsible_query,
        project_query=project_query,
    )


def _deadline_from_args(args: dict[str, Any]) -> tuple[str | None, str, bool, bool]:
    if _truthy(args.get("no_deadline")):
        return None, "LLM распознала, что задача должна быть без срока.", True, False
    raw_deadline = str(args.get("deadline") or args.get("deadline_iso") or "").strip()
    if not raw_deadline:
        return None, "", False, False
    try:
        datetime.fromisoformat(raw_deadline.replace("Z", "+00:00"))
    except ValueError:
        return None, "LLM вернула некорректный ISO-срок.", False, True
    return raw_deadline, "Срок взят из LLM-структуры.", False, False


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


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "да", "on"}
    return bool(value)


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
