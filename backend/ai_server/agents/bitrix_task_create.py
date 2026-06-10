from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_server.models import AgentTask
from ai_server.utils import compact_text, optional_int, unique


@dataclass(frozen=True)
class BitrixTaskCreateDraft:
    method: str = "tasks.task.add"
    params: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    contract_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    responsible_query: str = ""
    project_query: str = ""

    @property
    def is_ready(self) -> bool:
        return not self.contract_errors and bool(self.params.get("fields"))

    def as_action_details(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "summary": self.summary,
            "contract_errors": self.contract_errors,
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
    title = compact_text(str(args.get("title") or "")).strip(" .,:;-")
    description = compact_text(str(args.get("description") or ""))
    responsible_id = optional_int(args.get("responsible_id"))
    responsible_query = compact_text(str(args.get("responsible_query") or ""))
    responsible_note = ""
    group_id = optional_int(args.get("group_id"))
    project_query = compact_text(str(args.get("project_query") or ""))
    group_note = ""

    if responsible_id is None and _truthy(args.get("responsible_self")):
        responsible_id = optional_int(task.user.id)
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

    deadline, deadline_note, no_deadline, deadline_error = _deadline_from_args(args)

    contract_errors: list[str] = []
    notes = [note for note in (responsible_note, group_note, deadline_note, *resolution.notes) if note]
    fields: dict[str, Any] = {}
    current_user_id = optional_int(task.user.id)
    if title:
        fields["TITLE"] = title
        if description:
            fields["DESCRIPTION"] = description
    else:
        contract_errors.append("task_create_draft.title is required")

    if responsible_id is not None:
        fields["RESPONSIBLE_ID"] = responsible_id
    else:
        if responsible_query:
            contract_errors.append("task_create_draft.responsible_query was not resolved to a unique Bitrix user")
        elif _truthy(args.get("responsible_self")):
            contract_errors.append("task_create_draft.responsible_self requires channel user id")
        else:
            contract_errors.append(
                "task_create_draft requires one of responsible_id, responsible_query, or responsible_self"
            )

    if project_query and group_id is None:
        contract_errors.append("task_create_draft.project_query was not resolved to a unique Bitrix project")
    if deadline_error:
        contract_errors.append(deadline_error)

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
        contract_errors=unique(contract_errors),
        notes=notes,
        responsible_query=responsible_query,
        project_query=project_query,
    )


def _deadline_from_args(args: dict[str, Any]) -> tuple[str | None, str, bool, str]:
    if _truthy(args.get("no_deadline")):
        return None, "LLM explicitly selected no_deadline.", True, ""
    raw_deadline = str(args.get("deadline") or args.get("deadline_iso") or "").strip()
    if not raw_deadline:
        return None, "", False, "task_create_draft requires deadline_iso or no_deadline=true"
    try:
        datetime.fromisoformat(raw_deadline.replace("Z", "+00:00"))
    except ValueError:
        return None, "", False, "task_create_draft.deadline_iso must be ISO 8601"
    return raw_deadline, "LLM provided deadline_iso.", False, ""


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


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "да", "on"}
    return bool(value)
