from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_ports import BitrixToolClientPort

DEFAULT_MY_TASKS_LIMIT = 10
MAX_MY_TASKS_LIMIT = 50


class BitrixMyTasksTool:
    name = "bitrix_my_tasks"

    def __init__(self, client: BitrixToolClientPort | None = None) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Read-only lookup for the current user's Bitrix tasks. Use for generic requests like "
                "'мои задачи' or 'мои открытые задачи'. It includes tasks where the current user is a member "
                "(responsible/accomplice/auditor in Bitrix) and tasks created by the current user, then deduplicates "
                "and sorts by deadline ascending."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Task status filter. Default: open.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_MY_TASKS_LIMIT,
                        "description": "Maximum tasks to return. Default: 10.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Offset after sorting, for the next page.",
                    },
                },
            },
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if self._client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="BitrixClient is not injected")
        if user_id is None:
            return ToolResult(
                status=ToolStatus.DENIED,
                tool=self.name,
                error="Bitrix my-tasks lookup denied: current Bitrix user_id is missing.",
            )

        status = _status_arg(args.get("status"))
        limit = _bounded_int(args.get("limit"), default=DEFAULT_MY_TASKS_LIMIT, minimum=1, maximum=MAX_MY_TASKS_LIMIT)
        offset = _bounded_int(args.get("offset"), default=0, minimum=0, maximum=10_000)
        select = [
            "ID",
            "TITLE",
            "STATUS",
            "RESPONSIBLE_ID",
            "CREATED_BY",
            "DEADLINE",
            "GROUP_ID",
            "ACCOMPLICES",
            "AUDITORS",
        ]
        order = {"DEADLINE": "ASC", "ID": "ASC"}
        calls = [
            {"role_source": "member", "filter": {**_status_filter(status), "MEMBER": user_id}},
            {"role_source": "created_by", "filter": {**_status_filter(status), "CREATED_BY": user_id}},
        ]

        merged: dict[str, dict[str, Any]] = {}
        errors: list[dict[str, str]] = []
        for call in calls:
            params = {"filter": call["filter"], "select": select, "order": order}
            try:
                result = await self._client.result("tasks.task.list", params)
            except (BitrixApiError, BitrixConfigError) as exc:
                errors.append({"source": str(call["role_source"]), "error": str(exc)})
                continue
            for task in _extract_tasks(result):
                task_id = _task_id(task)
                if not task_id:
                    continue
                existing = merged.get(task_id)
                if existing is None:
                    merged[task_id] = dict(task)
                else:
                    existing.update({key: value for key, value in task.items() if value not in (None, "", [])})

        if errors and not merged:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error="Bitrix task lookup failed.",
                data={"status": status, "errors": errors},
            )

        sorted_tasks = sorted(
            merged.values(),
            key=lambda task: (_deadline_sort_key(task), _safe_int(_task_id(task)) or 0),
        )
        page = sorted_tasks[offset : offset + limit]
        items = [_task_summary(task, user_id=user_id) for task in page]
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "status": status,
                "user_id": user_id,
                "items": items,
                "total": len(sorted_tasks),
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < len(sorted_tasks),
                "errors": errors,
                "sources": [call["role_source"] for call in calls],
            },
        )


def _status_arg(value: object) -> str:
    text = str(value or "open").strip().casefold()
    return text if text in {"open", "closed", "all"} else "open"


def _status_filter(status: str) -> dict[str, Any]:
    if status == "closed":
        return {"STATUS": 5}
    if status == "all":
        return {}
    return {"!STATUS": 5}


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _extract_tasks(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ("tasks", "TASKS", "items", "result"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _extract_tasks(value)
                if nested:
                    return nested
    return []


def _task_summary(task: dict[str, Any], *, user_id: int) -> dict[str, Any]:
    task_id = _task_id(task)
    title = _first_text(task, "title", "TITLE") or "задача"
    status = _first_text(task, "status", "STATUS")
    deadline = _first_text(task, "deadline", "DEADLINE")
    return {
        "id": task_id,
        "title": title,
        "status": status,
        "status_label": _status_label(status),
        "deadline": deadline,
        "deadline_label": _deadline_label(deadline),
        "roles": _task_roles(task, user_id=user_id),
    }


def _task_roles(task: dict[str, Any], *, user_id: int) -> list[str]:
    roles: list[str] = []
    if _safe_int(_first_text(task, "responsibleId", "RESPONSIBLE_ID")) == user_id:
        roles.append("исполнитель")
    if _safe_int(_first_text(task, "createdBy", "CREATED_BY")) == user_id:
        roles.append("постановщик")
    if user_id in _int_list(task.get("accomplices") or task.get("ACCOMPLICES")):
        roles.append("соисполнитель")
    if user_id in _int_list(task.get("auditors") or task.get("AUDITORS")):
        roles.append("наблюдатель")
    return roles or ["участник"]


def _int_list(value: object) -> list[int]:
    if isinstance(value, dict):
        value = list(value)
    if not isinstance(value, list | tuple | set):
        return []
    result = []
    for item in value:
        parsed = _safe_int(item)
        if parsed is not None:
            result.append(parsed)
    return result


def _task_id(task: dict[str, Any]) -> str:
    return _first_text(task, "id", "ID")


def _first_text(task: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = task.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _deadline_sort_key(task: dict[str, Any]) -> tuple[int, str]:
    deadline = _first_text(task, "deadline", "DEADLINE")
    if not deadline:
        return (1, "")
    try:
        parsed = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    except ValueError:
        return (0, deadline)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return (0, parsed.astimezone(UTC).isoformat())


def _deadline_label(deadline: str) -> str:
    if not deadline:
        return "без срока"
    try:
        parsed = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    except ValueError:
        return deadline
    return parsed.strftime("%d.%m.%Y %H:%M")


def _status_label(status: str) -> str:
    return {
        "1": "новая",
        "2": "ждёт выполнения",
        "3": "выполняется",
        "4": "ждёт контроля",
        "5": "завершена",
        "6": "отложена",
        "7": "отклонена",
    }.get(str(status), str(status or ""))


def _safe_int(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


__all__ = ["BitrixMyTasksTool"]
