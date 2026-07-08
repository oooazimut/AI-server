from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ai_server.agents.bitrix24.tools.bitrix_api import (
    _extract_sonet_groups,
    _match_sonet_groups,
)
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_ports import BitrixToolClientPort

DEFAULT_MY_TASKS_LIMIT = 10
DEFAULT_TASK_SEARCH_LIMIT = 10
DEFAULT_PROJECT_SEARCH_LIMIT = 10
MAX_MY_TASKS_LIMIT = 50
MAX_TASK_SEARCH_LIMIT = 50
MAX_PROJECT_SEARCH_LIMIT = 20
ACTIVE_STATUS_VALUES = [1, 2, 3, 4]
TASK_SELECT = [
    "ID",
    "TITLE",
    "DESCRIPTION",
    "STATUS",
    "RESPONSIBLE_ID",
    "CREATED_BY",
    "DEADLINE",
    "GROUP_ID",
    "ACCOMPLICES",
    "AUDITORS",
]


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
                "and sorts by deadline ascending. Open means active Bitrix statuses 1, 2, 3, 4."
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
        calls = [
            {"role_source": "member", "filter": {**_status_filter(status), "MEMBER": user_id}},
            {"role_source": "created_by", "filter": {**_status_filter(status), "CREATED_BY": user_id}},
        ]

        sorted_tasks, errors = await _fetch_merged_tasks(self._client, calls)
        if errors and not sorted_tasks:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error="Bitrix task lookup failed.",
                data={"status": status, "errors": errors},
            )

        page = sorted_tasks[offset : offset + limit]
        items = [_task_summary(task, user_id=user_id) for task in page]
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "source": "live_bitrix_rest",
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


class BitrixTaskSearchTool:
    name = "bitrix_task_search"

    def __init__(self, client: BitrixToolClientPort | None = None) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Deterministic read-only Bitrix task search. Use instead of generic bitrix_api for common task "
                "reads: tasks where I am responsible, tasks created by me, task by ID/title, overdue tasks, and "
                "tasks in a project. Defaults to active Bitrix statuses 1, 2, 3, 4 and limit 10. The response is "
                "normalized so future PostgreSQL snapshot/index candidates can be plugged in without changing "
                "the user-facing format."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["my", "responsible", "created_by", "member", "all"],
                        "description": "Current-user role filter. Default: my.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "open", "closed", "overdue", "deferred", "declined", "all"],
                        "description": "Task status filter. Default: active/open.",
                    },
                    "include_closed": {
                        "type": "boolean",
                        "description": "Set true only when the user explicitly asks to include closed/deferred/declined tasks.",
                    },
                    "task_id": {"type": "integer", "description": "Exact task ID lookup."},
                    "query": {
                        "type": "string",
                        "description": "Text to match in task title or description after Bitrix returns candidates.",
                    },
                    "project_id": {"type": "integer", "description": "Bitrix workgroup/project ID."},
                    "project_name": {
                        "type": "string",
                        "description": "Project/workgroup name. The tool normalizes hyphens, case, and known aliases.",
                    },
                    "deadline_from": {
                        "type": "string",
                        "description": "Optional ISO date/datetime lower bound for task deadline.",
                    },
                    "deadline_to": {
                        "type": "string",
                        "description": "Optional ISO date/datetime upper bound for task deadline.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TASK_SEARCH_LIMIT,
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

        task_id = _safe_int(args.get("task_id") or args.get("id") or args.get("ID"))
        if task_id is not None:
            return await self._execute_task_detail(task_id, user_id=user_id)

        scope = _scope_arg(args.get("scope"))
        if scope in {"my", "responsible", "created_by", "member"} and user_id is None:
            return ToolResult(
                status=ToolStatus.DENIED,
                tool=self.name,
                error="Bitrix task search denied: current Bitrix user_id is missing.",
            )

        limit = _bounded_int(
            args.get("limit"), default=DEFAULT_TASK_SEARCH_LIMIT, minimum=1, maximum=MAX_TASK_SEARCH_LIMIT
        )
        offset = _bounded_int(args.get("offset"), default=0, minimum=0, maximum=10_000)
        status = _task_search_status_arg(args.get("status"))
        if status == "all" and not _truthy(args.get("include_closed")):
            status = "active"
        query = _first_arg_text(args, "query", "text", "title")
        project_id = _safe_int(args.get("project_id") or args.get("group_id") or args.get("GROUP_ID"))
        project_query = _first_arg_text(args, "project_name", "project", "group_name")
        project: dict[str, Any] | None = None
        errors: list[dict[str, str]] = []

        if project_id is None and project_query:
            project_matches, project_errors = await _search_projects(self._client, project_query, limit=1)
            errors.extend(project_errors)
            if not project_matches:
                return ToolResult(
                    status=ToolStatus.OK,
                    tool=self.name,
                    data={
                        "mode": "list",
                        "source": "live_bitrix_rest",
                        "scope": scope,
                        "status": status,
                        "query": query,
                        "project_query": project_query,
                        "project_not_found": True,
                        "items": [],
                        "total": 0,
                        "limit": limit,
                        "offset": offset,
                        "has_more": False,
                        "errors": errors,
                    },
                )
            project = _project_summary(project_matches[0])
            project_id = _safe_int(project.get("id"))
        elif project_id is not None:
            project = {"id": str(project_id), "name": project_query or ""}

        calls = _task_search_calls(
            scope=scope,
            status=status,
            user_id=user_id,
            project_id=project_id,
            query=query,
        )
        sorted_tasks, task_errors = await _fetch_merged_tasks(self._client, calls)
        errors.extend(task_errors)
        if errors and not sorted_tasks:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error="Bitrix task search failed.",
                data={"scope": scope, "status": status, "errors": errors},
            )

        filtered = [
            task
            for task in sorted_tasks
            if _matches_text_query(task, query)
            and _matches_task_status(task, status)
            and _matches_deadline_range(
                task,
                from_value=_first_arg_text(args, "deadline_from", "deadline_start"),
                to_value=_first_arg_text(args, "deadline_to", "deadline_end"),
            )
            and (status != "overdue" or _is_overdue(task))
        ]
        page = filtered[offset : offset + limit]
        items = [_task_summary(task, user_id=user_id or 0) for task in page]
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "mode": "list",
                "source": "live_bitrix_rest",
                "scope": scope,
                "scope_label": _scope_label(scope),
                "status": status,
                "query": query,
                "project": project,
                "project_query": project_query,
                "items": items,
                "total": len(filtered),
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < len(filtered),
                "errors": errors,
            },
        )

    async def _execute_task_detail(self, task_id: int, *, user_id: int | None) -> ToolResult:
        try:
            result = await self._client.result("tasks.task.get", {"taskId": task_id, "select": TASK_SELECT})
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=str(exc),
                data={"mode": "detail", "task_id": task_id},
            )
        task = _extract_task(result)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "mode": "detail",
                "source": "live_bitrix_rest",
                "task_id": str(task_id),
                "item": _task_summary(task, user_id=user_id or 0) if task else None,
            },
        )


class BitrixProjectSearchTool:
    name = "bitrix_project_search"

    def __init__(self, client: BitrixToolClientPort | None = None) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Deterministic read-only Bitrix project/workgroup search by name. Use instead of generic bitrix_api "
                "for project reads like 'найди проект Ларгус 2'. The tool normalizes hyphens, case, and known car "
                "project aliases before returning project links."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Project/workgroup name to search."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_PROJECT_SEARCH_LIMIT,
                        "description": "Maximum projects to return. Default: 10.",
                    },
                },
                "required": ["query"],
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
        query = _first_arg_text(args, "query", "project", "name")
        if not query:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error="Project query is required.",
                data={"query": ""},
            )
        limit = _bounded_int(
            args.get("limit"),
            default=DEFAULT_PROJECT_SEARCH_LIMIT,
            minimum=1,
            maximum=MAX_PROJECT_SEARCH_LIMIT,
        )
        matches, errors = await _search_projects(self._client, query, limit=limit)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "source": "live_bitrix_rest",
                "query": query,
                "items": [_project_summary(group) for group in matches],
                "total": len(matches),
                "limit": limit,
                "errors": errors,
            },
        )


def _status_arg(value: object) -> str:
    text = str(value or "open").strip().casefold()
    return text if text in {"open", "closed", "all"} else "open"


def _task_search_status_arg(value: object) -> str:
    text = str(value or "active").strip().casefold()
    if text == "open":
        return "active"
    return text if text in {"active", "closed", "overdue", "deferred", "declined", "all"} else "active"


def _scope_arg(value: object) -> str:
    text = str(value or "my").strip().casefold()
    return text if text in {"my", "responsible", "created_by", "member", "all"} else "my"


def _status_filter(status: str) -> dict[str, Any]:
    if status == "closed":
        return {"STATUS": 5}
    if status == "all":
        return {}
    return {"STATUS": ACTIVE_STATUS_VALUES}


def _task_search_status_filter(status: str) -> dict[str, Any]:
    if status == "closed":
        return {"STATUS": 5}
    if status == "deferred":
        return {"STATUS": 6}
    if status == "declined":
        return {"STATUS": 7}
    if status == "all":
        return {}
    return {"STATUS": ACTIVE_STATUS_VALUES}


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


async def _fetch_merged_tasks(
    client: BitrixToolClientPort,
    calls: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    merged: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for call in calls:
        params = {
            "filter": call["filter"],
            "select": TASK_SELECT,
            "order": {"DEADLINE": "ASC", "ID": "ASC"},
        }
        try:
            result = await client.result("tasks.task.list", params)
        except (BitrixApiError, BitrixConfigError) as exc:
            errors.append({"source": str(call.get("role_source") or "tasks"), "error": str(exc)})
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
    return (
        sorted(
            merged.values(),
            key=lambda task: (_deadline_sort_key(task), _safe_int(_task_id(task)) or 0),
        ),
        errors,
    )


def _task_search_calls(
    *,
    scope: str,
    status: str,
    user_id: int | None,
    project_id: int | None,
    query: str = "",
) -> list[dict[str, Any]]:
    base_filter: dict[str, Any] = {} if query else dict(_task_search_status_filter(status))
    if project_id is not None:
        base_filter["GROUP_ID"] = project_id
    if query:
        base_filter["%TITLE"] = query
    if scope == "responsible":
        return [{"role_source": "responsible", "filter": {**base_filter, "RESPONSIBLE_ID": user_id}}]
    if scope == "created_by":
        return [{"role_source": "created_by", "filter": {**base_filter, "CREATED_BY": user_id}}]
    if scope == "member":
        return [{"role_source": "member", "filter": {**base_filter, "MEMBER": user_id}}]
    if scope == "all":
        return [{"role_source": "all", "filter": base_filter}]
    return [
        {"role_source": "member", "filter": {**base_filter, "MEMBER": user_id}},
        {"role_source": "created_by", "filter": {**base_filter, "CREATED_BY": user_id}},
    ]


async def _search_projects(
    client: BitrixToolClientPort,
    query: str,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    initial_params = {"FILTER": {"%NAME": query}, "ORDER": {"NAME": "ASC"}}
    try:
        initial_result = await client.result("sonet_group.get", initial_params)
    except (BitrixApiError, BitrixConfigError) as exc:
        errors.append({"source": "sonet_group.get", "error": str(exc)})
        initial_result = []

    initial_groups = _extract_sonet_groups(initial_result)
    matches = _match_sonet_groups(initial_groups, query=query, limit=limit)
    if matches:
        return matches, errors

    fallback_params = {"FILTER": {}, "ORDER": {"NAME": "ASC"}}
    try:
        fallback_result = await client.result("sonet_group.get", fallback_params)
    except (BitrixApiError, BitrixConfigError) as exc:
        errors.append({"source": "sonet_group.get:fallback", "error": str(exc)})
        return [], errors
    return _match_sonet_groups(_extract_sonet_groups(fallback_result), query=query, limit=limit), errors


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


def _extract_task(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        for key in ("task", "TASK"):
            task = result.get(key)
            if isinstance(task, dict):
                return task
        nested = _extract_tasks(result)
        if nested:
            return nested[0]
        result_value = result.get("result")
        if isinstance(result_value, dict):
            return _extract_task(result_value)
    return {}


def _task_summary(task: dict[str, Any], *, user_id: int) -> dict[str, Any]:
    task_id = _task_id(task)
    title = _first_text(task, "title", "TITLE") or "задача"
    status = _first_text(task, "status", "STATUS")
    deadline = _first_text(task, "deadline", "DEADLINE")
    group_id = _first_text(task, "groupId", "GROUP_ID")
    return {
        "id": task_id,
        "title": title,
        "description": _first_text(task, "description", "DESCRIPTION"),
        "status": status,
        "status_label": _status_label(status),
        "deadline": deadline,
        "deadline_label": _deadline_label(deadline),
        "group_id": group_id,
        "roles": _task_roles(task, user_id=user_id) if user_id else [],
    }


def _project_summary(group: dict[str, Any]) -> dict[str, Any]:
    group_id = _first_text(group, "ID", "id")
    name = _first_text(group, "NAME", "name", "TITLE", "title") or "проект"
    return {
        "id": group_id,
        "name": name,
        "description": _first_text(group, "DESCRIPTION", "description"),
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
        value = list(value.values())
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


def _first_arg_text(args: dict[str, Any], *keys: str) -> str:
    return _first_text(args, *keys)


def _first_text(task: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = task.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _matches_text_query(task: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    needle = query.casefold().strip()
    haystack = " ".join(
        [
            _first_text(task, "title", "TITLE"),
            _first_text(task, "description", "DESCRIPTION"),
        ]
    ).casefold()
    return needle in haystack


def _matches_task_status(task: dict[str, Any], status: str) -> bool:
    if status == "all":
        return True
    task_status = _safe_int(_first_text(task, "status", "STATUS"))
    if task_status is None:
        return False
    if status == "closed":
        return task_status == 5
    if status == "deferred":
        return task_status == 6
    if status == "declined":
        return task_status == 7
    return task_status in ACTIVE_STATUS_VALUES


def _matches_deadline_range(task: dict[str, Any], *, from_value: str, to_value: str) -> bool:
    if not from_value and not to_value:
        return True
    deadline = _parse_datetime(_first_text(task, "deadline", "DEADLINE"))
    if deadline is None:
        return False
    from_dt = _parse_datetime(from_value)
    to_dt = _parse_datetime(to_value)
    if from_dt is not None and deadline < from_dt:
        return False
    return not (to_dt is not None and deadline > to_dt)


def _is_overdue(task: dict[str, Any]) -> bool:
    deadline = _parse_datetime(_first_text(task, "deadline", "DEADLINE"))
    if deadline is None:
        return False
    return deadline < datetime.now(UTC)


def _deadline_sort_key(task: dict[str, Any]) -> tuple[int, str]:
    deadline = _first_text(task, "deadline", "DEADLINE")
    if not deadline:
        return (1, "")
    parsed = _parse_datetime(deadline)
    if parsed is None:
        return (0, deadline)
    return (0, parsed.astimezone(UTC).isoformat())


def _parse_datetime(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _deadline_label(deadline: str) -> str:
    if not deadline:
        return "без срока"
    parsed = _parse_datetime(deadline)
    if parsed is None:
        return deadline
    return parsed.astimezone().strftime("%d.%m.%Y %H:%M")


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


def _scope_label(scope: str) -> str:
    return {
        "my": "мои задачи",
        "responsible": "задачи на мне",
        "created_by": "задачи, поставленные мной",
        "member": "задачи с моим участием",
        "all": "задачи",
    }.get(scope, "задачи")


def _safe_int(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "да", "on"}
    return bool(value)


__all__ = ["BitrixMyTasksTool", "BitrixProjectSearchTool", "BitrixTaskSearchTool"]
