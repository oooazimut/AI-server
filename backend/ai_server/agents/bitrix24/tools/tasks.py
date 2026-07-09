from __future__ import annotations

import html
import re
from datetime import UTC, date, datetime, time
from typing import Any

from ai_server.agents.bitrix24.tools.bitrix_api import (
    _extract_sonet_groups,
    _match_sonet_groups,
    _normalize_project_name,
)
from ai_server.agents.bitrix24.tools.read_client import resolve_current_user_read_client
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_ports import BitrixToolClientPort
from ai_server.tools.bitrix_search import PortalSearchPort

DEFAULT_MY_TASKS_LIMIT = 10
DEFAULT_TASK_SEARCH_LIMIT = 10
DEFAULT_PROJECT_SEARCH_LIMIT = 10
MAX_MY_TASKS_LIMIT = 50
MAX_TASK_SEARCH_LIMIT = 50
MAX_PROJECT_SEARCH_LIMIT = 20
ACTIVE_STATUS_VALUES = [1, 2, 3, 4]
_BITRIX_PAIRED_TAG_RE = re.compile(
    r"\[(USER|URL|B|I|U|S|QUOTE|CODE|COLOR|SIZE)[^\]]*\](.*?)\[/\1\]", re.IGNORECASE | re.DOTALL
)
_BITRIX_SINGLE_TAG_RE = re.compile(r"\[/?[A-Z][A-Z0-9_]*(?:=[^\]]*)?\]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
TASK_SELECT = [
    "ID",
    "TITLE",
    "DESCRIPTION",
    "STATUS",
    "RESPONSIBLE_ID",
    "RESPONSIBLE",
    "CREATED_BY",
    "CREATOR",
    "CREATED_DATE",
    "CLOSED_DATE",
    "DEADLINE",
    "GROUP_ID",
    "ACCOMPLICES",
    "AUDITORS",
]
DEFAULT_COMMENT_LOOKUP_TASK_LIMIT = 50
MAX_COMMENT_LOOKUP_TASK_LIMIT = 200


class BitrixMyTasksTool:
    name = "bitrix_my_tasks"

    def __init__(
        self,
        client: BitrixToolClientPort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
    ) -> None:
        self._client = client
        self._bitrix_oauth = bitrix_oauth

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

        read_client, access_actor, access_error = await resolve_current_user_read_client(
            self.name,
            fallback_client=self._client,
            bitrix_oauth=self._bitrix_oauth,
            user_id=user_id,
        )
        if access_error is not None:
            return access_error

        sorted_tasks, errors = await _fetch_merged_tasks(read_client, calls)
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
                "access_actor": access_actor,
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

    def __init__(
        self,
        client: BitrixToolClientPort | None = None,
        portal_search: PortalSearchPort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
    ) -> None:
        self._client = client
        self._portal_search = portal_search
        self._bitrix_oauth = bitrix_oauth

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
                    "include_comments": {
                        "type": "boolean",
                        "description": "Also search the query text in task comments. Default: false.",
                    },
                    "comment_query": {
                        "type": "string",
                        "description": "Text that must be present in at least one task comment.",
                    },
                    "project_id": {"type": "integer", "description": "Bitrix workgroup/project ID."},
                    "project_name": {
                        "type": "string",
                        "description": "Project/workgroup name. The tool normalizes hyphens, case, and known aliases.",
                    },
                    "created_from": {
                        "type": "string",
                        "description": "Optional ISO date/datetime lower bound for task creation date.",
                    },
                    "created_to": {
                        "type": "string",
                        "description": "Optional ISO date/datetime upper bound for task creation date.",
                    },
                    "deadline_from": {
                        "type": "string",
                        "description": "Optional ISO date/datetime lower bound for task deadline.",
                    },
                    "deadline_to": {
                        "type": "string",
                        "description": "Optional ISO date/datetime upper bound for task deadline.",
                    },
                    "closed_from": {
                        "type": "string",
                        "description": "Optional ISO date/datetime lower bound for task close date.",
                    },
                    "closed_to": {
                        "type": "string",
                        "description": "Optional ISO date/datetime upper bound for task close date.",
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
                    "comment_lookup_task_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_COMMENT_LOOKUP_TASK_LIMIT,
                        "description": "Safety cap for how many candidate tasks may load comments. Default: 50.",
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
        if self._client is None and self._portal_search is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="BitrixClient or portal search index is required.",
            )

        task_id = _safe_int(args.get("task_id") or args.get("id") or args.get("ID"))
        if task_id is not None:
            read_client, access_actor, access_error = await resolve_current_user_read_client(
                self.name,
                fallback_client=self._client,
                bitrix_oauth=self._bitrix_oauth,
                user_id=user_id,
            )
            if access_error is not None:
                return access_error
            return await self._execute_task_detail(
                task_id, user_id=user_id, client=read_client, access_actor=access_actor
            )

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
        comment_query = _first_arg_text(args, "comment_query", "comments_query", "comment_text", "comment")
        include_comments = _truthy(args.get("include_comments")) or bool(comment_query)
        comment_lookup_limit = _bounded_int(
            args.get("comment_lookup_task_limit"),
            default=DEFAULT_COMMENT_LOOKUP_TASK_LIMIT,
            minimum=1,
            maximum=MAX_COMMENT_LOOKUP_TASK_LIMIT,
        )
        project_id = _safe_int(args.get("project_id") or args.get("group_id") or args.get("GROUP_ID"))
        project_query = _first_arg_text(args, "project_name", "project", "group_name")
        project: dict[str, Any] | None = None
        errors: list[dict[str, str]] = []

        if project_id is None and project_query:
            read_client, access_actor, access_error = await resolve_current_user_read_client(
                self.name,
                fallback_client=self._client,
                bitrix_oauth=self._bitrix_oauth,
                user_id=user_id,
            )
            if access_error is not None:
                return access_error
            project_matches, project_errors = await _search_projects(read_client, project_query, limit=1)
            errors.extend(project_errors)
            if not project_matches:
                return ToolResult(
                    status=ToolStatus.OK,
                    tool=self.name,
                    data={
                        "mode": "list",
                        "source": "live_bitrix_rest",
                        "access_actor": access_actor,
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

        snapshot_result = _snapshot_task_search(
            self._portal_search,
            query=comment_query or query,
            scope=scope,
            status=status,
            user_id=user_id,
            project_id=project_id,
            created_from=_first_arg_text(args, "created_from", "created_start"),
            created_to=_first_arg_text(args, "created_to", "created_end"),
            deadline_from=_first_arg_text(args, "deadline_from", "deadline_start"),
            deadline_to=_first_arg_text(args, "deadline_to", "deadline_end"),
            closed_from=_first_arg_text(args, "closed_from", "closed_start"),
            closed_to=_first_arg_text(args, "closed_to", "closed_end"),
            limit=limit,
            offset=offset,
            project=project,
            include_comments=include_comments,
            comment_query=comment_query,
        )
        if snapshot_result is not None:
            return snapshot_result

        read_client, access_actor, access_error = await resolve_current_user_read_client(
            self.name,
            fallback_client=self._client,
            bitrix_oauth=self._bitrix_oauth,
            user_id=user_id,
        )
        if access_error is not None:
            return access_error

        calls = _task_search_calls(
            scope=scope,
            status=status,
            user_id=user_id,
            project_id=project_id,
            query="" if include_comments else query,
        )
        sorted_tasks, task_errors = await _fetch_merged_tasks(read_client, calls)
        errors.extend(task_errors)
        if errors and not sorted_tasks:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error="Bitrix task search failed.",
                data={"scope": scope, "status": status, "errors": errors},
            )

        pre_comment_filtered = [
            task
            for task in sorted_tasks
            if _matches_task_status(task, status)
            and _matches_date_range(
                task,
                keys=("createdDate", "CREATED_DATE"),
                from_value=_first_arg_text(args, "created_from", "created_start"),
                to_value=_first_arg_text(args, "created_to", "created_end"),
            )
            and _matches_deadline_range(
                task,
                from_value=_first_arg_text(args, "deadline_from", "deadline_start"),
                to_value=_first_arg_text(args, "deadline_to", "deadline_end"),
            )
            and _matches_date_range(
                task,
                keys=("closedDate", "CLOSED_DATE"),
                from_value=_first_arg_text(args, "closed_from", "closed_start"),
                to_value=_first_arg_text(args, "closed_to", "closed_end"),
            )
            and (status != "overdue" or _is_overdue(task))
        ]

        if include_comments:
            comment_errors = await _attach_task_comments(
                read_client,
                pre_comment_filtered,
                query=comment_query or query,
                limit=comment_lookup_limit,
            )
            errors.extend(comment_errors)

        filtered = [
            task
            for task in pre_comment_filtered
            if _matches_text_query(task, query, include_comments=include_comments)
            and _matches_comment_query(task, comment_query)
        ]
        page = filtered[offset : offset + limit]
        items = [_task_summary(task, user_id=user_id or 0) for task in page]
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "mode": "list",
                "source": "live_bitrix_rest",
                "access_actor": access_actor,
                "scope": scope,
                "scope_label": _scope_label(scope),
                "status": status,
                "query": query,
                "comment_query": comment_query,
                "include_comments": include_comments,
                "comment_lookup_task_limit": comment_lookup_limit if include_comments else 0,
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

    async def _execute_task_detail(
        self,
        task_id: int,
        *,
        user_id: int | None,
        client: BitrixToolClientPort,
        access_actor: str,
    ) -> ToolResult:
        try:
            result = await client.result("tasks.task.get", {"taskId": task_id, "select": TASK_SELECT})
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
                "access_actor": access_actor,
                "task_id": str(task_id),
                "item": _task_summary(task, user_id=user_id or 0) if task else None,
            },
        )


def _snapshot_task_search(
    portal_search: PortalSearchPort | None,
    *,
    query: str,
    scope: str,
    status: str,
    user_id: int | None,
    project_id: int | None,
    created_from: str,
    created_to: str,
    deadline_from: str,
    deadline_to: str,
    closed_from: str,
    closed_to: str,
    limit: int,
    offset: int,
    project: dict[str, Any] | None,
    include_comments: bool,
    comment_query: str,
) -> ToolResult | None:
    if portal_search is None or not query or not include_comments:
        return None
    if not _snapshot_scope_is_user_bounded(scope, user_id=user_id):
        return None
    try:
        stats = portal_search.stats()
    except Exception:
        return None
    if not bool(getattr(stats, "exists", False)):
        return None
    by_type = getattr(stats, "by_type", {}) or {}
    if int(by_type.get("task") or 0) <= 0:
        return None

    search_limit = min(500, max(100, offset + limit * 10))
    try:
        candidates = portal_search.search(query, entity_types={"task"}, limit=search_limit)
    except Exception:
        return None

    tasks = [_snapshot_result_to_task(item, query=query) for item in candidates]
    filtered = [
        task
        for task in tasks
        if _matches_snapshot_scope(task, scope=scope, user_id=user_id)
        and (project_id is None or _safe_int(_first_text(task, "groupId", "GROUP_ID")) == project_id)
        and _matches_task_status(task, status)
        and _matches_date_range(
            task, keys=("createdDate", "CREATED_DATE"), from_value=created_from, to_value=created_to
        )
        and _matches_deadline_range(task, from_value=deadline_from, to_value=deadline_to)
        and _matches_date_range(task, keys=("closedDate", "CLOSED_DATE"), from_value=closed_from, to_value=closed_to)
        and (status != "overdue" or _is_overdue(task))
        and _matches_text_query(task, query, include_comments=True)
        and _matches_comment_query(task, comment_query)
    ]
    filtered.sort(
        key=lambda task: (
            -int(task.get("_snapshot_score") or 0),
            _deadline_sort_key(task),
            _safe_int(_task_id(task)) or 0,
        )
    )
    page = filtered[offset : offset + limit]
    return ToolResult(
        status=ToolStatus.OK,
        tool=BitrixTaskSearchTool.name,
        data={
            "mode": "list",
            "source": "postgres_portal_snapshot",
            "scope": scope,
            "scope_label": _scope_label(scope),
            "status": status,
            "query": query if query != comment_query else "",
            "comment_query": comment_query or query,
            "include_comments": True,
            "comment_lookup_task_limit": 0,
            "project": project,
            "project_query": project.get("name", "") if project else "",
            "items": [_task_summary(task, user_id=user_id or 0) for task in page],
            "total": len(filtered),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(filtered),
            "errors": [],
        },
    )


def _snapshot_scope_is_user_bounded(scope: str, *, user_id: int | None) -> bool:
    return user_id is not None and scope in {"my", "responsible", "created_by", "member"}


def _snapshot_result_to_task(item: Any, *, query: str) -> dict[str, Any]:
    metadata = dict(getattr(item, "metadata", {}) or {})
    body = str(getattr(item, "body", "") or "")
    comments = _snapshot_comment_texts(body)
    matched_comments = [
        {"POST_MESSAGE": comment} for comment in comments if query.casefold().strip() in comment.casefold()
    ]
    snippet = _snapshot_comment_snippet(comments, query=query)
    responsible_label = str(metadata.get("responsible_label") or "")
    creator_label = str(metadata.get("creator_label") or "")
    return {
        "id": str(getattr(item, "entity_id", "") or ""),
        "title": str(getattr(item, "title", "") or "задача"),
        "description": body,
        "status": str(metadata.get("status") or ""),
        "responsibleId": str(metadata.get("responsible_id") or ""),
        "createdBy": str(metadata.get("created_by") or ""),
        "groupId": str(metadata.get("group_id") or ""),
        "deadline": str(metadata.get("deadline") or ""),
        "createdDate": str(metadata.get("created_date") or ""),
        "closedDate": str(metadata.get("closed_date") or ""),
        "accomplices": metadata.get("accomplices") or [],
        "auditors": metadata.get("auditors") or [],
        "responsible": {"name": responsible_label} if responsible_label else {},
        "creator": {"name": creator_label} if creator_label else {},
        "_comments": [{"POST_MESSAGE": comment} for comment in comments],
        "_matched_comments": matched_comments,
        "_comment_snippets": [snippet] if snippet else [],
        "_snapshot_score": int(getattr(item, "score", 0) or 0),
        "_snapshot_url": str(getattr(item, "url", "") or ""),
    }


def _snapshot_comment_texts(body: str) -> list[str]:
    marker = "Комментарии:"
    if marker not in body:
        return []
    tail = body.split(marker, 1)[1]
    comments = []
    for line in tail.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            comments.append(stripped[2:].strip())
    return [comment for comment in comments if comment]


def _snapshot_comment_snippet(comments: list[str], *, query: str) -> str:
    needle = query.casefold().strip()
    if not needle:
        return comments[0][:180] if comments else ""
    for comment in comments:
        lower = comment.casefold()
        pos = lower.find(needle)
        if pos < 0:
            continue
        start = max(0, pos - 60)
        end = min(len(comment), pos + len(query) + 100)
        prefix = "..." if start else ""
        suffix = "..." if end < len(comment) else ""
        return f"{prefix}{comment[start:end]}{suffix}"
    return ""


def _matches_snapshot_scope(task: dict[str, Any], *, scope: str, user_id: int | None) -> bool:
    if scope == "all":
        return True
    if user_id is None:
        return False
    if scope == "responsible":
        return _safe_int(_first_text(task, "responsibleId", "RESPONSIBLE_ID")) == user_id
    if scope == "created_by":
        return _safe_int(_first_text(task, "createdBy", "CREATED_BY")) == user_id
    return user_id in {
        _safe_int(_first_text(task, "responsibleId", "RESPONSIBLE_ID")),
        _safe_int(_first_text(task, "createdBy", "CREATED_BY")),
        *_int_list(task.get("accomplices") or task.get("ACCOMPLICES")),
        *_int_list(task.get("auditors") or task.get("AUDITORS")),
    }


class BitrixProjectSearchTool:
    name = "bitrix_project_search"

    def __init__(
        self,
        client: BitrixToolClientPort | None = None,
        portal_search: PortalSearchPort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
    ) -> None:
        self._client = client
        self._portal_search = portal_search
        self._bitrix_oauth = bitrix_oauth

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
        if self._client is None and self._portal_search is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="BitrixClient or PortalSearchIndex is required",
            )
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
        snapshot_matches = _search_projects_snapshot(self._portal_search, query, limit=limit)
        if snapshot_matches is not None:
            return ToolResult(
                status=ToolStatus.OK,
                tool=self.name,
                data={
                    "source": "postgres_portal_snapshot",
                    "query": query,
                    "items": snapshot_matches,
                    "total": len(snapshot_matches),
                    "limit": limit,
                    "errors": [],
                },
            )
        read_client, access_actor, access_error = await resolve_current_user_read_client(
            self.name,
            fallback_client=self._client,
            bitrix_oauth=self._bitrix_oauth,
            user_id=user_id,
        )
        if access_error is not None:
            return access_error
        matches, errors = await _search_projects(read_client, query, limit=limit)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "source": "live_bitrix_rest",
                "access_actor": access_actor,
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


def _search_projects_snapshot(
    portal_search: PortalSearchPort | None,
    query: str,
    *,
    limit: int,
) -> list[dict[str, Any]] | None:
    if portal_search is None:
        return None
    try:
        if not portal_search.stats().exists:
            return None
        results = []
        seen: set[tuple[str, str]] = set()
        for variant in _project_search_variants(query):
            for item in portal_search.search(variant, entity_types={"project"}, limit=limit):
                key = (str(getattr(item, "entity_type", "")), str(getattr(item, "entity_id", "")))
                if key in seen:
                    continue
                seen.add(key)
                results.append(item)
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
    except Exception:
        return None
    results = _filter_project_snapshot_results(results, query=query, limit=limit)
    if not results:
        return None
    return [_project_summary_from_index(item) for item in results]


def _project_search_variants(query: str) -> list[str]:
    variants = [query]
    normalized = _normalize_project_name(query)
    if normalized and normalized not in {item.casefold() for item in variants}:
        variants.append(normalized)
    return variants


def _filter_project_snapshot_results(items: list[Any], *, query: str, limit: int) -> list[Any]:
    normalized_query = _normalize_project_name(query)
    query_terms = set(normalized_query.split())
    scored: list[tuple[int, str, Any]] = []
    for item in items:
        name = str(getattr(item, "title", "") or "")
        normalized_name = _normalize_project_name(name)
        if not normalized_name:
            continue
        score = 0
        if normalized_name == normalized_query:
            score = 100
        elif normalized_query and normalized_query in normalized_name:
            score = 80
        elif query_terms and query_terms.issubset(set(normalized_name.split())):
            score = 70
        if score:
            scored.append((score, normalized_name, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _score, _name, item in scored[:limit]]


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
    created_date = _first_text(task, "createdDate", "CREATED_DATE")
    closed_date = _first_text(task, "closedDate", "CLOSED_DATE")
    group_id = _first_text(task, "groupId", "GROUP_ID")
    return {
        "id": task_id,
        "title": title,
        "description": _first_text(task, "description", "DESCRIPTION"),
        "status": status,
        "status_label": _status_label(status),
        "created_date": created_date,
        "created_label": _date_label(created_date),
        "closed_date": closed_date,
        "closed_label": _date_label(closed_date),
        "deadline": deadline,
        "deadline_label": _deadline_label(deadline),
        "group_id": group_id,
        "roles": _task_roles(task, user_id=user_id) if user_id else [],
        "responsible_label": _person_label(task.get("responsible") or task.get("RESPONSIBLE")),
        "creator_label": _person_label(task.get("creator") or task.get("CREATOR")),
        "comment_snippets": list(task.get("_comment_snippets") or []),
        "matched_comment_count": len(task.get("_matched_comments") or []),
    }


def _project_summary(group: dict[str, Any]) -> dict[str, Any]:
    group_id = _first_text(group, "ID", "id")
    name = _first_text(group, "NAME", "name", "TITLE", "title") or "проект"
    return {
        "id": group_id,
        "name": name,
        "description": _first_text(group, "DESCRIPTION", "description"),
    }


def _project_summary_from_index(item: Any) -> dict[str, Any]:
    return {
        "id": str(getattr(item, "entity_id", "") or ""),
        "name": str(getattr(item, "title", "") or ""),
        "description": _project_description_from_index(getattr(item, "body", "") or ""),
    }


def _project_description_from_index(body: object) -> str:
    lines = [line.strip() for line in str(body or "").splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    return "" if first.startswith("Проект:") or first.startswith("Владелец:") else first


def _person_label(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    name = _first_text(value, "name", "NAME")
    if name:
        return name
    parts = [
        _first_text(value, "lastName", "LAST_NAME", "last_name"),
        _first_text(value, "name", "NAME", "firstName", "FIRST_NAME"),
        _first_text(value, "secondName", "SECOND_NAME", "second_name"),
    ]
    return " ".join(part for part in parts if part)


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


def _matches_text_query(task: dict[str, Any], query: str, *, include_comments: bool = False) -> bool:
    if not query:
        return True
    needle = query.casefold().strip()
    parts = [
        _first_text(task, "title", "TITLE"),
        _first_text(task, "description", "DESCRIPTION"),
    ]
    if include_comments:
        parts.extend(_comment_texts(task.get("_comments")))
    haystack = " ".join([part for part in parts if part]).casefold()
    return needle in haystack


def _matches_comment_query(task: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    needle = query.casefold().strip()
    return any(needle in text.casefold() for text in _comment_texts(task.get("_comments")))


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
    return _matches_date_range(task, keys=("deadline", "DEADLINE"), from_value=from_value, to_value=to_value)


def _matches_date_range(
    task: dict[str, Any],
    *,
    keys: tuple[str, str],
    from_value: str,
    to_value: str,
) -> bool:
    if not from_value and not to_value:
        return True
    value = _parse_datetime(_first_text(task, *keys))
    if value is None:
        return False
    from_dt = _parse_range_datetime(from_value, is_end=False)
    to_dt = _parse_range_datetime(to_value, is_end=True)
    if from_dt is not None and value < from_dt:
        return False
    return not (to_dt is not None and value > to_dt)


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


def _parse_range_datetime(value: str, *, is_end: bool) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) == 10:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            return None
        boundary_time = time.max if is_end else time.min
        return datetime.combine(parsed_date, boundary_time, tzinfo=UTC)
    return _parse_datetime(text)


def _deadline_label(deadline: str) -> str:
    if not deadline:
        return "без срока"
    parsed = _parse_datetime(deadline)
    if parsed is None:
        return deadline
    return parsed.astimezone().strftime("%d.%m.%Y %H:%M")


def _date_label(value: str) -> str:
    parsed = _parse_datetime(value)
    if parsed is None:
        return value
    return parsed.astimezone().strftime("%d.%m.%Y %H:%M")


async def _attach_task_comments(
    client: BitrixToolClientPort,
    tasks: list[dict[str, Any]],
    *,
    query: str,
    limit: int,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    for task in tasks[:limit]:
        task_id = _task_id(task)
        if not task_id:
            continue
        try:
            result = await client.result("task.commentitem.getlist", {"TASKID": task_id})
        except (BitrixApiError, BitrixConfigError) as exc:
            errors.append({"source": f"task.commentitem.getlist:{task_id}", "error": str(exc)})
            continue
        comments = _extract_comments(result)
        task["_comments"] = comments
        matched = _matched_comments(comments, query)
        task["_matched_comments"] = matched
        task["_comment_snippets"] = [_comment_snippet(comment, query=query) for comment in matched[:2]]
    return errors


def _extract_comments(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ("comments", "COMMENTS", "items", "result"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _extract_comments(value)
                if nested:
                    return nested
    return []


def _matched_comments(comments: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    comments = [comment for comment in comments if not _is_system_comment(comment)]
    if not query:
        return comments
    needle = query.casefold().strip()
    return [comment for comment in comments if needle in _comment_text(comment).casefold()]


def _comment_texts(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        _comment_text(comment)
        for comment in value
        if isinstance(comment, dict) and _comment_text(comment) and not _is_system_comment(comment)
    ]


def _comment_text(comment: dict[str, Any]) -> str:
    return _clean_comment_text(
        _first_text(comment, "POST_MESSAGE", "POST_MESSAGE_HTML", "POST_MESSAGE_TEXT", "text", "message")
    )


def _clean_comment_text(value: str) -> str:
    text = html.unescape(str(value or ""))
    previous = None
    while previous != text:
        previous = text
        text = _BITRIX_PAIRED_TAG_RE.sub(r"\2", text)
    text = _BITRIX_SINGLE_TAG_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    return " ".join(text.split())


def _is_system_comment(comment: dict[str, Any]) -> bool:
    text = _comment_text(comment).casefold().strip()
    if not text:
        return True
    normalized = text.rstrip(".")
    if normalized.startswith("крайний срок изменен на:"):
        return True
    if normalized in {
        "задача завершена",
        "задача возвращена в работу",
        "задача почти просрочена",
    }:
        return True
    system_fragments = (
        "вы добавлены наблюдателем",
        "вы назначены исполнителем",
        "задача почти просрочена",
        "завершите задачу или передвиньте срок",
    )
    return any(fragment in normalized for fragment in system_fragments)


def _comment_snippet(comment: dict[str, Any], *, query: str) -> str:
    text = _comment_text(comment)
    if not text:
        return ""
    if not query:
        return text[:180]
    lower = text.casefold()
    pos = lower.find(query.casefold().strip())
    if pos < 0:
        return text[:180]
    start = max(0, pos - 60)
    end = min(len(text), pos + len(query) + 100)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


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
