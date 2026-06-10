from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixClient, BitrixConfigError
from ai_server.integrations.bitrix.dialog_state import BitrixPendingActionService, PendingBitrixAction
from ai_server.integrations.bitrix.portal_search import (
    PortalSearchIndex,
    entity_types_for_scope,
    format_portal_search_results,
)
from ai_server.models import ToolDefinition, ToolResult
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy
from ai_server.utils import optional_int


class BitrixToolset:
    def __init__(
        self,
        client: BitrixClient | None = None,
        *,
        portal_search: PortalSearchIndex | None = None,
        pending_actions: BitrixPendingActionService | None = None,
        dialog_key: str | None = None,
        user_id: int | None = None,
    ) -> None:
        self.client = client or BitrixClient()
        self.portal_search = portal_search or PortalSearchIndex()
        self.pending_actions = pending_actions
        self.dialog_key = dialog_key
        self.user_id = user_id

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="bitrix_api",
                description="Controlled Bitrix REST API access. Read methods can run immediately; writes return approval requests.",
                parameters={
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["call", "confirm_pending", "cancel_pending"]},
                        "method": {"type": "string"},
                        "params": {"type": "object"},
                        "summary": {"type": "string"},
                    },
                    "required": ["action", "method", "params"],
                },
            ),
            ToolDefinition(
                name="portal_search",
                description="Search the local Bitrix portal index from var/search_index.sqlite.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "scope": {"type": "string", "enum": ["all", "documents", "files", "tasks", "projects"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="resolve_user",
                description="Read-only Bitrix user resolver. Returns one user only when the query is unambiguous.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="resolve_project",
                description="Read-only Bitrix project/workgroup resolver. Returns one project only when the query is unambiguous.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="task_search",
                description="Read-only Bitrix task search/get wrapper over tasks.task.list and tasks.task.get.",
                parameters={
                    "type": "object",
                    "properties": {
                        "mode": {"type": "string", "enum": ["list", "get"]},
                        "task_id": {"type": "integer"},
                        "filter": {"type": "object"},
                        "select": {"type": "array", "items": {"type": "string"}},
                        "order": {"type": "object"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["mode"],
                },
            ),
            ToolDefinition(
                name="current_user_profile",
                description=(
                    "Read-only facts about the current Bitrix chat user for LLM permission reasoning: "
                    "active flag, admin flags when Bitrix exposes them, department ids, work position, and user type."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
        ]

    async def bitrix_api(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "call").strip().lower()
        method = str(args.get("method") or "").strip()
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        summary = str(args.get("summary") or method).strip()

        if action in {"confirm_pending", "cancel_pending"}:
            if not self.pending_actions or not self.dialog_key:
                return ToolResult(
                    status="not_configured",
                    tool="bitrix_api",
                    data={"action": action, "message": "Pending-action store is not bound to this tool call."},
                )
            if action == "cancel_pending":
                result = self.pending_actions.cancel(self.dialog_key)
            else:
                result = await self.pending_actions.confirm(self.dialog_key, user_id=self.user_id)
            return ToolResult(
                status=result.status,
                tool="bitrix_api",
                data={"action": action, "message": result.message, **result.data},
            )
        if action != "call":
            return ToolResult(status="invalid_tool_call", tool="bitrix_api", error=f"unknown action: {action}")

        decision = decide_bitrix_method_policy(method)
        if decision.decision == "deny":
            return ToolResult(
                status="denied",
                tool="bitrix_api",
                data={"method": method, "policy_reason": decision.reason},
            )
        if decision.decision == "confirm":
            if not params:
                return ToolResult(
                    status="invalid_tool_call",
                    tool="bitrix_api",
                    error="Bitrix write methods require real params before confirmation.",
                    data={"method": method},
                )
            if self.pending_actions and self.dialog_key:
                self.pending_actions.save_pending(
                    self.dialog_key,
                    PendingBitrixAction(
                        method=method,
                        params=params,
                        summary=summary,
                        created_by=self.user_id,
                    ),
                )
            return ToolResult(
                status="confirmation_required",
                tool="bitrix_api",
                data={"method": method, "params": params, "summary": summary, "policy_reason": decision.reason},
            )

        try:
            result = await self.client.result(method, params)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status="not_configured" if isinstance(exc, BitrixConfigError) else "error",
                tool="bitrix_api",
                error=str(exc),
                data={"method": method, "params": params},
            )
        return ToolResult(status="ok", tool="bitrix_api", data={"method": method, "params": params, "result": result})

    async def resolve_user(self, query: str, *, limit: int = 5) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult(status="invalid_tool_call", tool="resolve_user", error="query is required")
        try:
            users = await self.client.search_users(query, limit=limit)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status="not_configured" if isinstance(exc, BitrixConfigError) else "error",
                tool="resolve_user",
                error=str(exc),
                data={"query": query},
            )
        candidates = [_user_candidate(user) for user in users]
        candidates = [candidate for candidate in candidates if candidate is not None]
        return _resolution_result("resolve_user", query=query, candidates=candidates)

    async def resolve_project(self, query: str, *, limit: int = 5) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult(status="invalid_tool_call", tool="resolve_project", error="query is required")
        try:
            projects = await self.client.search_projects(query, limit=limit)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status="not_configured" if isinstance(exc, BitrixConfigError) else "error",
                tool="resolve_project",
                error=str(exc),
                data={"query": query},
            )
        candidates = [_project_candidate(project) for project in projects]
        candidates = [candidate for candidate in candidates if candidate is not None]
        return _resolution_result("resolve_project", query=query, candidates=candidates)

    async def task_search(self, args: dict[str, Any]) -> ToolResult:
        mode = str(args.get("mode") or "list").strip().lower()
        select = args.get("select") if isinstance(args.get("select"), list) else None
        if mode == "get":
            task_id = optional_int(args.get("task_id"))
            if task_id is None:
                return ToolResult(status="invalid_tool_call", tool="task_search", error="task_id is required")
            try:
                payload: dict[str, Any] = {"taskId": task_id}
                if select:
                    payload["select"] = select
                result = await self.client.result("tasks.task.get", payload)
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(
                    status="not_configured" if isinstance(exc, BitrixConfigError) else "error",
                    tool="task_search",
                    error=str(exc),
                    data={"mode": mode, "task_id": task_id},
                )
            task = _extract_task(result)
            if task is None:
                return ToolResult(status="not_found", tool="task_search", data={"mode": mode, "task_id": task_id})
            return ToolResult(status="ok", tool="task_search", data={"mode": mode, "task_id": task_id, "task": task})

        if mode != "list":
            return ToolResult(status="invalid_tool_call", tool="task_search", error=f"unknown mode: {mode}")

        filter_ = args.get("filter") if isinstance(args.get("filter"), dict) else {}
        order = args.get("order") if isinstance(args.get("order"), dict) else {"CHANGED_DATE": "desc"}
        limit = max(1, min(optional_int(args.get("limit")) or 5, 10))
        try:
            tasks = await self.client.list_all_tasks(
                filter_=filter_,
                select=select,
                order=order,
                limit=limit,
            )
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status="not_configured" if isinstance(exc, BitrixConfigError) else "error",
                tool="task_search",
                error=str(exc),
                data={"mode": mode, "filter": filter_, "order": order, "limit": limit},
            )
        return ToolResult(
            status="ok",
            tool="task_search",
            data={
                "mode": mode,
                "filter": filter_,
                "order": order,
                "limit": limit,
                "tasks": tasks,
                "total": len(tasks),
            },
        )

    async def current_user_profile(self, args: dict[str, Any] | None = None) -> ToolResult:
        args = args or {}
        user_id = self.user_id if self.user_id is not None else optional_int(args.get("user_id"))
        if user_id is None:
            return ToolResult(
                status="not_available",
                tool="current_user_profile",
                error="current Bitrix user id is not available",
            )
        try:
            user = await self.client.get_user(user_id)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status="not_configured" if isinstance(exc, BitrixConfigError) else "error",
                tool="current_user_profile",
                error=str(exc),
                data={"user_id": user_id},
            )
        if user is None:
            return ToolResult(status="not_found", tool="current_user_profile", data={"user_id": user_id})
        return ToolResult(
            status="ok",
            tool="current_user_profile",
            data={"user_id": user_id, "profile": _compact_user_profile(user)},
        )

    def portal_search_contract(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        scope = str(args.get("scope") or "all").strip().lower()
        limit = max(1, min(int(args.get("limit") or 10), 30))
        if not query:
            return ToolResult(status="invalid_tool_call", tool="portal_search", error="query is required")

        entity_types = entity_types_for_scope(scope)
        if entity_types is None and scope not in {"", "all"}:
            return ToolResult(status="invalid_tool_call", tool="portal_search", error=f"unknown scope: {scope}")

        stats = self.portal_search.stats()
        if not stats.exists:
            return ToolResult(
                status="not_configured",
                tool="portal_search",
                data={
                    "query": query,
                    "scope": scope,
                    "limit": limit,
                    "index_path": str(stats.path),
                    "message": "Local portal search index is missing. Run cutover var import or indexing first.",
                },
            )

        results = self.portal_search.search(query, entity_types=entity_types, limit=limit)
        return ToolResult(
            status="ok",
            tool="portal_search",
            data={
                "query": query,
                "scope": scope,
                "limit": limit,
                "index_path": str(stats.path),
                "summary": format_portal_search_results(results, query=query),
                "results": [result.as_dict() for result in results],
            },
        )


def _resolution_result(tool: str, *, query: str, candidates: list[dict[str, Any]]) -> ToolResult:
    if not candidates:
        return ToolResult(status="not_found", tool=tool, data={"query": query, "candidates": []})
    if len(candidates) == 1:
        return ToolResult(
            status="ok", tool=tool, data={"query": query, "candidate": candidates[0], "candidates": candidates}
        )
    return ToolResult(status="ambiguous", tool=tool, data={"query": query, "candidates": candidates})


def _user_candidate(user: dict[str, Any]) -> dict[str, Any] | None:
    user_id = optional_int(user.get("ID") or user.get("id"))
    if user_id is None:
        return None
    first_name = str(user.get("NAME") or user.get("name") or "").strip()
    last_name = str(user.get("LAST_NAME") or user.get("lastName") or user.get("last_name") or "").strip()
    second_name = str(user.get("SECOND_NAME") or user.get("secondName") or user.get("second_name") or "").strip()
    email = str(user.get("EMAIL") or user.get("email") or "").strip()
    full_name = " ".join(part for part in (last_name, first_name, second_name) if part).strip()
    return {
        "id": user_id,
        "label": full_name or email or f"Bitrix user #{user_id}",
        "email": email,
        "raw": user,
    }


def _project_candidate(project: dict[str, Any]) -> dict[str, Any] | None:
    project_id = optional_int(project.get("ID") or project.get("id"))
    if project_id is None:
        return None
    name = str(project.get("NAME") or project.get("name") or "").strip()
    return {
        "id": project_id,
        "label": name or f"Bitrix project #{project_id}",
        "raw": project,
    }


def _extract_task(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    task = result.get("task")
    if isinstance(task, dict):
        return task
    if result.get("id") or result.get("ID"):
        return result
    return None


def _compact_user_profile(user: dict[str, Any]) -> dict[str, Any]:
    user_id = optional_int(user.get("ID") or user.get("id"))
    first_name = str(user.get("NAME") or user.get("name") or "").strip()
    last_name = str(user.get("LAST_NAME") or user.get("lastName") or user.get("last_name") or "").strip()
    second_name = str(user.get("SECOND_NAME") or user.get("secondName") or user.get("second_name") or "").strip()
    full_name = " ".join(part for part in (last_name, first_name, second_name) if part).strip()
    email = str(user.get("EMAIL") or user.get("email") or "").strip()
    work_position = str(user.get("WORK_POSITION") or user.get("workPosition") or "").strip()
    departments = _int_list(user.get("UF_DEPARTMENT") or user.get("ufDepartment") or user.get("department"))
    return {
        "id": user_id,
        "label": full_name or email or (f"Bitrix user #{user_id}" if user_id is not None else "Bitrix user"),
        "active": _truthy(user.get("ACTIVE") or user.get("active"), default=True),
        "is_admin": _truthy(
            user.get("IS_ADMIN") or user.get("ADMIN") or user.get("isAdmin") or user.get("is_admin"),
            default=False,
        ),
        "department_ids": departments,
        "work_position": work_position,
        "user_type": str(user.get("USER_TYPE") or user.get("userType") or "").strip(),
        "raw_policy_fields": {
            key: user[key]
            for key in sorted(user)
            if key
            in {
                "ID",
                "ACTIVE",
                "IS_ADMIN",
                "ADMIN",
                "USER_TYPE",
                "UF_DEPARTMENT",
                "WORK_POSITION",
            }
        },
    }


def _int_list(value: object) -> list[int]:
    raw_values = value if isinstance(value, list) else [value]
    result: list[int] = []
    for raw in raw_values:
        item = optional_int(raw)
        if item is not None and item not in result:
            result.append(item)
    return result


def _truthy(value: object, *, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "да"}
    return bool(value)


