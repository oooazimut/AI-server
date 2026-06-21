from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.dialog_state import BitrixPendingActionService, PendingBitrixAction
from ai_server.integrations.bitrix.portal_search import (
    PortalSearchIndex,
    entity_types_for_scope,
    format_portal_search_results,
)
from ai_server.integrations.bitrix.ports import BitrixRestPort, BitrixToolClientPort
from ai_server.integrations.bitrix.profile import compact_user_profile
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy
from ai_server.utils import optional_int


class BitrixToolset:
    def __init__(
        self,
        client: BitrixToolClientPort | None = None,
        *,
        portal_search: PortalSearchIndex | None = None,
        pending_actions: BitrixPendingActionService | None = None,
        dialog_key: str | None = None,
        user_id: int | None = None,
        actor_client: BitrixRestPort | None = None,
        auto_execute: bool = False,
    ) -> None:
        self.client = client
        self.portal_search = portal_search
        self.pending_actions = pending_actions
        self.dialog_key = dialog_key
        self.user_id = user_id
        self._actor_client = actor_client
        self._auto_execute = auto_execute

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="bitrix_api",
                description=(
                    "Bitrix24 REST API access. Read methods (ending in .get/.list/.search) execute immediately. "
                    "Write methods require user confirmation. Dangerous methods (user management, bots) are denied."
                ),
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
                description="Search the local Bitrix portal index (var/search_index.sqlite). Use for full-text search across tasks, projects, documents and disk files.",
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
        if self.client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool="bitrix_api", error="BitrixClient is not injected")
        action = str(args.get("action") or "call").strip().lower()
        method = str(args.get("method") or "").strip()
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        summary = str(args.get("summary") or method).strip()

        if action in {"confirm_pending", "cancel_pending"}:
            return await self._handle_pending_action(action)
        if action != "call":
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool="bitrix_api", error=f"unknown action: {action}")
        return await self._call_api(method, params, summary)

    async def _handle_pending_action(self, action: str) -> ToolResult:
        if not self.pending_actions or not self.dialog_key:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
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

    async def _call_api(self, method: str, params: dict[str, Any], summary: str) -> ToolResult:
        decision = decide_bitrix_method_policy(method)
        if decision.decision == "deny":
            return ToolResult(
                status=ToolStatus.DENIED,
                tool="bitrix_api",
                data={"method": method, "policy_reason": decision.reason},
            )
        if decision.decision == "confirm":
            return await self._confirm_or_queue(method, params, summary, policy_reason=decision.reason)
        try:
            result = await self.client.result(method, params)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="bitrix_api",
                error=str(exc),
                data={"method": method, "params": params},
            )
        return ToolResult(
            status=ToolStatus.OK, tool="bitrix_api", data={"method": method, "params": params, "result": result}
        )

    async def _confirm_or_queue(
        self, method: str, params: dict[str, Any], summary: str, *, policy_reason: str
    ) -> ToolResult:
        if self._auto_execute:
            execute_client = self._actor_client or self.client
            try:
                result = await execute_client.result(method, params)
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                    tool="bitrix_api",
                    error=str(exc),
                    data={"method": method, "params": params},
                )
            return ToolResult(
                status=ToolStatus.OK, tool="bitrix_api", data={"method": method, "params": params, "result": result}
            )
        if not params:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
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
            status=ToolStatus.CONFIRMATION_REQUIRED,
            tool="bitrix_api",
            data={"method": method, "params": params, "summary": summary, "policy_reason": policy_reason},
        )

    async def resolve_user(self, query: str, *, limit: int = 5) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool="resolve_user", error="query is required")
        try:
            users = await self.client.search_users(query, limit=limit)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
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
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool="resolve_project", error="query is required")
        try:
            projects = await self.client.search_projects(query, limit=limit)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="resolve_project",
                error=str(exc),
                data={"query": query},
            )
        candidates = [_project_candidate(project) for project in projects]
        candidates = [candidate for candidate in candidates if candidate is not None]
        return _resolution_result("resolve_project", query=query, candidates=candidates)

    async def current_user_profile(self, args: dict[str, Any] | None = None) -> ToolResult:
        args = args or {}
        user_id = self.user_id if self.user_id is not None else optional_int(args.get("user_id"))
        if user_id is None:
            return ToolResult(
                status=ToolStatus.NOT_AVAILABLE,
                tool="current_user_profile",
                error="current Bitrix user id is not available",
            )
        try:
            user = await self.client.get_user(user_id)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="current_user_profile",
                error=str(exc),
                data={"user_id": user_id},
            )
        if user is None:
            return ToolResult(status=ToolStatus.NOT_FOUND, tool="current_user_profile", data={"user_id": user_id})
        return ToolResult(
            status=ToolStatus.OK,
            tool="current_user_profile",
            data={"user_id": user_id, "profile": compact_user_profile(user)},
        )

    def portal_search_contract(self, args: dict[str, Any]) -> ToolResult:
        if self.portal_search is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED, tool="portal_search", error="PortalSearchIndex is not injected"
            )
        query = str(args.get("query") or "").strip()
        scope = str(args.get("scope") or "all").strip().lower()
        limit = max(1, min(int(args.get("limit") or 10), 30))
        if not query:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool="portal_search", error="query is required")

        entity_types = entity_types_for_scope(scope)
        if entity_types is None and scope not in {"", "all"}:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL, tool="portal_search", error=f"unknown scope: {scope}"
            )

        stats = self.portal_search.stats()
        if not stats.exists:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
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
            status=ToolStatus.OK,
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
        return ToolResult(status=ToolStatus.NOT_FOUND, tool=tool, data={"query": query, "candidates": []})
    if len(candidates) == 1:
        return ToolResult(
            status=ToolStatus.OK, tool=tool, data={"query": query, "candidate": candidates[0], "candidates": candidates}
        )
    return ToolResult(status=ToolStatus.AMBIGUOUS, tool=tool, data={"query": query, "candidates": candidates})


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
