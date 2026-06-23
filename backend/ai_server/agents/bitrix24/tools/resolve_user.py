from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.ports import BitrixToolClientPort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.utils import optional_int


class ResolveUserTool:
    """Internal tool: resolve a Bitrix user by query. Not exposed to LLM."""

    name = "resolve_user"

    def __init__(self, client: BitrixToolClientPort | None = None) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="resolve_user",
            description="Internal: search Bitrix users by name or email.",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        query = str(args.get("query") or "").strip()
        limit = int(args.get("limit") or 5)
        if not query:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool="resolve_user", error="query is required")
        if self._client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED, tool="resolve_user", error="BitrixClient is not injected"
            )
        try:
            users = await self._client.search_users(query, limit=limit)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="resolve_user",
                error=str(exc),
                data={"query": query},
            )
        candidates = [c for c in (_user_candidate(u) for u in users) if c is not None]
        return _resolution_result("resolve_user", query=query, candidates=candidates)


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
