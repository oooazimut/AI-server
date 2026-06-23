from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.ports import BitrixToolClientPort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.utils import optional_int


class ResolveProjectTool:
    """Internal tool: resolve a Bitrix project by query. Not exposed to LLM."""

    name = "resolve_project"

    def __init__(self, client: BitrixToolClientPort | None = None) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="resolve_project",
            description="Internal: search Bitrix projects/groups by name.",
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
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool="resolve_project", error="query is required")
        if self._client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED, tool="resolve_project", error="BitrixClient is not injected"
            )
        try:
            projects = await self._client.search_projects(query, limit=limit)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="resolve_project",
                error=str(exc),
                data={"query": query},
            )
        candidates = [c for c in (_project_candidate(p) for p in projects) if c is not None]
        return _resolution_result("resolve_project", query=query, candidates=candidates)


def _resolution_result(tool: str, *, query: str, candidates: list[dict[str, Any]]) -> ToolResult:
    if not candidates:
        return ToolResult(status=ToolStatus.NOT_FOUND, tool=tool, data={"query": query, "candidates": []})
    if len(candidates) == 1:
        return ToolResult(
            status=ToolStatus.OK, tool=tool, data={"query": query, "candidate": candidates[0], "candidates": candidates}
        )
    return ToolResult(status=ToolStatus.AMBIGUOUS, tool=tool, data={"query": query, "candidates": candidates})


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
