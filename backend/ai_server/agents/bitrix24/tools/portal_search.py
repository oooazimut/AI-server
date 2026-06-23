from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.portal_search import (
    PortalSearchIndex,
    entity_types_for_scope,
    format_portal_search_results,
)
from ai_server.models import ToolDefinition, ToolResult, ToolStatus


class PortalSearchTool:
    name = "portal_search"

    def __init__(self, portal_search: PortalSearchIndex | None = None) -> None:
        self._portal_search = portal_search

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
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
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if self._portal_search is None:
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

        stats = self._portal_search.stats()
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

        results = self._portal_search.search(query, entity_types=entity_types, limit=limit)
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
