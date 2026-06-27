from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.portal_search import PortalSearchIndex, entity_types_for_scope
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.settings import Settings
from ai_server.tools.document_access.access_control import filter_portal_items_for_user
from ai_server.utils import optional_int


class DocumentSearchTool:
    name = "portal_document_search"

    def __init__(
        self,
        portal_search: PortalSearchIndex | None = None,
        *,
        settings: Settings,
    ) -> None:
        self._portal_search = portal_search
        self._settings = settings

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="portal_document_search",
            description="Search documents/files in the local Bitrix portal index.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {"type": "string", "enum": ["documents", "files"]},
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
        query = str(args.get("query") or "").strip()
        scope = str(args.get("scope") or "documents").strip().lower()
        limit = max(1, min(optional_int(args.get("limit")) or 10, 30))
        if not query:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL, tool="portal_document_search", error="query is required"
            )

        entity_types = entity_types_for_scope(scope)
        if entity_types is None or not entity_types <= entity_types_for_scope("files"):
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="portal_document_search",
                error=f"unknown document scope: {scope}",
            )

        if self._portal_search is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="portal_document_search",
                data={"message": "Portal search index not configured."},
            )
        stats = self._portal_search.stats()
        if not stats.exists:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="portal_document_search",
                data={"index_path": str(stats.path), "message": "Local portal search index is missing."},
            )

        results = filter_portal_items_for_user(
            self._portal_search.search(query, entity_types=entity_types, limit=limit),
            user_id=user_id,
            settings=self._settings,
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool="portal_document_search",
            data={
                "query": query,
                "scope": scope,
                "results": [item.as_dict() for item in results],
                "total": len(results),
            },
        )
