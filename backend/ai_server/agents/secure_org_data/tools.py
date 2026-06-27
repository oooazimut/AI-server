from __future__ import annotations

from typing import Any

from ai_server.agents.secure_org_data.store import SecureOrgDataStore
from ai_server.models import ToolDefinition, ToolResult, ToolStatus


class SecureOrgDataSearchTool:
    name = "search_org_data"

    def __init__(self, store: SecureOrgDataStore | None = None) -> None:
        self._store = store or SecureOrgDataStore()

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Read-only search over the local organization data/index. "
                "Access level is taken from existing metadata/index markers, not inferred by the LLM."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "include_paths": {"type": "boolean"},
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
        limit = int(args.get("limit") or 5)
        include_paths = bool(args.get("include_paths", True))
        if not query:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="query is required")
        result = self._store.search(query, user_id=user_id, limit=limit, include_paths=include_paths)
        if not result.get("configured", True):
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, data=result, error=result.get("error"))
        return ToolResult(status=ToolStatus.OK, tool=self.name, data=result)
