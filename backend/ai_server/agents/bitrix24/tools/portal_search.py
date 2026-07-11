from __future__ import annotations

from typing import Any

from ai_server.agents.bitrix24.tools.read_client import oauth_authorization_data, oauth_missing_error
from ai_server.integrations.bitrix.oauth import BitrixOAuthError, BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_ports import BitrixFileDownloadPort
from ai_server.tools.bitrix_search import PortalSearchPort, entity_types_for_scope, format_portal_search_results
from ai_server.utils import optional_int

_DENIED_AGENT_SCOPES = {"", "all", "tasks"}
_ACCESS_CHECKED_SCOPES = {"documents", "files"}
_DOCUMENT_ENTITY_TYPES = {"disk_file", "task_attachment"}


class PortalSearchTool:
    name = "portal_search"

    def __init__(
        self,
        portal_search: PortalSearchPort | None = None,
        bitrix_files: BitrixFileDownloadPort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
    ) -> None:
        self._portal_search = portal_search
        self._bitrix_files = bitrix_files
        self._bitrix_oauth = bitrix_oauth

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="portal_search",
            description=(
                "Search the local Bitrix portal index. Use only for focused document/file/project/catalog lookup. "
                "Do not use for tasks or unrestricted all-scope search; use bitrix_task_search for tasks."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "enum": [
                            "all",
                            "documents",
                            "files",
                            "tasks",
                            "projects",
                            "catalog",
                            "stores",
                            "products",
                            "stock",
                        ],
                    },
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
        if scope in _DENIED_AGENT_SCOPES:
            return ToolResult(
                status=ToolStatus.DENIED,
                tool="portal_search",
                error=(
                    "portal_search requires a focused non-task scope. "
                    "Use bitrix_task_search for tasks; use documents/files/projects/catalog/stores/products/stock for portal search."
                ),
                data={"query": query, "scope": scope, "limit": limit},
            )
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
        access_client: BitrixFileDownloadPort | None = None
        access_actor = "not_checked"
        if scope in _ACCESS_CHECKED_SCOPES:
            access_client, access_actor, access_error = await _resolve_document_access_client(
                tool_name=self.name,
                fallback_client=self._bitrix_files,
                bitrix_oauth=self._bitrix_oauth,
                user_id=user_id,
                query=query,
                scope=scope,
                limit=limit,
            )
            if access_error is not None:
                return access_error
        elif self._bitrix_oauth is not None:
            access_actor, access_error = await _resolve_index_access_actor(
                tool_name=self.name,
                bitrix_oauth=self._bitrix_oauth,
                user_id=user_id,
                query=query,
                scope=scope,
                limit=limit,
            )
            if access_error is not None:
                return access_error

        search_limit = min(100, max(limit, limit * 3)) if scope in _ACCESS_CHECKED_SCOPES else limit
        results = self._portal_search.search(query, entity_types=entity_types, limit=search_limit)
        access_filtered_count = 0
        if scope in _ACCESS_CHECKED_SCOPES:
            checked_results = []
            for item in results:
                if item.entity_type not in _DOCUMENT_ENTITY_TYPES:
                    access_filtered_count += 1
                    continue
                if await _document_item_is_accessible(access_client, item):
                    checked_results.append(item)
                    if len(checked_results) >= limit:
                        break
                else:
                    access_filtered_count += 1
            results = checked_results
        return ToolResult(
            status=ToolStatus.OK,
            tool="portal_search",
            data={
                "query": query,
                "scope": scope,
                "limit": limit,
                "index_path": str(stats.path),
                "access_checked": scope in _ACCESS_CHECKED_SCOPES,
                "access_actor": access_actor,
                "access_filtered_count": access_filtered_count,
                "summary": format_portal_search_results(results, query=query),
                "results": [result.as_dict() for result in results],
            },
        )


async def _resolve_index_access_actor(
    *,
    tool_name: str,
    bitrix_oauth: BitrixOAuthService,
    user_id: int | None,
    query: str,
    scope: str,
    limit: int,
) -> tuple[str, ToolResult | None]:
    if user_id is None:
        return (
            "none",
            ToolResult(
                status=ToolStatus.DENIED,
                tool=tool_name,
                error="portal_search lookup denied: current Bitrix user_id is missing.",
                data={"query": query, "scope": scope, "limit": limit},
            ),
        )

    try:
        await bitrix_oauth.client_for_user(user_id)
    except BitrixOAuthTokenMissing as exc:
        data = {"query": query, "scope": scope, "limit": limit}
        data.update(oauth_authorization_data(bitrix_oauth, user_id=exc.user_id))
        return (
            "none",
            ToolResult(
                status=ToolStatus.DENIED,
                tool=tool_name,
                error=oauth_missing_error(
                    "portal_search lookup denied",
                    user_id=exc.user_id,
                    authorization=data.get("authorization"),
                ),
                data=data,
            ),
        )
    except BitrixOAuthError as exc:
        return (
            "none",
            ToolResult(
                status=ToolStatus.ERROR,
                tool=tool_name,
                error=f"portal_search OAuth actor check failed: {exc}",
                data={"query": query, "scope": scope, "limit": limit},
            ),
        )
    return "oauth_current_user", None


async def _resolve_document_access_client(
    *,
    tool_name: str,
    fallback_client: BitrixFileDownloadPort | None,
    bitrix_oauth: BitrixOAuthService | None,
    user_id: int | None,
    query: str,
    scope: str,
    limit: int,
) -> tuple[BitrixFileDownloadPort | None, str, ToolResult | None]:
    if bitrix_oauth is None:
        if fallback_client is None:
            return (
                None,
                "none",
                ToolResult(
                    status=ToolStatus.DENIED,
                    tool=tool_name,
                    error="portal_search document/file lookup requires Bitrix live access check.",
                    data={"query": query, "scope": scope, "limit": limit},
                ),
            )
        return fallback_client, "configured_client", None

    if user_id is None:
        return (
            None,
            "none",
            ToolResult(
                status=ToolStatus.DENIED,
                tool=tool_name,
                error="portal_search document/file lookup denied: current Bitrix user_id is missing.",
                data={"query": query, "scope": scope, "limit": limit},
            ),
        )

    try:
        return await bitrix_oauth.client_for_user(user_id), "oauth_current_user", None
    except BitrixOAuthTokenMissing as exc:
        data = {"query": query, "scope": scope, "limit": limit}
        data.update(oauth_authorization_data(bitrix_oauth, user_id=exc.user_id))
        return (
            None,
            "none",
            ToolResult(
                status=ToolStatus.DENIED,
                tool=tool_name,
                error=oauth_missing_error(
                    "portal_search document/file lookup denied",
                    user_id=exc.user_id,
                    authorization=data.get("authorization"),
                ),
                data=data,
            ),
        )
    except BitrixOAuthError as exc:
        return (
            None,
            "none",
            ToolResult(
                status=ToolStatus.ERROR,
                tool=tool_name,
                error=f"portal_search document/file OAuth access check failed: {exc}",
                data={"query": query, "scope": scope, "limit": limit},
            ),
        )


async def _document_item_is_accessible(bitrix_files: BitrixFileDownloadPort | None, item: Any) -> bool:
    if bitrix_files is None:
        return False
    if item.entity_type == "disk_file":
        file_id = optional_int(item.metadata.get("disk_object_id")) or optional_int(item.entity_id)
        if file_id is None:
            return False
        try:
            await bitrix_files.get_disk_file_download_url(file_id)
        except Exception:
            return False
        return True
    if item.entity_type == "task_attachment":
        attached_id = optional_int(item.metadata.get("attached_object_id")) or optional_int(item.entity_id)
        if attached_id is None:
            return False
        try:
            attached = await bitrix_files.get_attached_object(attached_id)
        except Exception:
            return False
        return isinstance(attached, dict) and bool(attached)
    return False
