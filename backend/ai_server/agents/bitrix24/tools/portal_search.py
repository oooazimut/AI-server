from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_server.agents.bitrix24.tools.read_client import oauth_authorization_data, oauth_missing_error
from ai_server.integrations.bitrix.oauth import BitrixOAuthError, BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.integrations.bitrix.portal_search.types import PortalSearchResult
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_ports import BitrixFileDownloadPort
from ai_server.tools.bitrix_search import PortalSearchPort, entity_types_for_scope, format_portal_search_results
from ai_server.utils import optional_int

_DENIED_AGENT_SCOPES = {"", "all", "tasks"}
_ACCESS_CHECKED_SCOPES = {"documents", "files"}
_DOCUMENT_ENTITY_TYPES = {"disk_file", "task_attachment"}
_PAGINATION_FIELD = "portal_search_page"
_DEFAULT_PAGE_SIZE = 10
_SHOW_ALL_LIMIT = 50
_MAX_SEARCH_RESULTS = 500
_LIVE_MAX_STORAGES = 20
_LIVE_MAX_FOLDERS = 50
_LIVE_MAX_ITEMS = 500
_LIVE_MAX_DEPTH = 5


class PortalSearchTool:
    name = "portal_search"

    def __init__(
        self,
        portal_search: PortalSearchPort | None = None,
        bitrix_files: BitrixFileDownloadPort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
        state_store: Any | None = None,
        live_fallback_enabled: bool = False,
        index_max_age_seconds: int | None = None,
        index_freshness_path: Path | None = None,
    ) -> None:
        self._portal_search = portal_search
        self._bitrix_files = bitrix_files
        self._bitrix_oauth = bitrix_oauth
        self._state_store = state_store
        self._live_fallback_enabled = live_fallback_enabled
        self._index_max_age_seconds = index_max_age_seconds
        self._index_freshness_path = index_freshness_path

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
                    "limit": {"type": "integer", "minimum": 1, "maximum": _SHOW_ALL_LIMIT},
                    "offset": {"type": "integer", "minimum": 0},
                    "show_all": {"type": "boolean"},
                    "continuation": {"type": "string", "enum": ["next"]},
                },
                "anyOf": [{"required": ["query"]}, {"required": ["continuation"]}],
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
        continuation = str(args.get("continuation") or "").strip().lower()
        page_state: dict[str, Any] | None = None
        if continuation:
            if continuation != "next":
                return ToolResult(
                    status=ToolStatus.INVALID_TOOL_CALL,
                    tool=self.name,
                    error=f"unknown continuation: {continuation}",
                )
            page_state = await _load_pagination_state(self._state_store, dialog_key=dialog_key)
            if page_state is None:
                return ToolResult(
                    status=ToolStatus.INVALID_TOOL_CALL,
                    tool=self.name,
                    error="portal_search continuation requires an active dialog-bound result page.",
                )

        query = str((page_state or {}).get("query") or args.get("query") or "").strip()
        scope = str((page_state or {}).get("scope") or args.get("scope") or "all").strip().lower()
        show_all = bool(args.get("show_all")) and page_state is None
        requested_limit = (page_state or {}).get("limit") or args.get("limit") or _DEFAULT_PAGE_SIZE
        limit = max(1, min(int(requested_limit), _SHOW_ALL_LIMIT))
        if show_all:
            limit = _SHOW_ALL_LIMIT
        offset = max(0, int((page_state or {}).get("next_offset") or args.get("offset") or 0))
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
        if not stats.exists and not (scope in _ACCESS_CHECKED_SCOPES and self._live_fallback_enabled):
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="portal_search",
                data={
                    "query": query,
                    "scope": scope,
                    "limit": limit,
                    "offset": offset,
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

        index_state, index_age_seconds, index_freshness_source = _index_state(
            stats,
            max_age_seconds=self._index_max_age_seconds,
            freshness_path=self._index_freshness_path,
        )
        search_limit = (
            _MAX_SEARCH_RESULTS
            if scope in _ACCESS_CHECKED_SCOPES
            else min(_MAX_SEARCH_RESULTS, max(limit + offset, limit))
        )
        results = (
            self._portal_search.search(query, entity_types=entity_types, limit=search_limit) if stats.exists else []
        )
        access_filtered_count = 0
        if scope in _ACCESS_CHECKED_SCOPES:
            checked_results = []
            for item in results:
                if item.entity_type not in _DOCUMENT_ENTITY_TYPES:
                    access_filtered_count += 1
                    continue
                if await _document_item_is_accessible(access_client, item):
                    checked_results.append(item)
                else:
                    access_filtered_count += 1
            results = checked_results

        live_attempted = False
        live_error = ""
        live_results: list[PortalSearchResult] = []
        live_required = index_state in {"missing", "stale"}
        live_useful = live_required or not results
        if (
            scope in _ACCESS_CHECKED_SCOPES
            and self._live_fallback_enabled
            and live_useful
            and access_client is not None
            and access_actor == "oauth_current_user"
        ):
            live_attempted = True
            try:
                live_results = await _search_live_disk_current_user(access_client, query=query)
            except Exception as exc:
                live_error = f"{type(exc).__name__}: {exc}"
            else:
                _update_index_from_live_results(self._portal_search, live_results)

        stale_results_suppressed = 0
        source_mode = "bitrix_postgresql"
        if live_results:
            if live_required:
                stale_results_suppressed = len(results)
            results = live_results
            source_mode = "bitrix_live_current_user"
        elif live_required:
            if live_attempted and not live_error:
                source_mode = "bitrix_live_current_user"
            stale_results_suppressed = len(results)
            results = []
            if live_error:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="portal_search current-user live verification failed; stale or missing index results were suppressed.",
                    data={
                        "query": query,
                        "scope": scope,
                        "limit": limit,
                        "offset": offset,
                        "index_state": index_state,
                        "index_age_seconds": index_age_seconds,
                        "index_freshness_source": index_freshness_source,
                        "access_checked": True,
                        "access_actor": access_actor,
                        "live_attempted": True,
                        "live_error": live_error,
                        "stale_results_suppressed": stale_results_suppressed,
                        "results": [],
                    },
                )

        results = sorted(results, key=_stable_result_key)
        total = len(results)
        page = results[offset : offset + limit]
        shown = len(page)
        next_offset = offset + shown
        has_more = next_offset < total
        remaining = max(0, total - next_offset)
        pages = math.ceil(total / limit) if total else 0
        output_results = []
        for result in page:
            item = result.as_dict()
            item["source"] = source_mode
            output_results.append(item)

        if has_more:
            await _save_pagination_state(
                self._state_store,
                dialog_key=dialog_key,
                query=query,
                scope=scope,
                limit=limit,
                next_offset=next_offset,
            )
        else:
            await _clear_pagination_state(self._state_store, dialog_key=dialog_key)
        return ToolResult(
            status=ToolStatus.OK,
            tool="portal_search",
            data={
                "query": query,
                "scope": scope,
                "limit": limit,
                "offset": offset,
                "total": total,
                "shown": shown,
                "range_start": offset + 1 if shown else 0,
                "range_end": next_offset,
                "remaining": remaining,
                "pages": pages,
                "has_more": has_more,
                "next_offset": next_offset if has_more else None,
                "index_path": str(stats.path),
                "index_state": index_state,
                "index_age_seconds": index_age_seconds,
                "index_freshness_source": index_freshness_source,
                "access_checked": scope in _ACCESS_CHECKED_SCOPES,
                "access_actor": access_actor,
                "access_filtered_count": access_filtered_count,
                "source_mode": source_mode,
                "live_attempted": live_attempted,
                "live_error": live_error,
                "stale_results_suppressed": stale_results_suppressed,
                "summary": format_portal_search_results(page, query=query),
                "results": output_results,
            },
        )


def _index_state(
    stats: Any,
    *,
    max_age_seconds: int | None,
    freshness_path: Path | None,
) -> tuple[str, int | None, str]:
    if not bool(getattr(stats, "exists", False)):
        return "missing", None, "index_missing"
    if max_age_seconds is None:
        return "fresh", None, "freshness_not_required"
    value = ""
    source = "indexed_item_max"
    if freshness_path is not None:
        source = "indexer_state_missing_success"
        try:
            state = json.loads(freshness_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return "stale", None, source
        if not isinstance(state, dict):
            return "stale", None, source
        if int(state.get("consecutive_errors") or 0) > 0 or state.get("last_error"):
            return "stale", None, "indexer_state_error"
        candidates = [
            ("last_delta_sync_at", state.get("last_delta_sync_at")),
            ("last_metadata_sync_at", state.get("last_metadata_sync_at")),
        ]
        parsed_candidates: list[tuple[datetime, str]] = []
        for key, candidate in candidates:
            parsed = _parse_timestamp(candidate)
            if parsed is not None:
                parsed_candidates.append((parsed, key))
        if not parsed_candidates:
            return "stale", None, source
        newest, key = max(parsed_candidates, key=lambda item: item[0])
        value = newest.isoformat()
        source = f"indexer_state:{key}"
    else:
        value = str(getattr(stats, "last_indexed_at", None) or "").strip()
    if not value:
        return "stale", None, source
    parsed = _parse_timestamp(value)
    if parsed is None:
        return "stale", None, source
    age = max(0, int((datetime.now(UTC) - parsed).total_seconds()))
    return ("stale" if age > max_age_seconds else "fresh"), age, source


def _parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None
    return parsed.astimezone(UTC)


def _stable_result_key(item: PortalSearchResult) -> tuple[int, str, str, str]:
    return (-int(item.score), str(item.entity_type), str(item.title).casefold(), str(item.entity_id))


async def _load_pagination_state(store: Any | None, *, dialog_key: str | None) -> dict[str, Any] | None:
    if store is None or not dialog_key:
        return None
    raw = await store.get_kv(dialog_key, _PAGINATION_FIELD)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        await store.delete_kv(dialog_key, _PAGINATION_FIELD)
        return None
    if not isinstance(value, dict) or not value.get("query") or not value.get("scope"):
        await store.delete_kv(dialog_key, _PAGINATION_FIELD)
        return None
    return value


async def _save_pagination_state(
    store: Any | None,
    *,
    dialog_key: str | None,
    query: str,
    scope: str,
    limit: int,
    next_offset: int,
) -> None:
    if store is None or not dialog_key:
        return
    value = json.dumps(
        {"query": query, "scope": scope, "limit": limit, "next_offset": next_offset},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(value) <= 256:
        await store.set_kv(dialog_key, _PAGINATION_FIELD, value)
    else:
        await store.delete_kv(dialog_key, _PAGINATION_FIELD)


async def _clear_pagination_state(store: Any | None, *, dialog_key: str | None) -> None:
    if store is not None and dialog_key:
        await store.delete_kv(dialog_key, _PAGINATION_FIELD)


async def _search_live_disk_current_user(
    client: Any,
    *,
    query: str,
) -> list[PortalSearchResult]:
    list_storages = getattr(client, "list_disk_storages", None)
    list_children = getattr(client, "list_disk_folder_children_all", None)
    if not callable(list_storages) or not callable(list_children):
        raise RuntimeError("current-user Bitrix client does not support live disk search")

    storages = await list_storages(limit=_LIVE_MAX_STORAGES)
    terms = [term for term in query.casefold().split() if term]
    queue: list[tuple[int, str, str, int]] = []
    for storage in storages:
        root_id = optional_int(storage.get("ROOT_OBJECT_ID") or storage.get("rootObjectId"))
        if root_id is None:
            continue
        storage_name = str(storage.get("NAME") or storage.get("name") or f"Disk #{root_id}")
        queue.append((root_id, storage_name, storage_name, 0))

    matches: list[PortalSearchResult] = []
    visited: set[int] = set()
    seen_objects: set[str] = set()
    folders_scanned = 0
    items_seen = 0
    while queue and folders_scanned < _LIVE_MAX_FOLDERS and items_seen < _LIVE_MAX_ITEMS:
        folder_id, storage_name, path, depth = queue.pop(0)
        if folder_id in visited:
            continue
        visited.add(folder_id)
        folders_scanned += 1
        children = await list_children(
            folder_id=folder_id,
            limit=min(100, _LIVE_MAX_ITEMS - items_seen),
        )
        for child in children:
            items_seen += 1
            item_id = child.get("ID") or child.get("id")
            if item_id is None:
                continue
            object_key = str(item_id)
            if object_key in seen_objects:
                continue
            seen_objects.add(object_key)
            name = str(child.get("NAME") or child.get("name") or f"Object #{item_id}")
            item_type = str(child.get("TYPE") or child.get("type") or "").casefold()
            child_path = f"{path}/{name}"
            if item_type == "folder":
                child_id = optional_int(item_id)
                if child_id is not None and depth < _LIVE_MAX_DEPTH:
                    queue.append((child_id, storage_name, child_path, depth + 1))
                continue
            searchable = f"{name} {child_path}".casefold()
            if terms and not all(term in searchable for term in terms):
                continue
            metadata = {
                "type": item_type or "file",
                "path": child_path,
                "storage_name": storage_name,
                "parent_id": folder_id,
                "disk_object_id": item_id,
                "live_current_user_verified": True,
            }
            matches.append(
                PortalSearchResult(
                    entity_type="disk_file",
                    entity_id=str(item_id),
                    title=name,
                    body=f"Disk: {storage_name}\nPath: {child_path}",
                    url=str(child.get("DETAIL_URL") or child.get("detailUrl") or ""),
                    score=max(1, sum(10 for term in terms if term in name.casefold())),
                    metadata=metadata,
                )
            )
            if len(matches) >= _MAX_SEARCH_RESULTS:
                return sorted(matches, key=_stable_result_key)
            if items_seen >= _LIVE_MAX_ITEMS:
                break
    return sorted(matches, key=_stable_result_key)


def _update_index_from_live_results(index: Any, results: list[PortalSearchResult]) -> None:
    upsert = getattr(index, "upsert_item", None)
    if not callable(upsert):
        return
    for item in results:
        upsert(
            entity_type=item.entity_type,
            entity_id=item.entity_id,
            title=item.title,
            body=item.body,
            url=item.url,
            metadata=item.metadata,
            preserve_content=True,
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
