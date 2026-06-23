from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ai_server.document_text import extract_text_from_file
from ai_server.integrations.bitrix.portal_search import (
    PortalSearchIndex,
    PortalSearchResult,
    delete_portal_file_cache_path,
    portal_file_cache_path,
)
from ai_server.integrations.bitrix.ports import BitrixFileDownloadPort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.settings import Settings
from ai_server.tools.document_access.access_control import can_user_see_portal_item, filter_portal_items_for_user
from ai_server.tools.document_access.download import resolve_portal_file_download_url
from ai_server.tools.document_access.spreadsheet import (
    _direct_index_reference,
    _document_search_types,
    _is_file_item,
)
from ai_server.tools.document_access.types import ResolvedDocument
from ai_server.utils import optional_int


class DocumentReadTool:
    name = "document_read"

    def __init__(
        self,
        client: BitrixFileDownloadPort | None = None,
        *,
        portal_search: PortalSearchIndex | None = None,
        settings: Settings,
    ) -> None:
        self._client = client
        self._portal_search = portal_search
        self._settings = settings

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="document_read",
            description="Download one Bitrix document from the portal index and extract text.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "entity_id": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 1},
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
        if self._portal_search is None or self._client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="document_read",
                data={"message": "Portal search or Bitrix client not configured."},
            )
        resolved = self._resolve_document(args, user_id=user_id)
        if resolved is None:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool="document_read",
                data={"query": str(args.get("query") or ""), "candidates": []},
                error="document not found in portal index",
            )

        path: Path | None = None
        try:
            path = await self._ensure_local_document(resolved.item)
            extracted = await asyncio.to_thread(
                extract_text_from_file,
                path,
                original_name=resolved.item.title,
                max_chars=max(1, optional_int(args.get("max_chars")) or self._settings.search_content_max_chars),
            )
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool="document_read",
                error=f"{type(exc).__name__}: {exc}",
                data={"document": _document_dict(resolved.item)},
            )
        finally:
            self._delete_temp(path)

        return ToolResult(
            status=ToolStatus.OK,
            tool="document_read",
            data={
                "document": _document_dict(resolved.item),
                "text_status": extracted.status,
                "text": extracted.text,
                "reason": extracted.reason,
                "candidates": [_document_dict(item) for item in resolved.candidates],
            },
        )

    def _resolve_document(self, args: dict[str, Any], *, user_id: int | None) -> ResolvedDocument | None:
        entity_type = str(args.get("entity_type") or "").strip()
        entity_id = str(args.get("entity_id") or "").strip()
        if entity_type and entity_id:
            item = self._portal_search.get_item(entity_type=entity_type, entity_id=entity_id)
            if (
                item
                and _is_file_item(item)
                and can_user_see_portal_item(item, user_id=user_id, settings=self._settings)
            ):
                return ResolvedDocument(item=item, candidates=[item])

        query = str(args.get("query") or "").strip()
        direct = _direct_index_reference(self._portal_search, query)
        if (
            direct
            and _is_file_item(direct)
            and can_user_see_portal_item(direct, user_id=user_id, settings=self._settings)
        ):
            return ResolvedDocument(item=direct, candidates=[direct])
        if not query:
            return None

        candidates = filter_portal_items_for_user(
            [
                item
                for item in self._portal_search.search(
                    query,
                    entity_types=_document_search_types(),
                    limit=max(10, (optional_int(args.get("limit")) or 10) * 3),
                )
                if _is_file_item(item)
            ],
            user_id=user_id,
            settings=self._settings,
        )
        return ResolvedDocument(item=candidates[0], candidates=candidates[:10]) if candidates else None

    async def _ensure_local_document(self, item: PortalSearchResult) -> Path:
        path = portal_file_cache_path(item, self._settings)
        if path.exists() and path.stat().st_size > 0:
            return path
        download_url = await resolve_portal_file_download_url(self._client, item)
        if not download_url:
            raise ValueError(f"Bitrix did not return a download URL for {item.title}")
        await self._client.download_file_from_url(
            download_url,
            path,
            max_bytes=self._settings.search_content_max_bytes,
        )
        return path

    def _delete_temp(self, path: Path | None) -> None:
        if path is not None and not self._settings.search_content_keep_local_files:
            delete_portal_file_cache_path(path, self._settings)


def _document_dict(item: PortalSearchResult) -> dict[str, Any]:
    return {
        "entity_type": item.entity_type,
        "entity_id": item.entity_id,
        "title": item.title,
        "url": item.url,
        "metadata": item.metadata,
    }
