from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_server.integrations.bitrix.portal_search import (
    PortalSearchResult,
    delete_portal_file_cache_path,
    portal_file_cache_path,
)
from ai_server.settings import Settings
from ai_server.tools.bitrix_ports import BitrixFileDownloadPort
from ai_server.tools.document_access.download import resolve_portal_file_download_url
from ai_server.tools.document_access.types import ResolvedDocument


class BaseDocumentTool:
    """Common infrastructure for all PTO document tools.

    Subclasses provide specific document operations (read, preview, compare).
    Document resolution is handled externally: tools receive explicit entity_type + entity_id
    from task context (provided by orchestrator via Bitrix24 specialist).
    """

    def __init__(
        self,
        client: BitrixFileDownloadPort | None = None,
        *,
        settings: Settings,
    ) -> None:
        self._client = client
        self._settings = settings

    def _resolve_document(self, args: dict[str, Any], *, user_id: int | None) -> ResolvedDocument | None:
        """PTO tools will be refactored to receive documents via task context."""
        return None

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
