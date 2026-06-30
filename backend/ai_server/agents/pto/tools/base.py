from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_server.settings import Settings
from ai_server.tools.bitrix_ports import BitrixFileDownloadPort
from ai_server.tools.document_access.download import delete_portal_file_cache_path, ensure_local_document
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

    async def _ensure_local_document(self, item: Any) -> Path:
        assert self._client is not None
        return await ensure_local_document(
            self._client,
            item,
            max_bytes=self._settings.search_content_max_bytes,
            settings=self._settings,
        )

    def _delete_temp(self, path: Path | None) -> None:
        if path is not None and not self._settings.search_content_keep_local_files:
            delete_portal_file_cache_path(path, self._settings)


def _document_dict(item: Any) -> dict[str, Any]:
    return {
        "entity_type": item.entity_type,
        "entity_id": item.entity_id,
        "title": item.title,
        "url": item.url,
        "metadata": item.metadata,
    }
