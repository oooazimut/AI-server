from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ai_server.document_text import extract_text_from_file
from ai_server.integrations.bitrix.ports import BitrixFileDownloadPort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.settings import Settings
from ai_server.utils import optional_int

from .base import BaseDocumentTool, _document_dict


class DocumentReadTool(BaseDocumentTool):
    name = "document_read"

    def __init__(
        self,
        client: BitrixFileDownloadPort | None = None,
        *,
        settings: Settings,
    ) -> None:
        super().__init__(client, settings=settings)

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
        if self._client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="document_read",
                data={"message": "Bitrix client not configured."},
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
