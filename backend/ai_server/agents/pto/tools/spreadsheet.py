from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

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
from ai_server.tools.document_access.formatting import format_document_comparison_report
from ai_server.tools.document_access.spreadsheet import (
    _direct_index_reference,
    _document_search_types,
    _is_file_item,
    _is_spreadsheet,
    _spreadsheet_compare_schema_from_args,
    _spreadsheet_preview,
    compare_spreadsheets_by_query,
)
from ai_server.tools.document_access.types import ResolvedDocument
from ai_server.utils import optional_int


def _document_dict(item: PortalSearchResult) -> dict[str, Any]:
    return {
        "entity_type": item.entity_type,
        "entity_id": item.entity_id,
        "title": item.title,
        "url": item.url,
        "metadata": item.metadata,
    }


class _SpreadsheetBase:
    """Shared infrastructure for spreadsheet tools."""

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
                for item in self._portal_search.search(query, entity_types=_document_search_types(), limit=30)
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
        await self._client.download_file_from_url(download_url, path, max_bytes=self._settings.search_content_max_bytes)
        return path

    def _delete_temp(self, path: Path | None) -> None:
        if path is not None and not self._settings.search_content_keep_local_files:
            delete_portal_file_cache_path(path, self._settings)


class SpreadsheetPreviewTool(_SpreadsheetBase):
    name = "spreadsheet_preview"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spreadsheet_preview",
            description=(
                "Read a small preview of a spreadsheet so the PTO LLM can choose sheet, header row, "
                "key column and value columns before exact comparison."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "entity_id": {"type": "string"},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 30},
                    "max_sheets": {"type": "integer", "minimum": 1, "maximum": 10},
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
                tool="spreadsheet_preview",
                data={"message": "Portal search or Bitrix client not configured."},
            )
        resolved = self._resolve_document(args, user_id=user_id)
        if resolved is None:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool="spreadsheet_preview",
                data={"query": str(args.get("query") or ""), "candidates": []},
                error="spreadsheet document not found in portal index",
            )
        if not _is_spreadsheet(resolved.item):
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="spreadsheet_preview",
                data={"document": _document_dict(resolved.item)},
                error=f"document is not a supported spreadsheet: {resolved.item.title}",
            )

        path: Path | None = None
        max_rows = max(1, min(optional_int(args.get("max_rows")) or 12, 30))
        max_sheets = max(1, min(optional_int(args.get("max_sheets")) or 5, 10))
        try:
            path = await self._ensure_local_document(resolved.item)
            preview = await asyncio.to_thread(
                _spreadsheet_preview, resolved.item, path, max_rows=max_rows, max_sheets=max_sheets
            )
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool="spreadsheet_preview",
                error=f"{type(exc).__name__}: {exc}",
                data={"document": _document_dict(resolved.item)},
            )
        finally:
            self._delete_temp(path)

        return ToolResult(
            status=ToolStatus.OK,
            tool="spreadsheet_preview",
            data={
                "document": _document_dict(resolved.item),
                "sheets": preview,
                "candidates": [_document_dict(item) for item in resolved.candidates],
                "note": (
                    "Choose header_row_number, key_column and value_columns from this preview before "
                    "calling spreadsheet_compare. Column references may be preview index, Excel letter, or header text."
                ),
            },
        )


class SpreadsheetCompareTool(_SpreadsheetBase):
    name = "spreadsheet_compare"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spreadsheet_compare",
            description=(
                "Compare two spreadsheet documents exactly by an explicit schema chosen by the PTO LLM "
                "after spreadsheet_preview."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "first_query": {"type": "string"},
                    "second_query": {"type": "string"},
                    "header_row_number": {"type": "integer", "minimum": 1},
                    "first_header_row_number": {"type": "integer", "minimum": 1},
                    "second_header_row_number": {"type": "integer", "minimum": 1},
                    "key_column": {
                        "description": "0-based preview index, Excel letter, or exact header text.",
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                    },
                    "value_columns": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                        "minItems": 1,
                    },
                    "first_sheet": {"type": "string"},
                    "second_sheet": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["first_query", "second_query", "key_column", "value_columns"],
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
        first_query = str(args.get("first_query") or "").strip()
        second_query = str(args.get("second_query") or "").strip()
        limit = max(1, min(optional_int(args.get("limit")) or 20, 50))
        if not first_query or not second_query:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="spreadsheet_compare",
                error="first_query and second_query are required",
            )
        schema, schema_error = _spreadsheet_compare_schema_from_args(args)
        if schema_error is not None:
            return schema_error

        _settings = self._settings
        report = await compare_spreadsheets_by_query(
            self._client,
            self._portal_search,
            first_query=first_query,
            second_query=second_query,
            schema=schema,
            limit=limit,
            item_filter=lambda item: can_user_see_portal_item(item, user_id=user_id, settings=_settings),
            settings=_settings,
        )
        return ToolResult(
            status=ToolStatus.OK if not report.errors else ToolStatus.ERROR,
            tool="spreadsheet_compare",
            data={
                "summary": format_document_comparison_report(report, limit=limit),
                "report": report.as_dict(),
                "schema": schema.as_dict(),
            },
            error="; ".join(report.errors) if report.errors else None,
        )
