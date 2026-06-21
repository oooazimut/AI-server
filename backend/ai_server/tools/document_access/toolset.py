from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_server.document_text import extract_text_from_file
from ai_server.integrations.bitrix.portal_search import (
    PortalSearchIndex,
    PortalSearchResult,
    delete_portal_file_cache_path,
    entity_types_for_scope,
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


class DocumentToolset:
    def __init__(
        self,
        client: BitrixFileDownloadPort | None = None,
        *,
        portal_search: PortalSearchIndex | None = None,
        user_id: int | None = None,
        settings: Settings,
    ) -> None:
        self.client = client
        self.portal_search = portal_search
        self.user_id = user_id
        self._settings = settings

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
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
            ),
            ToolDefinition(
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
            ),
            ToolDefinition(
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
            ),
            ToolDefinition(
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
            ),
            ToolDefinition(
                name="document_draft_create",
                description="Create a local PTO document draft from explicit LLM-provided content.",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "extension": {"type": "string", "enum": [".txt", ".md"]},
                    },
                    "required": ["title", "content"],
                },
            ),
            ToolDefinition(
                name="document_draft_list",
                description="List recent local PTO document drafts.",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                    },
                },
            ),
        ]

    def portal_document_search(self, args: dict[str, Any]) -> ToolResult:
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

        if self.portal_search is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="portal_document_search",
                data={"message": "Portal search index not configured."},
            )
        stats = self.portal_search.stats()
        if not stats.exists:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="portal_document_search",
                data={"index_path": str(stats.path), "message": "Local portal search index is missing."},
            )

        results = filter_portal_items_for_user(
            self.portal_search.search(query, entity_types=entity_types, limit=limit),
            user_id=self.user_id,
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

    async def document_read(self, args: dict[str, Any]) -> ToolResult:
        if self.portal_search is None or self.client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="document_read",
                data={"message": "Portal search or Bitrix client not configured."},
            )
        resolved = self._resolve_document(args)
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

    async def spreadsheet_preview(self, args: dict[str, Any]) -> ToolResult:
        if self.portal_search is None or self.client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="spreadsheet_preview",
                data={"message": "Portal search or Bitrix client not configured."},
            )
        resolved = self._resolve_document(args)
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
                _spreadsheet_preview,
                resolved.item,
                path,
                max_rows=max_rows,
                max_sheets=max_sheets,
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

    async def spreadsheet_compare(self, args: dict[str, Any]) -> ToolResult:
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
            self.client,
            self.portal_search,
            first_query=first_query,
            second_query=second_query,
            schema=schema,
            limit=limit,
            item_filter=lambda item: can_user_see_portal_item(item, user_id=self.user_id, settings=_settings),
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

    def document_draft_create(self, args: dict[str, Any]) -> ToolResult:
        title = str(args.get("title") or "").strip()
        content = str(args.get("content") or "").strip()
        extension = str(args.get("extension") or ".md").strip().lower()
        if extension not in {".txt", ".md"}:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="document_draft_create",
                error="extension must be .txt or .md",
            )
        if not title or not content:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="document_draft_create",
                error="title and content are required",
            )

        drafts_dir = self._settings.document_drafts_dir
        drafts_dir.mkdir(parents=True, exist_ok=True)
        path = (
            drafts_dir
            / f"{datetime.now(UTC).astimezone().strftime('%Y%m%d-%H%M%S')}-{_safe_draft_name(title, extension)}"
        )
        path.write_text(content + "\n", encoding="utf-8")
        return ToolResult(
            status=ToolStatus.OK,
            tool="document_draft_create",
            data={
                "title": title,
                "path": str(path),
                "bytes": path.stat().st_size,
                "message": "Draft was created locally; uploading/sending requires a separate approved write action.",
            },
        )

    def document_draft_list(self, args: dict[str, Any]) -> ToolResult:
        limit = max(1, min(optional_int(args.get("limit")) or 10, 30))
        drafts_dir = self._settings.document_drafts_dir
        drafts_dir.mkdir(parents=True, exist_ok=True)
        drafts = sorted(
            [path for path in drafts_dir.iterdir() if path.is_file()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:limit]
        return ToolResult(
            status=ToolStatus.OK,
            tool="document_draft_list",
            data={
                "drafts": [
                    {
                        "name": path.name,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).astimezone().isoformat(),
                    }
                    for path in drafts
                ],
                "total": len(drafts),
            },
        )

    def _resolve_document(self, args: dict[str, Any]) -> ResolvedDocument | None:
        entity_type = str(args.get("entity_type") or "").strip()
        entity_id = str(args.get("entity_id") or "").strip()
        if entity_type and entity_id:
            item = self.portal_search.get_item(entity_type=entity_type, entity_id=entity_id)
            if (
                item
                and _is_file_item(item)
                and can_user_see_portal_item(item, user_id=self.user_id, settings=self._settings)
            ):
                return ResolvedDocument(item=item, candidates=[item])

        query = str(args.get("query") or "").strip()
        direct = _direct_index_reference(self.portal_search, query)
        if (
            direct
            and _is_file_item(direct)
            and can_user_see_portal_item(direct, user_id=self.user_id, settings=self._settings)
        ):
            return ResolvedDocument(item=direct, candidates=[direct])
        if not query:
            return None

        candidates = filter_portal_items_for_user(
            [
                item
                for item in self.portal_search.search(
                    query,
                    entity_types=_document_search_types(),
                    limit=max(10, (optional_int(args.get("limit")) or 10) * 3),
                )
                if _is_file_item(item)
            ],
            user_id=self.user_id,
            settings=self._settings,
        )
        return ResolvedDocument(item=candidates[0], candidates=candidates[:10]) if candidates else None

    async def _ensure_local_document(self, item: PortalSearchResult) -> Path:
        path = portal_file_cache_path(item, self._settings)
        if path.exists() and path.stat().st_size > 0:
            return path

        download_url = await resolve_portal_file_download_url(self.client, item)
        if not download_url:
            raise ValueError(f"Bitrix did not return a download URL for {item.title}")
        await self.client.download_file_from_url(
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


def _safe_draft_name(title: str, extension: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9а-яА-Я._-]+", "_", title).strip("._")
    if not name:
        name = "draft"
    if not name.lower().endswith(extension):
        name += extension
    return name[:120]
