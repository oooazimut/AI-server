from __future__ import annotations

import asyncio
import csv
import math
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ai_server.document_text import extract_text_from_file
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search import (
    PortalSearchIndex,
    PortalSearchResult,
    delete_portal_file_cache_path,
    entity_types_for_scope,
    portal_file_cache_path,
)
from ai_server.models import ToolDefinition, ToolResult
from ai_server.settings import get_settings
from ai_server.utils import optional_int

SPREADSHEET_EXTENSIONS = {".csv", ".xls", ".xlsx"}
VALUE_FIELD_PRIORITY = {"цена": 10, "стоимость": 20, "количество": 30}


@dataclass(frozen=True)
class ResolvedDocument:
    item: PortalSearchResult
    candidates: list[PortalSearchResult]


@dataclass(frozen=True)
class CellValue:
    label: str
    raw: str
    number: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "raw": self.raw, "number": self.number}


@dataclass(frozen=True)
class SpreadsheetEntry:
    key: str
    normalized_key: str
    values: dict[str, CellValue]
    sheet: str
    row_number: int


@dataclass(frozen=True)
class ComparedDocument:
    entity_type: str
    entity_id: str
    title: str
    url: str
    rows: int
    value_fields: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "title": self.title,
            "url": self.url,
            "rows": self.rows,
            "value_fields": self.value_fields,
        }


@dataclass(frozen=True)
class FieldDifference:
    field: str
    first: CellValue
    second: CellValue
    delta: float | None = None
    percent: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "first": self.first.as_dict(),
            "second": self.second.as_dict(),
            "delta": self.delta,
            "percent": self.percent,
        }


@dataclass(frozen=True)
class RowDifference:
    key: str
    first_sheet: str
    first_row: int
    second_sheet: str
    second_row: int
    fields: list[FieldDifference]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "first_sheet": self.first_sheet,
            "first_row": self.first_row,
            "second_sheet": self.second_sheet,
            "second_row": self.second_row,
            "fields": [field.as_dict() for field in self.fields],
        }


@dataclass(frozen=True)
class MissingEntry:
    key: str
    sheet: str
    row_number: int

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "sheet": self.sheet, "row_number": self.row_number}


@dataclass
class DocumentCompareReport:
    first_query: str
    second_query: str
    first: ComparedDocument | None = None
    second: ComparedDocument | None = None
    changed: list[RowDifference] = field(default_factory=list)
    only_first: list[MissingEntry] = field(default_factory=list)
    only_second: list[MissingEntry] = field(default_factory=list)
    common_rows: int = 0
    duplicate_keys_first: int = 0
    duplicate_keys_second: int = 0
    errors: list[str] = field(default_factory=list)
    first_candidates: list[PortalSearchResult] = field(default_factory=list)
    second_candidates: list[PortalSearchResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "first_query": self.first_query,
            "second_query": self.second_query,
            "first": self.first.as_dict() if self.first else None,
            "second": self.second.as_dict() if self.second else None,
            "changed": [item.as_dict() for item in self.changed],
            "only_first": [item.as_dict() for item in self.only_first],
            "only_second": [item.as_dict() for item in self.only_second],
            "common_rows": self.common_rows,
            "duplicate_keys_first": self.duplicate_keys_first,
            "duplicate_keys_second": self.duplicate_keys_second,
            "errors": self.errors,
            "first_candidates": [_candidate_dict(item) for item in self.first_candidates],
            "second_candidates": [_candidate_dict(item) for item in self.second_candidates],
        }


@dataclass(frozen=True)
class SheetRows:
    name: str
    rows: list[tuple[int, list[str]]]


@dataclass(frozen=True)
class SpreadsheetDataset:
    document: ComparedDocument
    entries: dict[str, SpreadsheetEntry]
    duplicates: int


@dataclass(frozen=True)
class SpreadsheetCompareSchema:
    header_row_number: int
    key_column: object
    value_columns: list[object]
    first_sheet: str = ""
    second_sheet: str = ""
    first_header_row_number: int | None = None
    second_header_row_number: int | None = None

    def first_header(self) -> int:
        return self.first_header_row_number or self.header_row_number

    def second_header(self) -> int:
        return self.second_header_row_number or self.header_row_number

    def as_dict(self) -> dict[str, Any]:
        return {
            "header_row_number": self.header_row_number,
            "first_header_row_number": self.first_header_row_number,
            "second_header_row_number": self.second_header_row_number,
            "key_column": self.key_column,
            "value_columns": self.value_columns,
            "first_sheet": self.first_sheet,
            "second_sheet": self.second_sheet,
        }


class DocumentToolset:
    def __init__(
        self,
        client: BitrixClient | None = None,
        *,
        portal_search: PortalSearchIndex | None = None,
        user_id: int | None = None,
    ) -> None:
        self.client = client or BitrixClient()
        self.portal_search = portal_search or PortalSearchIndex()
        self.user_id = user_id

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
            return ToolResult(status="invalid_tool_call", tool="portal_document_search", error="query is required")

        entity_types = entity_types_for_scope(scope)
        if entity_types is None or not entity_types <= entity_types_for_scope("files"):
            return ToolResult(
                status="invalid_tool_call",
                tool="portal_document_search",
                error=f"unknown document scope: {scope}",
            )

        stats = self.portal_search.stats()
        if not stats.exists:
            return ToolResult(
                status="not_configured",
                tool="portal_document_search",
                data={"index_path": str(stats.path), "message": "Local portal search index is missing."},
            )

        results = filter_portal_items_for_user(
            self.portal_search.search(query, entity_types=entity_types, limit=limit),
            user_id=self.user_id,
        )
        return ToolResult(
            status="ok",
            tool="portal_document_search",
            data={
                "query": query,
                "scope": scope,
                "results": [item.as_dict() for item in results],
                "total": len(results),
            },
        )

    async def document_read(self, args: dict[str, Any]) -> ToolResult:
        resolved = self._resolve_document(args)
        if resolved is None:
            return ToolResult(
                status="not_found",
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
                max_chars=max(1, optional_int(args.get("max_chars")) or get_settings().search_content_max_chars),
            )
        except Exception as exc:
            return ToolResult(
                status="error",
                tool="document_read",
                error=f"{type(exc).__name__}: {exc}",
                data={"document": _document_dict(resolved.item)},
            )
        finally:
            self._delete_temp(path)

        return ToolResult(
            status="ok",
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
        resolved = self._resolve_document(args)
        if resolved is None:
            return ToolResult(
                status="not_found",
                tool="spreadsheet_preview",
                data={"query": str(args.get("query") or ""), "candidates": []},
                error="spreadsheet document not found in portal index",
            )
        if not _is_spreadsheet(resolved.item):
            return ToolResult(
                status="invalid_tool_call",
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
                status="error",
                tool="spreadsheet_preview",
                error=f"{type(exc).__name__}: {exc}",
                data={"document": _document_dict(resolved.item)},
            )
        finally:
            self._delete_temp(path)

        return ToolResult(
            status="ok",
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
                status="invalid_tool_call",
                tool="spreadsheet_compare",
                error="first_query and second_query are required",
            )
        schema, schema_error = _spreadsheet_compare_schema_from_args(args)
        if schema_error is not None:
            return schema_error

        report = await compare_spreadsheets_by_query(
            self.client,
            self.portal_search,
            first_query=first_query,
            second_query=second_query,
            schema=schema,
            limit=limit,
            item_filter=lambda item: can_user_see_portal_item(item, user_id=self.user_id),
        )
        return ToolResult(
            status="ok" if not report.errors else "error",
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
                status="invalid_tool_call",
                tool="document_draft_create",
                error="extension must be .txt or .md",
            )
        if not title or not content:
            return ToolResult(
                status="invalid_tool_call",
                tool="document_draft_create",
                error="title and content are required",
            )

        drafts_dir = get_settings().document_drafts_dir
        drafts_dir.mkdir(parents=True, exist_ok=True)
        path = (
            drafts_dir
            / f"{datetime.now(UTC).astimezone().strftime('%Y%m%d-%H%M%S')}-{_safe_draft_name(title, extension)}"
        )
        path.write_text(content + "\n", encoding="utf-8")
        return ToolResult(
            status="ok",
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
        drafts_dir = get_settings().document_drafts_dir
        drafts_dir.mkdir(parents=True, exist_ok=True)
        drafts = sorted(
            [path for path in drafts_dir.iterdir() if path.is_file()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:limit]
        return ToolResult(
            status="ok",
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
            if item and _is_file_item(item) and can_user_see_portal_item(item, user_id=self.user_id):
                return ResolvedDocument(item=item, candidates=[item])

        query = str(args.get("query") or "").strip()
        direct = _direct_index_reference(self.portal_search, query)
        if direct and _is_file_item(direct) and can_user_see_portal_item(direct, user_id=self.user_id):
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
        )
        return ResolvedDocument(item=candidates[0], candidates=candidates[:10]) if candidates else None

    async def _ensure_local_document(self, item: PortalSearchResult) -> Path:
        path = portal_file_cache_path(item)
        if path.exists() and path.stat().st_size > 0:
            return path

        download_url = await resolve_portal_file_download_url(self.client, item)
        if not download_url:
            raise ValueError(f"Bitrix did not return a download URL for {item.title}")
        await self.client.download_file_from_url(
            download_url,
            path,
            max_bytes=get_settings().search_content_max_bytes,
        )
        return path

    def _delete_temp(self, path: Path | None) -> None:
        if path is not None and not get_settings().search_content_keep_local_files:
            delete_portal_file_cache_path(path)


async def compare_spreadsheets_by_query(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    first_query: str,
    second_query: str,
    schema: SpreadsheetCompareSchema,
    limit: int = 20,
    item_filter: Callable[[PortalSearchResult], bool] | None = None,
) -> DocumentCompareReport:
    report = DocumentCompareReport(first_query=first_query.strip(), second_query=second_query.strip())
    if not report.first_query or not report.second_query:
        report.errors.append("Для сравнения нужны два поисковых запроса или два файла.")
        return report

    first_item = _find_spreadsheet_document(index, report.first_query, item_filter=item_filter)
    if first_item is None:
        report.errors.append(f"Не нашёл табличный документ по первому запросу: {report.first_query}")
        report.first_candidates = _spreadsheet_candidates(index, report.first_query, limit=8, item_filter=item_filter)
        return report

    second_item = _find_spreadsheet_document(
        index,
        report.second_query,
        exclude={(first_item.entity_type, str(first_item.entity_id))},
        item_filter=item_filter,
    )
    if second_item is None:
        report.errors.append(f"Не нашёл табличный документ по второму запросу: {report.second_query}")
        report.second_candidates = _spreadsheet_candidates(index, report.second_query, limit=8, item_filter=item_filter)
        return report

    first_path: Path | None = None
    second_path: Path | None = None
    try:
        first_path = await _ensure_local_document(bitrix, first_item)
        second_path = await _ensure_local_document(bitrix, second_item)
        first_dataset, second_dataset = await asyncio.gather(
            asyncio.to_thread(
                _extract_spreadsheet_dataset_with_schema,
                first_item,
                first_path,
                sheet_name=schema.first_sheet,
                header_row_number=schema.first_header(),
                key_column=schema.key_column,
                value_columns=schema.value_columns,
            ),
            asyncio.to_thread(
                _extract_spreadsheet_dataset_with_schema,
                second_item,
                second_path,
                sheet_name=schema.second_sheet,
                header_row_number=schema.second_header(),
                key_column=schema.key_column,
                value_columns=schema.value_columns,
            ),
        )
    except Exception as exc:
        report.errors.append(f"Не смог разобрать документы: {type(exc).__name__}: {exc}")
        return report
    finally:
        if not get_settings().search_content_keep_local_files:
            for path in (first_path, second_path):
                if path is not None:
                    delete_portal_file_cache_path(path)

    compared = _compare_datasets(
        first_dataset, second_dataset, first_query=report.first_query, second_query=report.second_query
    )
    compared.changed = compared.changed[:limit]
    compared.only_first = compared.only_first[:limit]
    compared.only_second = compared.only_second[:limit]
    return compared


def format_document_comparison_report(report: DocumentCompareReport, *, limit: int = 20) -> str:
    if report.errors:
        lines = list(report.errors)
        candidates = report.first_candidates or report.second_candidates
        if candidates:
            lines.extend(["", "Похожие табличные документы:"])
            lines.extend(f"- {_document_link(item)} ({item.entity_type} #{item.entity_id})" for item in candidates[:8])
        return "\n".join(lines)
    if not report.first or not report.second:
        return "Не смог сравнить документы: не определил оба файла."

    lines = [
        "Сравнил документы:",
        f"1. {_format_document_link(report.first.title, report.first.url)}",
        f"2. {_format_document_link(report.second.title, report.second.url)}",
        "",
        f"Совпавших позиций: {report.common_rows}",
        f"Отличий по значениям: {len(report.changed)}",
        f"Только в первом документе: {len(report.only_first)}",
        f"Только во втором документе: {len(report.only_second)}",
    ]
    if report.changed:
        lines.extend(["", "Изменившиеся позиции:"])
        for difference in report.changed[:limit]:
            lines.append(
                f"- {difference.key} "
                f"({difference.first_sheet}!{difference.first_row} -> {difference.second_sheet}!{difference.second_row})"
            )
            for field in difference.fields:
                lines.append(f"  {field.first.label}: {_format_field_difference(field)}")
    if report.only_first:
        lines.extend(["", "Есть только в первом документе:"])
        lines.extend(f"- {entry.key} ({entry.sheet}!{entry.row_number})" for entry in report.only_first[:limit])
    if report.only_second:
        lines.extend(["", "Есть только во втором документе:"])
        lines.extend(f"- {entry.key} ({entry.sheet}!{entry.row_number})" for entry in report.only_second[:limit])
    if not report.changed and not report.only_first and not report.only_second:
        lines.extend(["", "По распознанным табличным позициям различий не нашёл."])
    return "\n".join(lines)


async def resolve_portal_file_download_url(bitrix: BitrixClient, item: PortalSearchResult) -> str | None:
    if item.entity_type == "task_attachment":
        attached_object_id = optional_int(item.metadata.get("attached_object_id")) or optional_int(item.entity_id)
        if attached_object_id is None:
            return None
        attached = await bitrix.get_attached_object(attached_object_id)
        if isinstance(attached, dict):
            return str(_first(attached, "DOWNLOAD_URL", "downloadUrl") or "")
        return None

    if item.entity_type == "disk_file":
        disk_file_id = optional_int(item.metadata.get("disk_object_id")) or optional_int(item.entity_id)
        if disk_file_id is None:
            return None
        return await bitrix.get_disk_file_download_url(disk_file_id)
    return None


def user_has_private_disk_restrictions(user_id: int | None) -> bool:
    return user_id is not None and user_id in get_settings().resolved_agent_private_disk_restricted_user_ids


def is_private_disk_item(item: Any) -> bool:
    entity_type = str(getattr(item, "entity_type", "") or _dict_value(item, "entity_type") or "").lower()
    if entity_type not in {"disk_file", "disk_folder", "task_attachment"}:
        return False
    metadata = getattr(item, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = _dict_value(item, "metadata") or {}
    path = str(metadata.get("path") or "")
    if _path_has_private_marker(path):
        return True
    title = str(getattr(item, "title", "") or _dict_value(item, "title") or "")
    return entity_type == "disk_folder" and _matches_private_marker(title)


def can_user_see_portal_item(item: Any, *, user_id: int | None) -> bool:
    return not user_has_private_disk_restrictions(user_id) or not is_private_disk_item(item)


def filter_portal_items_for_user(items: list[Any], *, user_id: int | None) -> list[Any]:
    return (
        items
        if not user_has_private_disk_restrictions(user_id)
        else [item for item in items if not is_private_disk_item(item)]
    )


async def _ensure_local_document(bitrix: BitrixClient, item: PortalSearchResult) -> Path:
    path = portal_file_cache_path(item)
    if path.exists() and path.stat().st_size > 0:
        return path
    download_url = await resolve_portal_file_download_url(bitrix, item)
    if not download_url:
        raise ValueError(f"Bitrix не вернул ссылку на скачивание: {item.title}")
    await bitrix.download_file_from_url(download_url, path, max_bytes=get_settings().search_content_max_bytes)
    return path


def _find_spreadsheet_document(
    index: PortalSearchIndex,
    query: str,
    *,
    exclude: set[tuple[str, str]] | None = None,
    item_filter: Callable[[PortalSearchResult], bool] | None = None,
) -> PortalSearchResult | None:
    direct = _direct_index_reference(index, query)
    if (
        direct
        and _is_spreadsheet(direct)
        and (direct.entity_type, str(direct.entity_id)) not in (exclude or set())
        and (item_filter is None or item_filter(direct))
    ):
        return direct
    for item in _spreadsheet_candidates(index, query, limit=12, item_filter=item_filter):
        if (item.entity_type, str(item.entity_id)) not in (exclude or set()):
            return item
    return None


def _spreadsheet_candidates(
    index: PortalSearchIndex,
    query: str,
    *,
    limit: int,
    item_filter: Callable[[PortalSearchResult], bool] | None = None,
) -> list[PortalSearchResult]:
    candidates: list[PortalSearchResult] = []
    seen: set[tuple[str, str]] = set()
    for item in index.search(query or "смет", entity_types=_document_search_types(), limit=max(limit * 3, 10)):
        if not _is_spreadsheet(item) or (item_filter is not None and not item_filter(item)):
            continue
        key = (item.entity_type, str(item.entity_id))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)
        if len(candidates) >= limit:
            break
    return candidates


def _spreadsheet_compare_schema_from_args(args: dict[str, Any]) -> tuple[SpreadsheetCompareSchema, ToolResult | None]:
    header_row_number = optional_int(args.get("header_row_number"))
    first_header_row_number = optional_int(args.get("first_header_row_number"))
    second_header_row_number = optional_int(args.get("second_header_row_number"))
    if header_row_number is None:
        if first_header_row_number is None or second_header_row_number is None:
            return _empty_schema(), ToolResult(
                status="contract_violation",
                tool="spreadsheet_compare",
                error=(
                    "spreadsheet_compare requires header_row_number, or both first_header_row_number "
                    "and second_header_row_number. Call spreadsheet_preview first and let the PTO LLM choose it."
                ),
            )
        header_row_number = first_header_row_number
    if (
        header_row_number <= 0
        or (first_header_row_number is not None and first_header_row_number <= 0)
        or (second_header_row_number is not None and second_header_row_number <= 0)
    ):
        return _empty_schema(), ToolResult(
            status="invalid_tool_call",
            tool="spreadsheet_compare",
            error="header row numbers must be positive",
        )

    key_column = args.get("key_column")
    if not _has_column_reference(key_column):
        return _empty_schema(), ToolResult(
            status="contract_violation",
            tool="spreadsheet_compare",
            error="spreadsheet_compare requires key_column selected by the PTO LLM from spreadsheet_preview.",
        )

    raw_value_columns = args.get("value_columns")
    if isinstance(raw_value_columns, list):
        value_columns = [value for value in raw_value_columns if _has_column_reference(value)]
    elif _has_column_reference(raw_value_columns):
        value_columns = [raw_value_columns]
    else:
        value_columns = []
    if not value_columns:
        return _empty_schema(), ToolResult(
            status="contract_violation",
            tool="spreadsheet_compare",
            error="spreadsheet_compare requires at least one value_columns entry selected by the PTO LLM from spreadsheet_preview.",
        )

    return (
        SpreadsheetCompareSchema(
            header_row_number=header_row_number,
            first_header_row_number=first_header_row_number,
            second_header_row_number=second_header_row_number,
            key_column=key_column,
            value_columns=value_columns,
            first_sheet=str(args.get("first_sheet") or "").strip(),
            second_sheet=str(args.get("second_sheet") or "").strip(),
        ),
        None,
    )


def _empty_schema() -> SpreadsheetCompareSchema:
    return SpreadsheetCompareSchema(header_row_number=1, key_column="", value_columns=[])


def _has_column_reference(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) or bool(str(value or "").strip())


def _compare_datasets(
    first: SpreadsheetDataset,
    second: SpreadsheetDataset,
    *,
    first_query: str,
    second_query: str,
) -> DocumentCompareReport:
    first_keys = set(first.entries)
    second_keys = set(second.entries)
    common_keys = first_keys & second_keys
    changed: list[RowDifference] = []
    for key in sorted(common_keys):
        first_entry = first.entries[key]
        second_entry = second.entries[key]
        fields: list[FieldDifference] = []
        for column in sorted(
            set(first_entry.values) & set(second_entry.values),
            key=lambda item: (VALUE_FIELD_PRIORITY.get(_normalize_header(item), 100), _normalize_header(item)),
        ):
            difference = _compare_values(column, first_entry.values[column], second_entry.values[column])
            if difference:
                fields.append(difference)
        if fields:
            changed.append(
                RowDifference(
                    first_entry.key,
                    first_entry.sheet,
                    first_entry.row_number,
                    second_entry.sheet,
                    second_entry.row_number,
                    fields,
                )
            )
    return DocumentCompareReport(
        first_query=first_query,
        second_query=second_query,
        first=first.document,
        second=second.document,
        changed=changed,
        only_first=[
            MissingEntry(entry.key, entry.sheet, entry.row_number)
            for key, entry in sorted(first.entries.items())
            if key not in second_keys
        ],
        only_second=[
            MissingEntry(entry.key, entry.sheet, entry.row_number)
            for key, entry in sorted(second.entries.items())
            if key not in first_keys
        ],
        common_rows=len(common_keys),
        duplicate_keys_first=first.duplicates,
        duplicate_keys_second=second.duplicates,
    )


def _spreadsheet_preview(
    item: PortalSearchResult,
    path: Path,
    *,
    max_rows: int,
    max_sheets: int,
) -> list[dict[str, Any]]:
    sheets = _read_spreadsheet(path, original_name=item.title)
    if not sheets:
        raise ValueError("в табличном документе не нашёл непустых строк")
    preview: list[dict[str, Any]] = []
    for sheet in sheets[:max_sheets]:
        rows: list[dict[str, Any]] = []
        for row_number, row in sheet.rows[:max_rows]:
            rows.append(
                {
                    "row_number": row_number,
                    "cells": [
                        {"index": index, "letter": _column_letter(index), "value": _cell(row, index)}
                        for index in range(len(row))
                    ],
                }
            )
        preview.append({"name": sheet.name, "rows": rows, "total_non_empty_rows": len(sheet.rows)})
    return preview


def _extract_spreadsheet_dataset_with_schema(
    item: PortalSearchResult,
    path: Path,
    *,
    sheet_name: str,
    header_row_number: int,
    key_column: object,
    value_columns: list[object],
) -> SpreadsheetDataset:
    sheets = _read_spreadsheet(path, original_name=item.title)
    sheet = _select_sheet(sheets, sheet_name)
    if sheet is None:
        requested = f" '{sheet_name}'" if sheet_name else ""
        raise ValueError(f"не нашёл лист{requested} в документе {item.title}")

    header_position: int | None = None
    header_row: list[str] | None = None
    for position, (row_number, row) in enumerate(sheet.rows):
        if row_number == header_row_number:
            header_position = position
            header_row = row
            break
    if header_position is None or header_row is None:
        raise ValueError(f"не нашёл строку заголовков #{header_row_number} на листе {sheet.name}")

    key_index = _column_index_from_reference(key_column, header_row)
    value_indexes = _value_column_indexes_from_references(value_columns, header_row)
    if key_index in value_indexes:
        raise ValueError("key_column не может одновременно быть value_columns")

    used_labels: set[str] = set()
    value_labels: dict[int, str] = {}
    for column_index in value_indexes:
        label = _cell(header_row, column_index) or _column_letter(column_index)
        label = _dedupe_label(label, used_labels)
        used_labels.add(label)
        value_labels[column_index] = label

    entries: dict[str, SpreadsheetEntry] = {}
    duplicates = 0
    for row_number, row in sheet.rows[header_position + 1 :]:
        key = _cell(row, key_index)
        normalized_key = _normalize_key(key)
        if not normalized_key:
            continue
        values: dict[str, CellValue] = {}
        for column_index, field_label in value_labels.items():
            raw = _cell(row, column_index)
            if raw:
                values[field_label] = CellValue(field_label, raw, _parse_number(raw))
        if not values:
            continue
        if normalized_key in entries:
            duplicates += 1
            continue
        entries[normalized_key] = SpreadsheetEntry(key, normalized_key, values, sheet.name, row_number)
    if not entries:
        raise ValueError("после заголовков не нашёл позиций со значениями")
    return SpreadsheetDataset(
        document=ComparedDocument(
            entity_type=item.entity_type,
            entity_id=str(item.entity_id),
            title=item.title,
            url=item.url,
            rows=len(entries),
            value_fields=sorted({field for entry in entries.values() for field in entry.values}, key=_normalize_header),
        ),
        entries=entries,
        duplicates=duplicates,
    )


def _read_spreadsheet(path: Path, *, original_name: str) -> list[SheetRows]:
    extension = Path(original_name or path.name).suffix.lower()
    if extension == ".csv":
        return _read_csv(path)
    if extension == ".xlsx":
        return _read_xlsx(path)
    if extension == ".xls":
        return _read_xls(path)
    raise ValueError(f"формат {extension or '<none>'} пока не поддержан для сравнения")


def _read_csv(path: Path) -> list[SheetRows]:
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    sample = text[:4096]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t") if sample.strip() else csv.excel
    rows = [
        (index + 1, [str(value or "").strip() for value in row])
        for index, row in enumerate(csv.reader(text.splitlines(), dialect))
        if any(str(value or "").strip() for value in row)
    ]
    return [SheetRows(path.stem, rows)] if rows else []


def _read_xls(path: Path) -> list[SheetRows]:
    import xlrd

    workbook = xlrd.open_workbook(filename=str(path), on_demand=True)
    sheets: list[SheetRows] = []
    try:
        for sheet in workbook.sheets():
            rows = []
            for row_index in range(sheet.nrows):
                values = [
                    _format_xls_value(sheet.cell_value(row_index, column_index)) for column_index in range(sheet.ncols)
                ]
                if any(value.strip() for value in values):
                    rows.append((row_index + 1, values))
            if rows:
                sheets.append(SheetRows(sheet.name, rows))
    finally:
        workbook.release_resources()
    return sheets


def _read_xlsx(path: Path) -> list[SheetRows]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        sheets = []
        for sheet_name, worksheet_path in _xlsx_worksheet_paths(archive):
            root = ElementTree.fromstring(archive.read(worksheet_path))
            rows: list[tuple[int, list[str]]] = []
            for row in root.iter():
                if _local_name(row.tag) != "row":
                    continue
                row_number = int(row.attrib.get("r") or len(rows) + 1)
                values: list[str] = []
                for cell in row:
                    if _local_name(cell.tag) != "c":
                        continue
                    column_index = _xlsx_cell_column_index(cell)
                    while len(values) <= column_index:
                        values.append("")
                    values[column_index] = _xlsx_cell_value(cell, shared_strings)
                if any(value.strip() for value in values):
                    rows.append((row_number, values))
            if rows:
                sheets.append(SheetRows(sheet_name, rows))
        return sheets


def _select_sheet(sheets: list[SheetRows], sheet_name: str) -> SheetRows | None:
    if not sheets:
        return None
    requested = sheet_name.strip()
    if not requested:
        return sheets[0]
    for sheet in sheets:
        if sheet.name.casefold() == requested.casefold():
            return sheet
    normalized = _normalize_text(requested)
    for sheet in sheets:
        if _normalize_text(sheet.name) == normalized:
            return sheet
    return None


def _value_column_indexes_from_references(values: list[object], header_row: list[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        index = _column_index_from_reference(value, header_row)
        if index not in result:
            result.append(index)
    return result


def _column_index_from_reference(value: object, header_row: list[str]) -> int:
    if isinstance(value, bool):
        raise ValueError("column reference must be index, Excel letter, or header text")
    if isinstance(value, int):
        return _validate_column_index(value, header_row)

    text = str(value or "").strip()
    if not text:
        raise ValueError("empty column reference")

    letter_index = _column_index_from_letter(text)
    if letter_index is not None:
        return _validate_column_index(letter_index, header_row)

    if text.isdigit():
        numeric_index = int(text)
        if 0 <= numeric_index < len(header_row):
            return numeric_index
        if 1 <= numeric_index <= len(header_row):
            return numeric_index - 1

    normalized = _normalize_header(text)
    matches = [index for index, header in enumerate(header_row) if _normalize_header(header) == normalized]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"неоднозначная колонка {text!r}: совпало несколько заголовков")
    raise ValueError(f"не нашёл колонку {text!r}; используй index, letter или header из spreadsheet_preview")


def _validate_column_index(index: int, header_row: list[str]) -> int:
    if 0 <= index < len(header_row):
        return index
    raise ValueError(f"индекс колонки {index} вне диапазона preview 0..{max(len(header_row) - 1, 0)}")


def _column_index_from_letter(value: str) -> int | None:
    text = value.strip().upper()
    if not re.fullmatch(r"[A-Z]+", text):
        return None
    result = 0
    for char in text:
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def _column_letter(index: int) -> str:
    number = index + 1
    letters = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _compare_values(field: str, first: CellValue, second: CellValue) -> FieldDifference | None:
    if first.number is not None and second.number is not None:
        if math.isclose(first.number, second.number, abs_tol=0.01):
            return None
        delta = second.number - first.number
        percent = (delta / first.number * 100) if not math.isclose(first.number, 0.0) else None
        return FieldDifference(field, first, second, delta, percent)
    if _normalize_text(first.raw) == _normalize_text(second.raw):
        return None
    return FieldDifference(field, first, second)


def _direct_index_reference(index: PortalSearchIndex, query: str) -> PortalSearchResult | None:
    match = re.search(r"\b(disk_file|task_attachment)\s*[:#]?\s*(\d+)\b", query, flags=re.I)
    return index.get_item(entity_type=match.group(1).lower(), entity_id=match.group(2)) if match else None


def _is_file_item(item: PortalSearchResult) -> bool:
    return item.entity_type in {"disk_file", "task_attachment"}


def _document_search_types() -> set[str]:
    return entity_types_for_scope("documents") or {"disk_file", "task_attachment"}


def _is_spreadsheet(item: PortalSearchResult) -> bool:
    return Path(item.title).suffix.lower() in SPREADSHEET_EXTENSIONS


def _path_has_private_marker(path: str) -> bool:
    components = [part.strip().casefold() for part in path.replace("\\", "/").split("/") if part.strip()]
    markers = get_settings().resolved_agent_private_disk_path_markers
    return any(marker.strip().casefold() in components for marker in markers if marker.strip())


def _matches_private_marker(value: str) -> bool:
    normalized = value.strip().casefold()
    return bool(normalized) and any(
        normalized == marker.strip().casefold()
        for marker in get_settings().resolved_agent_private_disk_path_markers
        if marker.strip()
    )


def _document_dict(item: PortalSearchResult) -> dict[str, Any]:
    return {
        "entity_type": item.entity_type,
        "entity_id": item.entity_id,
        "title": item.title,
        "url": item.url,
        "metadata": item.metadata,
    }


def _candidate_dict(item: PortalSearchResult) -> dict[str, Any]:
    return {**_document_dict(item), "score": item.score}


def _document_link(item: PortalSearchResult) -> str:
    return _format_document_link(item.title, item.url)


def _format_document_link(title: str, url: str) -> str:
    return f"[{title}]({url})" if url else title


def _safe_draft_name(title: str, extension: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9а-яА-Я._-]+", "_", title).strip("._")
    if not name:
        name = "draft"
    if not name.lower().endswith(extension):
        name += extension
    return name[:120]


def _format_field_difference(difference: FieldDifference) -> str:
    if difference.first.number is not None and difference.second.number is not None:
        delta = _format_signed_number(difference.delta or 0)
        percent = f"; {_format_signed_number(difference.percent)}%" if difference.percent is not None else ""
        return f"{_format_number(difference.first.number)} -> {_format_number(difference.second.number)} ({delta}{percent})"
    return f"{difference.first.raw} -> {difference.second.raw}"


def _dedupe_label(label: str, used: set[str]) -> str:
    if label not in used:
        return label
    index = 2
    while f"{label} {index}" in used:
        index += 1
    return f"{label} {index}"


def _parse_number(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    cleaned = re.sub(r"[^0-9,.\-]", "", text.replace("\u00a0", " "))
    if not re.search(r"\d", cleaned):
        return None
    if "," in cleaned and "." in cleaned:
        decimal_separator = "," if cleaned.rfind(",") > cleaned.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
        cleaned = cleaned.replace(thousands_separator, "").replace(decimal_separator, ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_key(value: str) -> str:
    text = _normalize_text(value)
    text = re.sub(r"^\d+[\).,\-:\s]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_header(value: object) -> str:
    return _normalize_text(value).replace("-", " ")


def _normalize_text(value: object) -> str:
    text = str(value or "").lower().replace("ё", "е")
    text = re.sub(r"[^\w№%.,/+\- ]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _cell(row: list[str], index: int) -> str:
    return str(row[index] or "").strip() if 0 <= index < len(row) else ""


def _format_number(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=0.001):
        return f"{int(round(value)):,}".replace(",", " ")
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def _format_signed_number(value: float | None) -> str:
    number = value or 0.0
    return ("+" if number > 0 else "") + _format_number(number)


def _format_xls_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _xlsx_worksheet_paths(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    except Exception:
        return [
            (Path(name).stem, name)
            for name in sorted(archive.namelist())
            if name.startswith("xl/worksheets/") and name.endswith(".xml")
        ]
    rel_targets = {
        rel.attrib.get("Id"): rel.attrib.get("Target")
        for rel in rels
        if rel.attrib.get("Id") and rel.attrib.get("Target")
    }
    result: list[tuple[str, str]] = []
    for sheet in workbook.iter():
        if _local_name(sheet.tag) != "sheet":
            continue
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        target = rel_targets.get(rel_id)
        if target:
            result.append((str(sheet.attrib.get("name") or Path(target).stem), f"xl/{target}".replace("xl/../", "")))
    return result


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter() if _local_name(node.tag) == "t").strip()
        for item in root
        if _local_name(item.tag) == "si"
    ]


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if _local_name(node.tag) == "t").strip()
    raw_value = ""
    for child in cell:
        if _local_name(child.tag) == "v" and child.text:
            raw_value = child.text.strip()
            break
    if cell_type == "s" and raw_value.isdigit():
        index = int(raw_value)
        if 0 <= index < len(shared_strings):
            return shared_strings[index]
    return raw_value


def _xlsx_cell_column_index(cell: ElementTree.Element) -> int:
    match = re.match(r"([A-Z]+)", str(cell.attrib.get("r") or ""), flags=re.I)
    if not match:
        return 0
    result = 0
    for char in match.group(1).upper():
        result = result * 26 + (ord(char) - ord("A") + 1)
    return max(result - 1, 0)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def _dict_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None
