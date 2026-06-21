from __future__ import annotations

import asyncio
import csv
import math
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from ai_server.integrations.bitrix.portal_search import (
    PortalSearchIndex,
    PortalSearchResult,
    delete_portal_file_cache_path,
    entity_types_for_scope,
)
from ai_server.integrations.bitrix.ports import BitrixFileDownloadPort
from ai_server.models import ToolResult, ToolStatus
from ai_server.settings import Settings
from ai_server.tools.document_access.download import _ensure_local_document
from ai_server.tools.document_access.types import (
    CellValue,
    ComparedDocument,
    DocumentCompareReport,
    FieldDifference,
    MissingEntry,
    RowDifference,
    SheetRows,
    SpreadsheetCompareSchema,
    SpreadsheetDataset,
    SpreadsheetEntry,
)
from ai_server.utils import optional_int

SPREADSHEET_EXTENSIONS = {".csv", ".xls", ".xlsx"}
VALUE_FIELD_PRIORITY = {"цена": 10, "стоимость": 20, "количество": 30}


async def compare_spreadsheets_by_query(
    bitrix: BitrixFileDownloadPort,
    index: PortalSearchIndex,
    *,
    first_query: str,
    second_query: str,
    schema: SpreadsheetCompareSchema,
    limit: int = 20,
    item_filter: Callable[[PortalSearchResult], bool] | None = None,
    settings: Settings,
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
        first_path = await _ensure_local_document(
            bitrix, first_item, max_bytes=settings.search_content_max_bytes, settings=settings
        )
        second_path = await _ensure_local_document(
            bitrix, second_item, max_bytes=settings.search_content_max_bytes, settings=settings
        )
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
        if not settings.search_content_keep_local_files:
            for path in (first_path, second_path):
                if path is not None:
                    delete_portal_file_cache_path(path, settings)

    compared = _compare_datasets(
        first_dataset, second_dataset, first_query=report.first_query, second_query=report.second_query
    )
    compared.changed = compared.changed[:limit]
    compared.only_first = compared.only_first[:limit]
    compared.only_second = compared.only_second[:limit]
    return compared


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
                status=ToolStatus.CONTRACT_VIOLATION,
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
            status=ToolStatus.INVALID_TOOL_CALL,
            tool="spreadsheet_compare",
            error="header row numbers must be positive",
        )

    key_column = args.get("key_column")
    if not _has_column_reference(key_column):
        return _empty_schema(), ToolResult(
            status=ToolStatus.CONTRACT_VIOLATION,
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
            status=ToolStatus.CONTRACT_VIOLATION,
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


def _parse_number(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    cleaned = re.sub(r"[^0-9,.\-]", "", text.replace(" ", " "))
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


def _dedupe_label(label: str, used: set[str]) -> str:
    if label not in used:
        return label
    index = 2
    while f"{label} {index}" in used:
        index += 1
    return f"{label} {index}"


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
