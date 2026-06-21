from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_server.integrations.bitrix.portal_search import PortalSearchResult


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
            "first_candidates": [
                {
                    "entity_type": i.entity_type,
                    "entity_id": i.entity_id,
                    "title": i.title,
                    "url": i.url,
                    "metadata": i.metadata,
                    "score": i.score,
                }
                for i in self.first_candidates
            ],
            "second_candidates": [
                {
                    "entity_type": i.entity_type,
                    "entity_id": i.entity_id,
                    "title": i.title,
                    "url": i.url,
                    "metadata": i.metadata,
                    "score": i.score,
                }
                for i in self.second_candidates
            ],
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
