from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTENT_INDEX_VERSION = "2026-05-07-doc-v3"
CONTENT_TERMINAL_STATUSES = {"empty", "too_large", "failed", "no_download_url"}


@dataclass(frozen=True)
class PortalIndexStats:
    total_items: int
    by_type: dict[str, int]
    content_by_status: dict[str, int]
    last_indexed_at: str | None
    path: Path
    exists: bool


@dataclass
class PortalSyncStats:
    tasks: int = 0
    projects: int = 0
    disk_items: int = 0
    storages: int = 0
    task_attachments: int = 0
    catalog_products: int = 0
    catalog_stores: int = 0
    catalog_stock_rows: int = 0
    stale_deleted: int = 0
    prune_skipped: list[str] | None = None
    content: PortalContentSyncStats | None = None
    errors: list[str] | None = None

    @property
    def total(self) -> int:
        return (
            self.tasks
            + self.projects
            + self.disk_items
            + self.task_attachments
            + self.catalog_products
            + self.catalog_stores
            + self.catalog_stock_rows
        )


@dataclass
class PortalDeltaSyncStats:
    folders_scanned: int = 0
    items_seen: int = 0
    items_changed: int = 0
    files_changed: int = 0
    folders_changed: int = 0
    deleted: int = 0
    cursor_type: str | None = None
    cursor_id: str | None = None
    wrapped: bool = False
    errors: list[str] | None = None


@dataclass
class PortalContentSyncStats:
    candidates: int = 0
    downloaded: int = 0
    indexed: int = 0
    skipped: int = 0
    unsupported: int = 0
    failed: int = 0
    errors: list[str] | None = None


@dataclass(frozen=True)
class PortalContentReadiness:
    total_documents: int
    supported_documents: int
    indexed: int
    pending: int
    terminal: int
    unsupported: int
    indexed_by_extension: dict[str, int]
    pending_by_extension: dict[str, int]
    pending_by_status: dict[str, int]
    terminal_by_status: dict[str, int]
    unsupported_by_extension: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "supported_documents": self.supported_documents,
            "indexed": self.indexed,
            "pending": self.pending,
            "terminal": self.terminal,
            "unsupported": self.unsupported,
            "indexed_by_extension": self.indexed_by_extension,
            "pending_by_extension": self.pending_by_extension,
            "pending_by_status": self.pending_by_status,
            "terminal_by_status": self.terminal_by_status,
            "unsupported_by_extension": self.unsupported_by_extension,
        }


@dataclass(frozen=True)
class PortalSearchResult:
    entity_type: str
    entity_id: str
    title: str
    body: str
    url: str
    score: int
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "title": self.title,
            "body": self.body,
            "url": self.url,
            "score": self.score,
            "metadata": self.metadata,
        }
