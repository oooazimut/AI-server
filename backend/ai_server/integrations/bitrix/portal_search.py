from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from ai_server.document_text import extract_text_from_file
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.runtime import runtime_paths
from ai_server.settings import get_settings
from ai_server.utils import MOSCOW_TZ

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
    stale_deleted: int = 0
    prune_skipped: list[str] | None = None
    content: PortalContentSyncStats | None = None
    errors: list[str] | None = None

    @property
    def total(self) -> int:
        return self.tasks + self.projects + self.disk_items + self.task_attachments + self.catalog_products


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


class PortalSearchIndex:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else runtime_paths().search_index_db

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS portal_search_items (
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    url TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source_updated_at TEXT,
                    last_seen_at TEXT,
                    indexed_at TEXT NOT NULL,
                    PRIMARY KEY (entity_type, entity_id)
                )
                """
            )
            self._ensure_column(connection, "portal_search_items", "last_seen_at", "TEXT")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_portal_search_type ON portal_search_items(entity_type)")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_portal_search_indexed ON portal_search_items(indexed_at)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_portal_search_seen ON portal_search_items(last_seen_at)")

    def upsert_item(
        self,
        *,
        entity_type: str,
        entity_id: object,
        title: str,
        body: str = "",
        url: str = "",
        metadata: dict[str, Any] | None = None,
        source_updated_at: str | None = None,
        preserve_content: bool = True,
    ) -> None:
        self.ensure_schema()
        now = datetime.now(MOSCOW_TZ).isoformat()
        normalized_title = _clean_text(title) or f"{entity_type} #{entity_id}"
        normalized_body = _clean_text(body)
        normalized_metadata = dict(metadata or {})
        existing = self._get_existing_item(entity_type=entity_type, entity_id=entity_id)
        if (
            preserve_content
            and existing
            and _should_preserve_content(
                existing_metadata=existing["metadata"],
                new_metadata=normalized_metadata,
                existing_source_updated_at=existing["source_updated_at"],
                new_source_updated_at=source_updated_at,
            )
        ):
            normalized_metadata = _merge_content_metadata(
                base=normalized_metadata,
                existing=existing["metadata"],
            )
            existing_content = _content_text_from_body(existing["body"])
            if existing_content:
                normalized_body = _body_with_content(normalized_body, existing_content)

        search_text = _normalize_search_text(" ".join([entity_type, str(entity_id), normalized_title, normalized_body]))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO portal_search_items (
                    entity_type,
                    entity_id,
                    title,
                    body,
                    url,
                    search_text,
                    metadata_json,
                    source_updated_at,
                    last_seen_at,
                    indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, entity_id) DO UPDATE SET
                    title = excluded.title,
                    body = excluded.body,
                    url = excluded.url,
                    search_text = excluded.search_text,
                    metadata_json = excluded.metadata_json,
                    source_updated_at = excluded.source_updated_at,
                    last_seen_at = excluded.last_seen_at,
                    indexed_at = excluded.indexed_at
                """,
                (
                    entity_type,
                    str(entity_id),
                    normalized_title,
                    normalized_body,
                    url,
                    search_text,
                    json.dumps(normalized_metadata, ensure_ascii=False),
                    source_updated_at,
                    now,
                    now,
                ),
            )

    def delete_item(self, *, entity_type: str, entity_id: object) -> bool:
        self.ensure_schema()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM portal_search_items
                WHERE entity_type = ? AND entity_id = ?
                """,
                (entity_type, str(entity_id)),
            )
            return bool(cursor.rowcount)

    def delete_stale_items(self, *, entity_types: set[str], seen_before: str) -> int:
        self.ensure_schema()
        if not entity_types:
            return 0
        placeholders = ",".join("?" for _ in entity_types)
        params: list[object] = [*sorted(entity_types), seen_before]
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM portal_search_items
                WHERE entity_type IN ({placeholders})
                  AND (last_seen_at IS NULL OR last_seen_at < ?)
                """,
                params,
            )
            return int(cursor.rowcount or 0)

    def search(
        self,
        query: str,
        *,
        entity_types: set[str] | None = None,
        limit: int = 10,
    ) -> list[PortalSearchResult]:
        self.ensure_schema()
        term_groups = _query_term_groups(query)
        terms = _flatten_unique(term_groups)
        if not term_groups:
            return []

        where = []
        params: list[object] = []
        for group in term_groups:
            group_where = []
            for term in group:
                group_where.append("search_text LIKE ? ESCAPE '\\'")
                params.append(f"%{_escape_like(term)}%")
            where.append("(" + " OR ".join(group_where) + ")")
        if entity_types:
            placeholders = ",".join("?" for _ in entity_types)
            where.append(f"entity_type IN ({placeholders})")
            params.extend(sorted(entity_types))

        sql = (
            "SELECT entity_type, entity_id, title, body, url, metadata_json, search_text "
            "FROM portal_search_items WHERE " + " AND ".join(where) + " LIMIT 500"
        )
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        normalized_query = _normalize_search_text(query)
        scored: list[PortalSearchResult] = []
        for row in rows:
            metadata = _safe_json(row["metadata_json"])
            score = _score_result(
                normalized_query,
                terms,
                title=_normalize_search_text(row["title"]),
                body=_normalize_search_text(row["body"]),
                search_text=row["search_text"],
            )
            scored.append(
                PortalSearchResult(
                    entity_type=row["entity_type"],
                    entity_id=row["entity_id"],
                    title=row["title"],
                    body=row["body"],
                    url=row["url"],
                    score=score,
                    metadata=metadata,
                )
            )
        return sorted(scored, key=lambda item: (-item.score, item.entity_type, item.title))[:limit]

    def disk_delta_folder_candidates(
        self,
        *,
        cursor_type: str | None,
        cursor_id: str | None,
        limit: int,
    ) -> tuple[list[PortalSearchResult], str | None, str | None, bool]:
        self.ensure_schema()
        normalized_type = cursor_type or ""
        normalized_id = _safe_int(cursor_id) or 0
        rows = self._select_disk_delta_folder_rows(
            cursor_type=normalized_type,
            cursor_id=normalized_id,
            limit=limit,
        )
        wrapped = False
        if not rows and (normalized_type or normalized_id):
            rows = self._select_disk_delta_folder_rows(cursor_type="", cursor_id=0, limit=limit)
            wrapped = bool(rows)

        candidates = [_row_to_search_result(row) for row in rows]
        next_type: str | None = None
        next_id: str | None = None
        if candidates:
            last = candidates[-1]
            next_type = last.entity_type
            next_id = last.entity_id
        return candidates, next_type, next_id, wrapped

    def children_by_parent_id(self, parent_id: object) -> list[PortalSearchResult]:
        self.ensure_schema()
        parent = str(parent_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM portal_search_items
                WHERE entity_type IN ('disk_file', 'disk_folder')
                  AND (
                    metadata_json LIKE ? ESCAPE '\\'
                    OR metadata_json LIKE ? ESCAPE '\\'
                  )
                """,
                (
                    f'%"parent_id": "{_escape_like(parent)}"%',
                    f'%"parent_id": {parent}%',
                ),
            ).fetchall()
        return [_row_to_search_result(row) for row in rows]

    def content_candidates(self, *, limit: int) -> list[PortalSearchResult]:
        self.ensure_schema()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM portal_search_items
                WHERE entity_type IN ('disk_file', 'task_attachment')
                ORDER BY indexed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_row_to_search_result(row) for row in rows]

    def content_readiness(self, *, allowed_extensions: set[str]) -> PortalContentReadiness:
        self.ensure_schema()
        normalized_allowed_extensions = _normalize_extensions(allowed_extensions)
        indexed_by_extension: dict[str, int] = {}
        pending_by_extension: dict[str, int] = {}
        pending_by_status: dict[str, int] = {}
        terminal_by_status: dict[str, int] = {}
        unsupported_by_extension: dict[str, int] = {}
        total_documents = 0
        supported_documents = 0
        indexed = 0
        pending = 0
        terminal = 0
        unsupported = 0

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT title, metadata_json
                FROM portal_search_items
                WHERE entity_type IN ('disk_file', 'task_attachment')
                """
            ).fetchall()

        for row in rows:
            total_documents += 1
            extension = _file_extension(str(row["title"] or "")) or "<none>"
            metadata = _safe_json(row["metadata_json"])
            status = str(metadata.get("content_index_status") or "none")
            content_version = str(metadata.get("content_index_version") or "")

            if extension not in normalized_allowed_extensions:
                unsupported += 1
                _increment(unsupported_by_extension, extension)
                continue

            supported_documents += 1
            if status == "indexed":
                indexed += 1
                _increment(indexed_by_extension, extension)
            elif content_version == CONTENT_INDEX_VERSION and status in CONTENT_TERMINAL_STATUSES:
                terminal += 1
                _increment(terminal_by_status, status)
            else:
                pending += 1
                _increment(pending_by_extension, extension)
                _increment(pending_by_status, status)

        return PortalContentReadiness(
            total_documents=total_documents,
            supported_documents=supported_documents,
            indexed=indexed,
            pending=pending,
            terminal=terminal,
            unsupported=unsupported,
            indexed_by_extension=indexed_by_extension,
            pending_by_extension=pending_by_extension,
            pending_by_status=pending_by_status,
            terminal_by_status=terminal_by_status,
            unsupported_by_extension=unsupported_by_extension,
        )

    def update_item_body_metadata(
        self,
        *,
        entity_type: str,
        entity_id: object,
        body: str,
        metadata: dict[str, Any],
    ) -> None:
        self.ensure_schema()
        now = datetime.now(MOSCOW_TZ).isoformat()
        normalized_body = _clean_text(body)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT title
                FROM portal_search_items
                WHERE entity_type = ? AND entity_id = ?
                """,
                (entity_type, str(entity_id)),
            ).fetchone()
            if not row:
                return
            search_text = _normalize_search_text(" ".join([entity_type, str(entity_id), row["title"], normalized_body]))
            connection.execute(
                """
                UPDATE portal_search_items
                SET body = ?,
                    search_text = ?,
                    metadata_json = ?,
                    indexed_at = ?
                WHERE entity_type = ? AND entity_id = ?
                """,
                (
                    normalized_body,
                    search_text,
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    entity_type,
                    str(entity_id),
                ),
            )

    def stats(self) -> PortalIndexStats:
        if not self.path.exists():
            return PortalIndexStats(
                total_items=0,
                by_type={},
                content_by_status={},
                last_indexed_at=None,
                path=self.path,
                exists=False,
            )
        self.ensure_schema()
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM portal_search_items").fetchone()[0]
            by_type_rows = connection.execute(
                "SELECT entity_type, COUNT(*) AS count FROM portal_search_items GROUP BY entity_type"
            ).fetchall()
            last_indexed_at = connection.execute("SELECT MAX(indexed_at) FROM portal_search_items").fetchone()[0]
            content_rows = connection.execute(
                """
                SELECT metadata_json
                FROM portal_search_items
                WHERE entity_type IN ('disk_file', 'task_attachment')
                """
            ).fetchall()

        content_by_status: dict[str, int] = {}
        for row in content_rows:
            status = _safe_json(row["metadata_json"]).get("content_index_status")
            if not status:
                continue
            status_key = str(status)
            content_by_status[status_key] = content_by_status.get(status_key, 0) + 1
        return PortalIndexStats(
            total_items=int(total),
            by_type={str(row["entity_type"]): int(row["count"]) for row in by_type_rows},
            content_by_status=content_by_status,
            last_indexed_at=str(last_indexed_at) if last_indexed_at else None,
            path=self.path,
            exists=True,
        )

    def get_item(self, *, entity_type: str, entity_id: object) -> PortalSearchResult | None:
        self.ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM portal_search_items
                WHERE entity_type = ? AND entity_id = ?
                """,
                (entity_type, str(entity_id)),
            ).fetchone()
        if not row:
            return None
        return _row_to_search_result(row)

    def item_snapshot(self, *, entity_type: str, entity_id: object) -> dict[str, Any] | None:
        self.ensure_schema()
        return self._get_existing_item(entity_type=entity_type, entity_id=entity_id)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        columns = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def _select_disk_delta_folder_rows(
        self,
        *,
        cursor_type: str,
        cursor_id: int,
        limit: int,
    ) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM portal_search_items
                WHERE entity_type IN ('disk_storage', 'disk_folder')
                  AND (
                    ? = ''
                    OR entity_type > ?
                    OR (entity_type = ? AND CAST(entity_id AS INTEGER) > ?)
                  )
                ORDER BY entity_type, CAST(entity_id AS INTEGER)
                LIMIT ?
                """,
                (cursor_type, cursor_type, cursor_type, cursor_id, limit),
            ).fetchall()

    def _get_existing_item(
        self,
        *,
        entity_type: str,
        entity_id: object,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT body, metadata_json, source_updated_at
                FROM portal_search_items
                WHERE entity_type = ? AND entity_id = ?
                """,
                (entity_type, str(entity_id)),
            ).fetchone()
        if not row:
            return None
        return {
            "body": str(row["body"] or ""),
            "metadata": _safe_json(row["metadata_json"]),
            "source_updated_at": row["source_updated_at"],
        }


async def sync_portal_index(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    include_content: bool = False,
) -> PortalSyncStats:
    sync_started_at = datetime.now(MOSCOW_TZ).isoformat()
    settings = get_settings()
    stats = PortalSyncStats(errors=[], prune_skipped=[])

    stats.projects = await _sync_projects(bitrix, index)
    if stats.projects < settings.search_index_max_projects:
        stats.stale_deleted += index.delete_stale_items(
            entity_types={"project"},
            seen_before=sync_started_at,
        )
    else:
        stats.prune_skipped.append("projects: reached configured limit")

    task_sync = await _sync_tasks(bitrix, index)
    stats.tasks = int(task_sync["tasks"])
    stats.task_attachments = int(task_sync["attachments"])
    task_prune_types = set()
    if bool(task_sync["tasks_complete"]):
        task_prune_types.add("task")
    else:
        stats.prune_skipped.append("tasks: reached configured limit")
    if bool(task_sync["attachments_complete"]):
        task_prune_types.add("task_attachment")
    elif settings.search_index_include_task_attachments:
        stats.prune_skipped.append("task attachments: reached configured limit")
    if task_prune_types:
        stats.stale_deleted += index.delete_stale_items(
            entity_types=task_prune_types,
            seen_before=sync_started_at,
        )

    if settings.search_index_include_catalog:
        try:
            catalog_stats = await _sync_catalog(bitrix, index)
            stats.catalog_products = catalog_stats["products"]
            stats.catalog_stores = catalog_stats["stores"]
            if stats.catalog_products < settings.search_index_max_catalog_products:
                stats.stale_deleted += index.delete_stale_items(
                    entity_types={"catalog_product", "catalog_store"},
                    seen_before=sync_started_at,
                )
            else:
                stats.prune_skipped = (stats.prune_skipped or []) + ["catalog: reached configured limit"]
        except Exception as exc:
            stats.errors = (stats.errors or []) + [f"catalog: {type(exc).__name__}: {exc}"]

    if settings.search_index_include_disk:
        try:
            disk_stats = await _sync_disk(bitrix, index)
            stats.storages = int(disk_stats["storages"])
            stats.disk_items = int(disk_stats["items"])
            if bool(disk_stats["complete"]):
                stats.stale_deleted += index.delete_stale_items(
                    entity_types={"disk_storage", "disk_folder", "disk_file"},
                    seen_before=sync_started_at,
                )
            else:
                stats.prune_skipped.append("disk: reached configured limit")
        except Exception as exc:
            stats.errors.append(f"disk: {type(exc).__name__}: {exc}")

    if include_content and settings.search_content_enabled:
        try:
            stats.content = await sync_portal_content_index(bitrix, index)
        except Exception as exc:
            stats.errors.append(f"content: {type(exc).__name__}: {exc}")
    if not stats.prune_skipped:
        stats.prune_skipped = None
    if not stats.errors:
        stats.errors = None
    return stats


async def sync_portal_content_index(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    extensions: set[str] | None = None,
) -> PortalContentSyncStats:
    settings = get_settings()
    stats = PortalContentSyncStats(errors=[])
    allowed_extensions = _normalize_extensions(settings.resolved_search_content_allowed_extensions)
    if extensions:
        allowed_extensions &= _normalize_extensions(extensions)
    candidate_limit = max(settings.search_content_max_files * 50, settings.search_content_max_files)
    processed_downloads = 0

    for item in index.content_candidates(limit=candidate_limit):
        metadata = dict(item.metadata)
        status = str(metadata.get("content_index_status") or "")
        content_version = str(metadata.get("content_index_version") or "")
        extension = _file_extension(item.title)
        if extensions and extension not in allowed_extensions:
            continue
        if status == "indexed" and (content_version == CONTENT_INDEX_VERSION or not extensions):
            continue
        if content_version == CONTENT_INDEX_VERSION and status in {
            "unsupported",
            "too_large",
            "empty",
            "failed",
            "no_download_url",
        }:
            continue

        stats.candidates += 1
        if extension not in allowed_extensions:
            _mark_content_status(
                index,
                item,
                metadata,
                status="unsupported",
                reason=f"extension {extension or '<none>'} is not enabled",
            )
            stats.unsupported += 1
            continue

        size = _safe_int(metadata.get("size"))
        if size and size > settings.search_content_max_bytes:
            _mark_content_status(
                index,
                item,
                metadata,
                status="too_large",
                reason=f"file exceeds {settings.search_content_max_bytes} bytes",
            )
            stats.skipped += 1
            continue

        if processed_downloads >= settings.search_content_max_files:
            break
        processed_downloads += 1

        item_stats = await sync_portal_content_item(
            bitrix,
            index,
            item,
            extensions={extension},
        )
        stats.downloaded += item_stats.downloaded
        stats.indexed += item_stats.indexed
        stats.skipped += item_stats.skipped
        stats.unsupported += item_stats.unsupported
        stats.failed += item_stats.failed
        if item_stats.errors:
            stats.errors.extend(item_stats.errors)

    if not stats.errors:
        stats.errors = None
    return stats


async def sync_portal_content_item(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    item: PortalSearchResult,
    *,
    extensions: set[str] | None = None,
) -> PortalContentSyncStats:
    settings = get_settings()
    stats = PortalContentSyncStats(errors=[])
    allowed_extensions = _normalize_extensions(settings.resolved_search_content_allowed_extensions)
    if extensions:
        allowed_extensions &= _normalize_extensions(extensions)

    metadata = dict(item.metadata)
    extension = _file_extension(item.title)
    if extensions and extension not in allowed_extensions:
        return stats

    stats.candidates = 1
    if extension not in allowed_extensions:
        _mark_content_status(
            index,
            item,
            metadata,
            status="unsupported",
            reason=f"extension {extension or '<none>'} is not enabled",
        )
        stats.unsupported += 1
        return stats

    size = _safe_int(metadata.get("size"))
    if size and size > settings.search_content_max_bytes:
        _mark_content_status(
            index,
            item,
            metadata,
            status="too_large",
            reason=f"file exceeds {settings.search_content_max_bytes} bytes",
        )
        stats.skipped += 1
        return stats

    target_path: Path | None = None
    downloaded_for_indexing = False
    try:
        download_url = await _resolve_download_url(bitrix, item)
        if not download_url:
            _mark_content_status(
                index,
                item,
                metadata,
                status="no_download_url",
                reason="Bitrix did not return a download URL",
            )
            stats.failed += 1
            return stats

        target_path = portal_file_cache_path(item)
        downloaded_bytes = await bitrix.download_file_from_url(
            download_url,
            target_path,
            max_bytes=settings.search_content_max_bytes,
        )
        downloaded_for_indexing = True
        stats.downloaded += 1

        extracted = await asyncio.to_thread(
            extract_text_from_file,
            target_path,
            original_name=item.title,
            max_chars=settings.search_content_max_chars,
        )
        metadata.update(
            {
                "content_index_status": extracted.status,
                "content_index_version": CONTENT_INDEX_VERSION,
                "content_index_reason": extracted.reason,
                "content_indexed_at": datetime.now(MOSCOW_TZ).isoformat(),
                "content_bytes": downloaded_bytes,
                "content_extension": extension,
                "content_text_length": len(extracted.text),
            }
        )
        if extracted.status == "indexed":
            index.update_item_body_metadata(
                entity_type=item.entity_type,
                entity_id=item.entity_id,
                body=_body_with_content(item.body, extracted.text),
                metadata=metadata,
            )
            stats.indexed += 1
        else:
            index.update_item_body_metadata(
                entity_type=item.entity_type,
                entity_id=item.entity_id,
                body=item.body,
                metadata=metadata,
            )
            if extracted.status == "unsupported":
                stats.unsupported += 1
            elif extracted.status == "failed":
                stats.failed += 1
            else:
                stats.skipped += 1
    except Exception as exc:
        metadata.update(
            {
                "content_index_status": "failed",
                "content_index_version": CONTENT_INDEX_VERSION,
                "content_index_reason": type(exc).__name__,
                "content_indexed_at": datetime.now(MOSCOW_TZ).isoformat(),
                "content_extension": extension,
            }
        )
        index.update_item_body_metadata(
            entity_type=item.entity_type,
            entity_id=item.entity_id,
            body=item.body,
            metadata=metadata,
        )
        stats.failed += 1
        stats.errors = [f"{item.entity_type} #{item.entity_id} {item.title}: {type(exc).__name__}"]
    finally:
        if downloaded_for_indexing and target_path is not None and not settings.search_content_keep_local_files:
            delete_portal_file_cache_path(target_path)

    return stats


async def sync_disk_delta_index(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    cursor_type: str | None,
    cursor_id: str | None,
    folder_limit: int,
    child_limit: int,
) -> PortalDeltaSyncStats:
    stats = PortalDeltaSyncStats(errors=[])
    folders, next_type, next_id, wrapped = index.disk_delta_folder_candidates(
        cursor_type=cursor_type,
        cursor_id=cursor_id,
        limit=folder_limit,
    )
    stats.cursor_type = next_type
    stats.cursor_id = next_id
    stats.wrapped = wrapped

    for folder in folders:
        folder_id = _delta_folder_id(folder)
        if folder_id is None:
            continue
        try:
            folder_stats = await _sync_disk_folder_delta(
                bitrix,
                index,
                folder_id=folder_id,
                storage_name=_delta_storage_name(folder),
                path=_delta_folder_path(folder),
                child_limit=child_limit,
            )
            stats.folders_scanned += 1
            stats.items_seen += int(folder_stats["items_seen"])
            stats.items_changed += int(folder_stats["items_changed"])
            stats.files_changed += int(folder_stats["files_changed"])
            stats.folders_changed += int(folder_stats["folders_changed"])
            stats.deleted += int(folder_stats["deleted"])
        except Exception as exc:
            stats.errors.append(f"{folder.entity_type} #{folder.entity_id}: {type(exc).__name__}: {exc}")
    if not stats.errors:
        stats.errors = None
    return stats


async def sync_disk_file_item(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    file_id: int,
    preserve_content: bool = True,
) -> PortalSearchResult | None:
    file_data = await bitrix.get_disk_file(file_id)
    if not isinstance(file_data, dict):
        return None

    item_id = _first(file_data, "ID", "id") or file_id
    name = str(_first(file_data, "NAME", "name") or f"Файл #{item_id}")
    item_type = str(_first(file_data, "TYPE", "type") or "file").lower()
    detail_url = _normalize_url(_to_str(_first(file_data, "DETAIL_URL", "detailUrl")))
    storage_name = _to_str(_first(file_data, "STORAGE_NAME", "storageName"))
    path = _to_str(_first(file_data, "PATH", "path"))
    storage_id = _first(file_data, "STORAGE_ID", "storageId")
    parent_id = _first(file_data, "PARENT_ID", "parentId")
    update_time = _to_str(_first(file_data, "UPDATE_TIME", "updateTime", "UPDATED_TIME", "updatedTime"))

    body_parts = [
        f"Диск: {storage_name}" if storage_name else "",
        f"Путь: {path}" if path else "",
        f"Тип: {item_type}",
        f"Хранилище ID: {storage_id}" if storage_id else "",
        f"Папка ID: {parent_id}" if parent_id else "",
    ]
    index.upsert_item(
        entity_type="disk_file",
        entity_id=item_id,
        title=name,
        body="\n".join(part for part in body_parts if part),
        url=detail_url or _disk_object_url(item_id),
        metadata={
            "type": item_type,
            "path": path,
            "storage_name": storage_name,
            "storage_id": storage_id,
            "parent_id": parent_id,
            "disk_object_id": item_id,
            "detail_url": detail_url,
            "size": _first(file_data, "SIZE", "size"),
            "created_by": _first(file_data, "CREATED_BY", "createdBy"),
            "updated_by": _first(file_data, "UPDATED_BY", "updatedBy"),
            "webhook_synced_at": datetime.now(MOSCOW_TZ).isoformat(),
        },
        source_updated_at=update_time,
        preserve_content=preserve_content,
    )
    return index.get_item(entity_type="disk_file", entity_id=item_id)


async def _resolve_download_url(bitrix: BitrixClient, item: PortalSearchResult) -> str | None:
    if item.entity_type == "task_attachment":
        attached_object_id = _safe_int(item.metadata.get("attached_object_id")) or _safe_int(item.entity_id)
        if not attached_object_id:
            return None
        attached = await bitrix.get_attached_object(attached_object_id)
        if isinstance(attached, dict):
            return _to_str(_first(attached, "DOWNLOAD_URL", "downloadUrl"))
        return None

    if item.entity_type == "disk_file":
        disk_file_id = _safe_int(item.metadata.get("disk_object_id")) or _safe_int(item.entity_id)
        if not disk_file_id:
            return None
        return await bitrix.get_disk_file_download_url(disk_file_id)

    return None


def _mark_content_status(
    index: PortalSearchIndex,
    item: PortalSearchResult,
    metadata: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> None:
    metadata.update(
        {
            "content_index_status": status,
            "content_index_version": CONTENT_INDEX_VERSION,
            "content_index_reason": reason,
            "content_indexed_at": datetime.now(MOSCOW_TZ).isoformat(),
            "content_extension": _file_extension(item.title),
        }
    )
    index.update_item_body_metadata(
        entity_type=item.entity_type,
        entity_id=item.entity_id,
        body=item.body,
        metadata=metadata,
    )


def portal_file_cache_path(item: PortalSearchResult) -> Path:
    extension = _file_extension(item.title) or ".bin"
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{item.entity_type}_{item.entity_id}")
    return get_settings().search_content_storage_dir / item.entity_type / f"{safe_id}{extension}"


def delete_portal_file_cache_path(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return

    storage_root = get_settings().search_content_storage_dir.resolve()
    parent = path.parent
    while True:
        try:
            if parent.resolve() == storage_root:
                break
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def format_portal_sync_stats(stats: PortalSyncStats) -> str:
    lines = [
        "Индекс портала обновлён.",
        f"Задачи: {stats.tasks}",
        f"Проекты: {stats.projects}",
        f"Диск: {stats.disk_items} объектов в {stats.storages} хранилищах",
        f"Вложения задач: {stats.task_attachments}",
        f"Всего: {stats.total}",
    ]
    if stats.stale_deleted:
        lines.append(f"Удалено из индекса как исчезнувшее: {stats.stale_deleted}")
    if stats.prune_skipped:
        lines.append("")
        lines.append("Удаление пропущено для неполных обходов:")
        lines.extend(f"- {item}" for item in stats.prune_skipped)
    if stats.content:
        lines.append("")
        lines.extend(format_portal_content_sync_stats(stats.content).splitlines())
    if stats.errors:
        lines.append("")
        lines.append("Есть предупреждения:")
        lines.extend(f"- {error}" for error in stats.errors)
    return "\n".join(lines)


def format_portal_content_sync_stats(stats: PortalContentSyncStats) -> str:
    lines = [
        "Содержимое документов обработано.",
        f"Кандидатов: {stats.candidates}",
        f"Скачано: {stats.downloaded}",
        f"Текст добавлен: {stats.indexed}",
        f"Пропущено: {stats.skipped}",
        f"Неподдерживаемый формат: {stats.unsupported}",
        f"Ошибок: {stats.failed}",
    ]
    if stats.errors:
        lines.append("")
        lines.append("Ошибки по файлам:")
        lines.extend(f"- {error}" for error in stats.errors[:10])
    return "\n".join(lines)


def format_portal_delta_sync_stats(stats: PortalDeltaSyncStats) -> str:
    lines = [
        "Дельта-индексация Диска выполнена.",
        f"Папок проверено: {stats.folders_scanned}",
        f"Объектов увидено: {stats.items_seen}",
        f"Изменений: {stats.items_changed}",
        f"Изменённых файлов: {stats.files_changed}",
        f"Изменённых папок: {stats.folders_changed}",
        f"Удалено из индекса: {stats.deleted}",
    ]
    if stats.wrapped:
        lines.append("Курсор дошёл до конца списка папок и начал новый круг.")
    if stats.errors:
        lines.append("")
        lines.append("Есть предупреждения:")
        lines.extend(f"- {error}" for error in stats.errors[:10])
    return "\n".join(lines)


def format_portal_search_results(results: list[PortalSearchResult], *, query: str) -> str:
    if not results:
        return f"В индексе портала ничего не нашёл по запросу: {query}"

    lines = [f"Нашёл по порталу: {len(results)}"]
    for result in results:
        label = _entity_type_label(result.entity_type)
        title = f"[{result.title}]({result.url})" if result.url else result.title
        snippet = _make_snippet(result.body, query=query)
        lines.append(f"- {label}: {title}")
        if snippet:
            lines.append(f"  {snippet}")
    return "\n".join(lines)


def format_portal_index_stats(stats: PortalIndexStats) -> str:
    if not stats.exists:
        return f"Локальный индекс портала пока не найден: {stats.path}"
    lines = [
        f"В индексе портала: {stats.total_items} объектов.",
        f"Последнее обновление: {stats.last_indexed_at or 'нет данных'}",
    ]
    for entity_type, count in sorted(stats.by_type.items()):
        lines.append(f"- {_entity_type_label(entity_type)}: {count}")
    if stats.content_by_status:
        lines.append("")
        lines.append("Текст документов:")
        for status, count in sorted(stats.content_by_status.items()):
            lines.append(f"- {_content_status_label(status)}: {count}")
    return "\n".join(lines)


def entity_types_for_scope(scope: str) -> set[str] | None:
    normalized = scope.strip().lower()
    if normalized in {"", "all"}:
        return None
    return {
        "documents": {"disk_file", "task_attachment"},
        "files": {"disk_file", "disk_folder", "disk_storage", "task_attachment"},
        "tasks": {"task", "task_attachment"},
        "projects": {"project"},
    }.get(normalized)


async def _sync_catalog(bitrix: BitrixClient, index: PortalSearchIndex) -> dict[str, int]:
    settings = get_settings()
    products_count = 0
    stores_count = 0

    try:
        catalogs = await bitrix.list_catalogs()
    except Exception:
        catalogs = []

    for catalog in catalogs:
        iblock_id = _first(catalog, "iblockId", "IBLOCK_ID")
        if iblock_id is None:
            continue
        try:
            products = await bitrix.list_catalog_products(
                int(iblock_id), limit=settings.search_index_max_catalog_products
            )
        except Exception:
            continue
        for product in products:
            product_id = _first(product, "id", "ID")
            if product_id is None:
                continue
            name = str(_first(product, "name", "NAME") or f"Товар #{product_id}")
            body_parts = [
                str(_first(product, "previewText", "PREVIEW_TEXT") or ""),
                str(_first(product, "detailText", "DETAIL_TEXT") or ""),
                f"Каталог iblockId:{iblock_id}",
            ]
            index.upsert_item(
                entity_type="catalog_product",
                entity_id=product_id,
                title=name,
                body="\n".join(p for p in body_parts if p.strip()),
                url=_catalog_product_url(iblock_id, product_id),
                metadata={"iblock_id": iblock_id},
            )
            products_count += 1

    try:
        stores = await bitrix.list_catalog_stores()
    except Exception:
        stores = []

    for store in stores:
        store_id = _first(store, "id", "ID")
        if store_id is None:
            continue
        title = str(_first(store, "title", "TITLE") or f"Склад #{store_id}")
        address = str(_first(store, "address", "ADDRESS") or "")
        description = str(_first(store, "description", "DESCRIPTION") or "")
        index.upsert_item(
            entity_type="catalog_store",
            entity_id=store_id,
            title=title,
            body="\n".join(p for p in [address, description] if p.strip()),
            url=_catalog_store_url(store_id),
            metadata={
                "active": _first(store, "active", "ACTIVE"),
                "is_default": _first(store, "isDefault", "IS_DEFAULT"),
            },
        )
        stores_count += 1

    return {"products": products_count, "stores": stores_count}


async def _sync_tasks(bitrix: BitrixClient, index: PortalSearchIndex) -> dict[str, object]:
    settings = get_settings()
    tasks = await bitrix.list_all_tasks(
        select=[
            "ID",
            "TITLE",
            "DESCRIPTION",
            "STATUS",
            "RESPONSIBLE_ID",
            "CREATED_BY",
            "GROUP_ID",
            "DEADLINE",
            "CHANGED_DATE",
            "CLOSED_DATE",
            "UF_TASK_WEBDAV_FILES",
        ],
        order={"CHANGED_DATE": "DESC"},
        limit=settings.search_index_max_tasks,
    )
    indexed_attachments = 0
    seen_attachments: set[int] = set()
    for task in tasks:
        task_id = _first(task, "id", "ID")
        if task_id is None:
            continue
        title = str(_first(task, "title", "TITLE") or "Без названия")
        index.upsert_item(
            entity_type="task",
            entity_id=task_id,
            title=title,
            body="\n".join(
                str(value)
                for value in (
                    _first(task, "description", "DESCRIPTION"),
                    f"Статус: {_first(task, 'status', 'STATUS')}",
                    f"Исполнитель: {_first(task, 'responsibleId', 'RESPONSIBLE_ID')}",
                    f"Проект: {_first(task, 'groupId', 'GROUP_ID')}",
                    f"Срок: {_first(task, 'deadline', 'DEADLINE')}",
                )
                if value
            ),
            url=_task_url(task_id),
            metadata={
                "status": _first(task, "status", "STATUS"),
                "responsible_id": _first(task, "responsibleId", "RESPONSIBLE_ID"),
                "created_by": _first(task, "createdBy", "CREATED_BY"),
                "group_id": _first(task, "groupId", "GROUP_ID"),
                "deadline": _first(task, "deadline", "DEADLINE"),
            },
            source_updated_at=_to_str(_first(task, "changedDate", "CHANGED_DATE")),
        )

        if not settings.search_index_include_task_attachments:
            continue
        if indexed_attachments >= settings.search_index_max_task_attachments:
            continue
        attachment_ids = _attachment_ids(_first(task, "ufTaskWebdavFiles", "UF_TASK_WEBDAV_FILES"))
        for attached_object_id in attachment_ids:
            if attached_object_id in seen_attachments:
                continue
            seen_attachments.add(attached_object_id)
            try:
                attached = await bitrix.get_attached_object(attached_object_id)
            except Exception:
                continue
            if isinstance(attached, dict):
                _index_task_attachment(
                    index,
                    attached=attached,
                    task_id=task_id,
                    task_title=title,
                    task_updated_at=_to_str(_first(task, "changedDate", "CHANGED_DATE")),
                )
                indexed_attachments += 1
            if indexed_attachments >= settings.search_index_max_task_attachments:
                break
    return {
        "tasks": len(tasks),
        "attachments": indexed_attachments,
        "tasks_complete": len(tasks) < settings.search_index_max_tasks,
        "attachments_complete": (
            not settings.search_index_include_task_attachments
            or indexed_attachments < settings.search_index_max_task_attachments
        ),
    }


async def _sync_projects(bitrix: BitrixClient, index: PortalSearchIndex) -> int:
    settings = get_settings()
    projects = await bitrix.search_projects("", limit=settings.search_index_max_projects)
    if not isinstance(projects, list):
        return 0
    for project in projects:
        project_id = _first(project, "ID", "id")
        if project_id is None:
            continue
        name = str(_first(project, "NAME", "name") or f"Проект #{project_id}")
        index.upsert_item(
            entity_type="project",
            entity_id=project_id,
            title=name,
            body="\n".join(
                str(value)
                for value in (
                    _first(project, "DESCRIPTION", "description"),
                    f"Проект: {name}",
                    f"Владелец: {_first(project, 'OWNER_ID', 'ownerId')}",
                )
                if value
            ),
            url=_project_url(project_id),
            metadata={
                "owner_id": _first(project, "OWNER_ID", "ownerId"),
                "active": _first(project, "ACTIVE", "active"),
                "project": _first(project, "PROJECT", "project"),
            },
            source_updated_at=_to_str(_first(project, "DATE_UPDATE", "dateUpdate")),
        )
    return len(projects)


async def _sync_disk(bitrix: BitrixClient, index: PortalSearchIndex) -> dict[str, object]:
    settings = get_settings()
    storages = await bitrix.list_disk_storages(limit=settings.search_index_max_storages)
    indexed_items = 0
    for storage in storages:
        storage_id = _first(storage, "ID", "id")
        root_id = _first(storage, "ROOT_OBJECT_ID", "rootObjectId")
        name = str(_first(storage, "NAME", "name") or f"Диск #{storage_id}")
        if storage_id is None:
            continue

        index.upsert_item(
            entity_type="disk_storage",
            entity_id=storage_id,
            title=name,
            body=f"Хранилище Bitrix Disk: {name}",
            url="",
            metadata={
                "storage_id": storage_id,
                "root_object_id": root_id,
                "entity_type": _first(storage, "ENTITY_TYPE", "entityType"),
                "entity_id": _first(storage, "ENTITY_ID", "entityId"),
            },
        )
        indexed_items += 1

        if root_id is None or indexed_items >= settings.search_index_max_disk_items:
            continue
        indexed_items += await _sync_disk_folder(
            bitrix,
            index,
            folder_id=int(root_id),
            storage_name=name,
            path=name,
            depth=0,
            remaining=settings.search_index_max_disk_items - indexed_items,
        )
        if indexed_items >= settings.search_index_max_disk_items:
            break
    return {
        "storages": len(storages),
        "items": indexed_items,
        "complete": (
            len(storages) < settings.search_index_max_storages and indexed_items < settings.search_index_max_disk_items
        ),
    }


async def _sync_disk_folder(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    folder_id: int,
    storage_name: str,
    path: str,
    depth: int,
    remaining: int,
) -> int:
    settings = get_settings()
    if remaining <= 0 or depth > settings.search_index_disk_max_depth:
        return 0

    children = await bitrix.list_disk_folder_children_all(folder_id=folder_id, limit=remaining)
    count = 0
    for child in children:
        item_id = _first(child, "ID", "id")
        if item_id is None:
            continue
        name = str(_first(child, "NAME", "name") or f"Объект #{item_id}")
        item_type = str(_first(child, "TYPE", "type") or "").lower()
        child_path = f"{path}/{name}"
        entity_type = "disk_folder" if item_type == "folder" else "disk_file"
        detail_url = _normalize_url(_to_str(_first(child, "DETAIL_URL", "detailUrl")))
        index.upsert_item(
            entity_type=entity_type,
            entity_id=item_id,
            title=name,
            body="\n".join(
                str(value)
                for value in (
                    f"Диск: {storage_name}",
                    f"Путь: {child_path}",
                    f"Тип: {item_type}",
                )
                if value
            ),
            url=detail_url or _disk_object_url(item_id),
            metadata={
                "type": item_type,
                "path": child_path,
                "storage_name": storage_name,
                "parent_id": folder_id,
                "disk_object_id": item_id,
                "detail_url": detail_url,
                "size": _first(child, "SIZE", "size"),
                "created_by": _first(child, "CREATED_BY", "createdBy"),
                "updated_by": _first(child, "UPDATED_BY", "updatedBy"),
            },
            source_updated_at=_to_str(_first(child, "UPDATE_TIME", "updateTime")),
        )
        count += 1
        if entity_type == "disk_folder" and depth < settings.search_index_disk_max_depth:
            count += await _sync_disk_folder(
                bitrix,
                index,
                folder_id=int(item_id),
                storage_name=storage_name,
                path=child_path,
                depth=depth + 1,
                remaining=remaining - count,
            )
        if count >= remaining:
            break
    return count


async def _sync_disk_folder_delta(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    folder_id: int,
    storage_name: str,
    path: str,
    child_limit: int,
) -> dict[str, int]:
    children = await bitrix.list_disk_folder_children_all(folder_id=folder_id, limit=child_limit)
    seen_ids: set[str] = set()
    items_seen = 0
    items_changed = 0
    files_changed = 0
    folders_changed = 0

    for child in children:
        item_id = _first(child, "ID", "id")
        if item_id is None:
            continue
        seen_ids.add(str(item_id))
        items_seen += 1
        name = str(_first(child, "NAME", "name") or f"Объект #{item_id}")
        item_type = str(_first(child, "TYPE", "type") or "").lower()
        child_path = f"{path}/{name}" if path else name
        entity_type = "disk_folder" if item_type == "folder" else "disk_file"
        detail_url = _normalize_url(_to_str(_first(child, "DETAIL_URL", "detailUrl")))
        source_updated_at = _to_str(_first(child, "UPDATE_TIME", "updateTime"))
        metadata = {
            "type": item_type,
            "path": child_path,
            "storage_name": storage_name,
            "parent_id": folder_id,
            "disk_object_id": item_id,
            "detail_url": detail_url,
            "size": _first(child, "SIZE", "size"),
            "created_by": _first(child, "CREATED_BY", "createdBy"),
            "updated_by": _first(child, "UPDATED_BY", "updatedBy"),
            "delta_synced_at": datetime.now(MOSCOW_TZ).isoformat(),
        }
        snapshot = index.item_snapshot(entity_type=entity_type, entity_id=item_id)
        changed = _disk_delta_item_changed(
            snapshot=snapshot,
            new_metadata=metadata,
            new_source_updated_at=source_updated_at,
        )
        index.upsert_item(
            entity_type=entity_type,
            entity_id=item_id,
            title=name,
            body="\n".join(
                str(value)
                for value in (
                    f"Диск: {storage_name}",
                    f"Путь: {child_path}",
                    f"Тип: {item_type}",
                )
                if value
            ),
            url=detail_url or _disk_object_url(item_id),
            metadata=metadata,
            source_updated_at=source_updated_at,
        )
        if changed:
            items_changed += 1
            if entity_type == "disk_file":
                files_changed += 1
            else:
                folders_changed += 1

    deleted = 0
    if not child_limit or len(children) < child_limit:
        deleted = _delete_missing_delta_children(index, parent_id=folder_id, seen_ids=seen_ids)
    return {
        "items_seen": items_seen,
        "items_changed": items_changed,
        "files_changed": files_changed,
        "folders_changed": folders_changed,
        "deleted": deleted,
    }


def _index_task_attachment(
    index: PortalSearchIndex,
    *,
    attached: dict[str, Any],
    task_id: object,
    task_title: str,
    task_updated_at: str | None,
) -> None:
    attached_id = _first(attached, "ID", "id")
    object_id = _first(attached, "OBJECT_ID", "objectId")
    name = str(_first(attached, "NAME", "name") or f"Вложение #{attached_id}")
    index.upsert_item(
        entity_type="task_attachment",
        entity_id=attached_id,
        title=name,
        body="\n".join(
            str(value)
            for value in (
                f"Вложение задачи: {task_title}",
                f"Задача: #{task_id}",
                f"Имя файла: {name}",
                f"Размер: {_first(attached, 'SIZE', 'size')}",
            )
            if value
        ),
        url=_task_url(task_id),
        metadata={
            "task_id": task_id,
            "task_title": task_title,
            "attached_object_id": attached_id,
            "disk_object_id": object_id,
            "size": _first(attached, "SIZE", "size"),
            "created_by": _first(attached, "CREATED_BY", "createdBy"),
            "create_time": _first(attached, "CREATE_TIME", "createTime"),
            "download_available": bool(_first(attached, "DOWNLOAD_URL", "downloadUrl")),
        },
        source_updated_at=_to_str(_first(attached, "CREATE_TIME", "createTime")) or task_updated_at,
    )


def _delete_missing_delta_children(
    index: PortalSearchIndex,
    *,
    parent_id: int,
    seen_ids: set[str],
) -> int:
    deleted = 0
    for existing in index.children_by_parent_id(parent_id):
        if existing.entity_id in seen_ids:
            continue
        if existing.entity_type == "disk_file":
            delete_portal_file_cache_path(portal_file_cache_path(existing))
        if index.delete_item(entity_type=existing.entity_type, entity_id=existing.entity_id):
            deleted += 1
    return deleted


def _disk_delta_item_changed(
    *,
    snapshot: dict[str, Any] | None,
    new_metadata: dict[str, Any],
    new_source_updated_at: object,
) -> bool:
    if not snapshot:
        return True
    existing_source = _to_str(snapshot.get("source_updated_at"))
    new_source = _to_str(new_source_updated_at)
    if existing_source or new_source:
        return existing_source != new_source
    existing_metadata = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {}
    return (
        _safe_int(existing_metadata.get("size")) != _safe_int(new_metadata.get("size"))
        or _to_str(existing_metadata.get("path")) != _to_str(new_metadata.get("path"))
        or _to_str(existing_metadata.get("detail_url")) != _to_str(new_metadata.get("detail_url"))
    )


def _delta_folder_id(folder: PortalSearchResult) -> int | None:
    if folder.entity_type == "disk_storage":
        return _safe_int(folder.metadata.get("root_object_id"))
    return _safe_int(folder.metadata.get("disk_object_id")) or _safe_int(folder.entity_id)


def _delta_storage_name(folder: PortalSearchResult) -> str:
    return _to_str(folder.metadata.get("storage_name")) or folder.title


def _delta_folder_path(folder: PortalSearchResult) -> str:
    return _to_str(folder.metadata.get("path")) or folder.title


def _row_to_search_result(row: sqlite3.Row) -> PortalSearchResult:
    return PortalSearchResult(
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        title=row["title"],
        body=row["body"],
        url=row["url"],
        score=0,
        metadata=_safe_json(row["metadata_json"]),
    )


def _score_result(
    normalized_query: str,
    terms: list[str],
    *,
    title: str,
    body: str,
    search_text: str,
) -> int:
    score = 0
    if normalized_query and normalized_query in title:
        score += 80
    if normalized_query and normalized_query in body:
        score += 30
    for term in terms:
        if term in title:
            score += 12
        if term in body:
            score += 4
        if term in search_text:
            score += 1
    return score


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[\w#№.-]+", _normalize_search_text(query), flags=re.UNICODE)
    return [term for term in terms if len(term) > 1 and term not in SEARCH_STOP_WORDS]


def _query_term_groups(query: str) -> list[list[str]]:
    return [_search_variants(term) for term in _query_terms(query)]


def _search_variants(term: str) -> list[str]:
    variants = [term]
    if len(term) > 4 and term[-1:] in {"а", "ы", "и", "у", "е", "о"}:
        stem = term[:-1]
        variants.extend([stem, stem + "а", stem + "ы", stem + "и", stem + "у", stem + "е"])
    if len(term) > 5 and term.endswith(("ам", "ям", "ах", "ях", "ой", "ей", "ом", "ем")):
        stem = term[:-2]
        variants.extend([stem, stem + "а", stem + "ы", stem + "и", stem + "у", stem + "е"])
    return _flatten_unique([variants])


def _flatten_unique(groups: list[list[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            cleaned = value.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", _clean_text(text).lower().replace("ё", "е"))


def _clean_text(text: object) -> str:
    if text is None:
        return ""
    value = re.sub(r"\[/?[a-zA-Z0-9_]+(?:=[^\]]*)?\]", "", str(text))
    value = re.sub(r"<[^>]+>", "", value)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _safe_json(value: object) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _make_snippet(body: str, *, query: str = "", max_length: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", body).strip()
    if len(cleaned) <= max_length:
        return cleaned

    terms = _flatten_unique(_query_term_groups(query)) if query else []
    lowered = cleaned.lower().replace("ё", "е")
    positions = [lowered.find(term) for term in terms if term and lowered.find(term) >= 0]
    if positions:
        position = min(positions)
        start = max(0, position - max_length // 3)
        end = min(len(cleaned), start + max_length)
        start = max(0, end - max_length)
        snippet = cleaned[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(cleaned):
            snippet = snippet.rstrip() + "..."
        return snippet

    return cleaned[: max_length - 3].rstrip() + "..."


def _body_with_content(body: str, content_text: str) -> str:
    base_body = body.split("\n\nТекст файла:\n", 1)[0].strip()
    if base_body:
        return f"{base_body}\n\nТекст файла:\n{content_text}"
    return f"Текст файла:\n{content_text}"


def _content_text_from_body(body: str) -> str:
    marker = "\n\nТекст файла:\n"
    if marker in body:
        return body.split(marker, 1)[1].strip()
    if body.startswith("Текст файла:\n"):
        return body.split("Текст файла:\n", 1)[1].strip()
    return ""


def _merge_content_metadata(
    *,
    base: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in existing.items():
        if key.startswith("content_"):
            merged[key] = value
    return merged


def _should_preserve_content(
    *,
    existing_metadata: dict[str, Any],
    new_metadata: dict[str, Any],
    existing_source_updated_at: object,
    new_source_updated_at: object,
) -> bool:
    if not existing_metadata.get("content_index_status"):
        return False
    existing_source = _to_str(existing_source_updated_at)
    new_source = _to_str(new_source_updated_at)
    if existing_source or new_source:
        return existing_source == new_source
    existing_size = _safe_int(existing_metadata.get("size"))
    new_size = _safe_int(new_metadata.get("size"))
    return existing_size is not None and existing_size == new_size


def _file_extension(name: str) -> str:
    return Path(name).suffix.lower()


def _normalize_extensions(extensions: set[str]) -> set[str]:
    return {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
        if extension
    }


def _safe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _normalize_url(value: str | None) -> str:
    if not value:
        return ""
    return quote(value.strip(), safe=":/?#[]@!$&'*,;=%")


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _attachment_ids(value: object) -> list[int]:
    if value is None or value == "":
        return []
    raw_items = value if isinstance(value, list) else [value]
    ids = []
    for item in raw_items:
        normalized = str(item).strip().removeprefix("n")
        if normalized.isdigit():
            ids.append(int(normalized))
    return ids


def _to_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _entity_type_label(entity_type: str) -> str:
    return {
        "task": "Задача",
        "task_attachment": "Вложение задачи",
        "project": "Проект",
        "disk_storage": "Хранилище",
        "disk_folder": "Папка",
        "disk_file": "Файл",
    }.get(entity_type, entity_type)


def _content_status_label(status: str) -> str:
    return {
        "none": "ещё не брал в обработку",
        "indexed": "текст извлечён",
        "unsupported": "формат пока не поддержан",
        "too_large": "слишком большой файл",
        "empty": "текст не найден",
        "failed": "ошибка извлечения",
        "no_download_url": "нет ссылки скачивания",
    }.get(status, status)


def _task_url(task_id: object) -> str:
    domain = _portal_domain()
    if not domain:
        return f"/company/personal/user/0/tasks/task/view/{task_id}/"
    return f"https://{domain}/company/personal/user/0/tasks/task/view/{task_id}/"


def _project_url(project_id: object) -> str:
    domain = _portal_domain()
    if not domain:
        return f"/workgroups/group/{project_id}/"
    return f"https://{domain}/workgroups/group/{project_id}/"


def _catalog_product_url(iblock_id: object, product_id: object) -> str:
    domain = _portal_domain()
    if not domain:
        return f"/shop/documents-catalog/{iblock_id}/product/{product_id}/"
    return f"https://{domain}/shop/documents-catalog/{iblock_id}/product/{product_id}/"


def _catalog_store_url(store_id: object) -> str:
    domain = _portal_domain()
    if not domain:
        return f"/crm/store/detail/{store_id}/"
    return f"https://{domain}/crm/store/detail/{store_id}/"


def _disk_object_url(object_id: object) -> str:
    domain = _portal_domain()
    if not domain:
        return f"/docs/file/{object_id}/"
    return f"https://{domain}/docs/file/{object_id}/"


def _portal_domain() -> str:
    settings = get_settings()
    candidates = (
        settings.bitrix_domain,
        settings.bitrix_rest_webhook_url,
        settings.bitrix_projects_webhook_url,
    )
    for candidate in candidates:
        domain = _domain_from_value(candidate)
        if domain:
            return domain
    return ""


def _domain_from_value(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if "://" not in cleaned:
        cleaned = "https://" + cleaned
    parts = urlsplit(cleaned)
    return parts.netloc.strip().rstrip("/")


SEARCH_STOP_WORDS = {
    "а",
    "в",
    "во",
    "все",
    "всем",
    "всех",
    "всю",
    "где",
    "для",
    "документ",
    "документы",
    "документа",
    "документов",
    "и",
    "или",
    "любые",
    "мне",
    "на",
    "найди",
    "найти",
    "папка",
    "папки",
    "папку",
    "по",
    "поищи",
    "покажи",
    "портал",
    "портале",
    "порталу",
    "проект",
    "проекта",
    "проектам",
    "проектами",
    "проектах",
    "проектов",
    "проекты",
    "список",
    "файл",
    "файлы",
    "файлов",
}
