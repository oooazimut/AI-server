from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_server.agent_store import SqliteStore
from ai_server.integrations.bitrix.portal_search.text_utils import (
    body_with_content,
    clean_text,
    content_text_from_body,
    escape_like,
    file_extension,
    flatten_unique,
    increment,
    normalize_extensions,
    normalize_search_text,
    query_term_groups,
    safe_int,
    safe_json,
)
from ai_server.integrations.bitrix.portal_search.types import (
    CONTENT_INDEX_VERSION,
    CONTENT_TERMINAL_STATUSES,
    PortalContentReadiness,
    PortalIndexStats,
    PortalSearchResult,
)
from ai_server.runtime import runtime_paths
from ai_server.utils import MOSCOW_TZ


class PortalSearchIndex(SqliteStore):
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
        normalized_title = clean_text(title) or f"{entity_type} #{entity_id}"
        normalized_body = clean_text(body)
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
            existing_content = content_text_from_body(existing["body"])
            if existing_content:
                normalized_body = body_with_content(normalized_body, existing_content)

        search_text = normalize_search_text(" ".join([entity_type, str(entity_id), normalized_title, normalized_body]))
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
        term_groups = query_term_groups(query)
        terms = flatten_unique(term_groups)
        if not term_groups:
            return []

        where = []
        params: list[object] = []
        for group in term_groups:
            group_where = []
            for term in group:
                group_where.append("search_text LIKE ? ESCAPE '\\'")
                params.append(f"%{escape_like(term)}%")
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

        normalized_query = normalize_search_text(query)
        scored: list[PortalSearchResult] = []
        for row in rows:
            metadata = safe_json(row["metadata_json"])
            score = _score_result(
                normalized_query,
                terms,
                title=normalize_search_text(row["title"]),
                body=normalize_search_text(row["body"]),
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
        normalized_id = safe_int(cursor_id) or 0
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
                    f'%"parent_id": "{escape_like(parent)}"%',
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
        normalized_allowed_extensions = normalize_extensions(allowed_extensions)
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
            extension = file_extension(str(row["title"] or "")) or "<none>"
            metadata = safe_json(row["metadata_json"])
            status = str(metadata.get("content_index_status") or "none")
            content_version = str(metadata.get("content_index_version") or "")

            if extension not in normalized_allowed_extensions:
                unsupported += 1
                increment(unsupported_by_extension, extension)
                continue

            supported_documents += 1
            if status == "indexed":
                indexed += 1
                increment(indexed_by_extension, extension)
            elif content_version == CONTENT_INDEX_VERSION and status in CONTENT_TERMINAL_STATUSES:
                terminal += 1
                increment(terminal_by_status, status)
            else:
                pending += 1
                increment(pending_by_extension, extension)
                increment(pending_by_status, status)

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
        normalized_body = clean_text(body)
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
            search_text = normalize_search_text(" ".join([entity_type, str(entity_id), row["title"], normalized_body]))
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
            status = safe_json(row["metadata_json"]).get("content_index_status")
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
            "metadata": safe_json(row["metadata_json"]),
            "source_updated_at": row["source_updated_at"],
        }


def _row_to_search_result(row: sqlite3.Row) -> PortalSearchResult:
    return PortalSearchResult(
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        title=row["title"],
        body=row["body"],
        url=row["url"],
        score=0,
        metadata=safe_json(row["metadata_json"]),
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
    existing_source = to_str(existing_source_updated_at)
    new_source = to_str(new_source_updated_at)
    if existing_source or new_source:
        return existing_source == new_source
    existing_size = safe_int(existing_metadata.get("size"))
    new_size = safe_int(new_metadata.get("size"))
    return existing_size is not None and existing_size == new_size


def to_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
