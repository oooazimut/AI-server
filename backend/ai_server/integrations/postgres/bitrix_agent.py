from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_server.integrations.bitrix.portal_search.search_index import (
    _merge_content_metadata,
    _score_result,
    _should_preserve_content,
)
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
from ai_server.utils import MOSCOW_TZ

from .agent_schema import PostgresAgentSchema

_TABLE = "bitrix24.portal_search_items"


class PostgresBitrixAgentStore(PostgresAgentSchema):
    """Bitrix24 agent store: owns all Bitrix24 agent data in the 'bitrix24' schema.

    Tables: dialog_history, incomplete_proposals, portal_search_items.
    Satisfies AgentStorePort (dialogs) and provides portal search methods.
    """

    _SCHEMA = "bitrix24"

    async def ensure_schema(self) -> None:
        await super().ensure_schema()  # creates bitrix24 schema + dialog_history table
        with self._sync_connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix24.incomplete_proposals (
                    id SERIAL PRIMARY KEY,
                    task_id INTEGER NOT NULL,
                    task_title TEXT,
                    missing_parts TEXT,
                    responsible_id INTEGER,
                    responsible_dialog_id TEXT,
                    responsible_response TEXT,
                    status TEXT NOT NULL DEFAULT 'awaiting_response',
                    created_at TEXT,
                    scheduled_for TEXT
                )
                """
            )
            db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    entity_type      TEXT NOT NULL,
                    entity_id        TEXT NOT NULL,
                    title            TEXT NOT NULL,
                    body             TEXT NOT NULL,
                    url              TEXT NOT NULL,
                    search_text      TEXT NOT NULL,
                    metadata_json    TEXT NOT NULL,
                    source_updated_at TEXT,
                    last_seen_at     TEXT,
                    indexed_at       TEXT NOT NULL,
                    PRIMARY KEY (entity_type, entity_id)
                )
                """
            )
            db.execute(f"CREATE INDEX IF NOT EXISTS idx_psi_type    ON {_TABLE}(entity_type)")
            db.execute(f"CREATE INDEX IF NOT EXISTS idx_psi_indexed ON {_TABLE}(indexed_at)")
            db.execute(f"CREATE INDEX IF NOT EXISTS idx_psi_seen    ON {_TABLE}(last_seen_at)")

    def save_proposal(
        self,
        *,
        task_id: int,
        task_title: str = "",
        missing_parts: str = "",
        responsible_id: int | None = None,
        responsible_dialog_id: str = "",
        scheduled_for: str = "",
    ) -> int:
        with self._sync_connect() as db:
            row = db.execute(
                """
                INSERT INTO bitrix24.incomplete_proposals
                    (task_id, task_title, missing_parts, responsible_id, responsible_dialog_id,
                     status, created_at, scheduled_for)
                VALUES (%s, %s, %s, %s, %s, 'awaiting_response', %s, %s)
                RETURNING id
                """,
                (
                    task_id,
                    task_title,
                    missing_parts,
                    responsible_id,
                    responsible_dialog_id,
                    datetime.now(UTC).isoformat(),
                    scheduled_for,
                ),
            ).fetchone()
        return int(row["id"]) if row else 0

    def get_proposal_by_id(self, proposal_id: int) -> dict[str, Any] | None:
        with self._sync_connect() as db:
            row = db.execute("SELECT * FROM bitrix24.incomplete_proposals WHERE id = %s", (proposal_id,)).fetchone()
        return dict(row) if row else None

    def get_proposals_for_manager(self) -> list[dict[str, Any]]:
        with self._sync_connect() as db:
            rows = db.execute(
                """
                SELECT * FROM bitrix24.incomplete_proposals
                WHERE status IN ('awaiting_response', 'proposed')
                ORDER BY created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_pending_for_responsible(self, responsible_id: int) -> dict[str, Any] | None:
        with self._sync_connect() as db:
            row = db.execute(
                """
                SELECT * FROM bitrix24.incomplete_proposals
                WHERE responsible_id = %s AND status = 'awaiting_response'
                ORDER BY created_at LIMIT 1
                """,
                (responsible_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_responsible_response(self, proposal_id: int, response_text: str) -> None:
        with self._sync_connect() as db:
            db.execute(
                "UPDATE bitrix24.incomplete_proposals SET responsible_response = %s WHERE id = %s",
                (response_text, proposal_id),
            )

    def mark_status(self, proposal_id: int, status: str) -> None:
        with self._sync_connect() as db:
            db.execute(
                "UPDATE bitrix24.incomplete_proposals SET status = %s WHERE id = %s",
                (status, proposal_id),
            )

    def delete_proposal(self, proposal_id: int) -> None:
        with self._sync_connect() as db:
            db.execute("DELETE FROM bitrix24.incomplete_proposals WHERE id = %s", (proposal_id,))

    # ------------------------------------------------------------------
    # Portal search index
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        try:
            with self._sync_connect() as db:
                row = db.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'bitrix24' AND table_name = 'portal_search_items'",
                ).fetchone()
            return bool(row)
        except Exception:
            return False

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
            normalized_metadata = _merge_content_metadata(base=normalized_metadata, existing=existing["metadata"])
            existing_content = content_text_from_body(existing["body"])
            if existing_content:
                normalized_body = body_with_content(normalized_body, existing_content)

        search_text = normalize_search_text(" ".join([entity_type, str(entity_id), normalized_title, normalized_body]))
        with self._sync_connect() as db:
            db.execute(
                f"""
                INSERT INTO {_TABLE} (
                    entity_type, entity_id, title, body, url,
                    search_text, metadata_json, source_updated_at, last_seen_at, indexed_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (entity_type, entity_id) DO UPDATE SET
                    title             = EXCLUDED.title,
                    body              = EXCLUDED.body,
                    url               = EXCLUDED.url,
                    search_text       = EXCLUDED.search_text,
                    metadata_json     = EXCLUDED.metadata_json,
                    source_updated_at = EXCLUDED.source_updated_at,
                    last_seen_at      = EXCLUDED.last_seen_at,
                    indexed_at        = EXCLUDED.indexed_at
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
        with self._sync_connect() as db:
            cur = db.execute(
                f"DELETE FROM {_TABLE} WHERE entity_type = %s AND entity_id = %s",
                (entity_type, str(entity_id)),
            )
        return bool(cur.rowcount)

    def delete_stale_items(self, *, entity_types: set[str], seen_before: str) -> int:
        if not entity_types:
            return 0
        with self._sync_connect() as db:
            cur = db.execute(
                f"""
                DELETE FROM {_TABLE}
                WHERE entity_type = ANY(%s)
                  AND (last_seen_at IS NULL OR last_seen_at < %s)
                """,
                (list(sorted(entity_types)), seen_before),
            )
        return int(cur.rowcount or 0)

    def search(
        self,
        query: str,
        *,
        entity_types: set[str] | None = None,
        limit: int = 10,
    ) -> list[PortalSearchResult]:
        term_groups = query_term_groups(query)
        terms = flatten_unique(term_groups)
        if not term_groups:
            return []

        where: list[str] = []
        params: list[object] = []
        for group in term_groups:
            group_where = []
            for term in group:
                group_where.append("search_text LIKE %s ESCAPE '\\'")
                params.append(f"%{escape_like(term)}%")
            where.append("(" + " OR ".join(group_where) + ")")
        if entity_types:
            where.append("entity_type = ANY(%s)")
            params.append(list(sorted(entity_types)))

        query_sql = (
            f"SELECT entity_type, entity_id, title, body, url, metadata_json, search_text "
            f"FROM {_TABLE} WHERE " + " AND ".join(where) + " LIMIT 500"
        )
        with self._sync_connect() as db:
            rows = db.execute(query_sql, params).fetchall()

        normalized_query = normalize_search_text(query)
        scored: list[PortalSearchResult] = []
        for row in rows:
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
                    metadata=safe_json(row["metadata_json"]),
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
        normalized_type = cursor_type or ""
        normalized_id = safe_int(cursor_id) or 0
        rows = self._select_disk_delta_folder_rows(cursor_type=normalized_type, cursor_id=normalized_id, limit=limit)
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
        parent = str(parent_id)
        with self._sync_connect() as db:
            rows = db.execute(
                f"""
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM {_TABLE}
                WHERE entity_type IN ('disk_file', 'disk_folder')
                  AND (
                    metadata_json LIKE %s ESCAPE '\\'
                    OR metadata_json LIKE %s ESCAPE '\\'
                  )
                """,
                (
                    f'%"parent_id": "{escape_like(parent)}"%',
                    f'%"parent_id": {parent}%',
                ),
            ).fetchall()
        return [_row_to_search_result(row) for row in rows]

    def content_candidates(self, *, limit: int) -> list[PortalSearchResult]:
        with self._sync_connect() as db:
            rows = db.execute(
                f"""
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM {_TABLE}
                WHERE entity_type IN ('disk_file', 'task_attachment')
                ORDER BY indexed_at DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [_row_to_search_result(row) for row in rows]

    def content_readiness(self, *, allowed_extensions: set[str]) -> PortalContentReadiness:
        normalized_allowed = normalize_extensions(allowed_extensions)
        with self._sync_connect() as db:
            rows = db.execute(
                f"SELECT title, metadata_json FROM {_TABLE} WHERE entity_type IN ('disk_file', 'task_attachment')"
            ).fetchall()

        indexed_by_extension: dict[str, int] = {}
        pending_by_extension: dict[str, int] = {}
        pending_by_status: dict[str, int] = {}
        terminal_by_status: dict[str, int] = {}
        unsupported_by_extension: dict[str, int] = {}
        total_documents = supported_documents = indexed = pending = terminal = unsupported = 0

        for row in rows:
            total_documents += 1
            ext = file_extension(str(row["title"] or "")) or "<none>"
            metadata = safe_json(row["metadata_json"])
            status = str(metadata.get("content_index_status") or "none")
            content_version = str(metadata.get("content_index_version") or "")

            if ext not in normalized_allowed:
                unsupported += 1
                increment(unsupported_by_extension, ext)
                continue

            supported_documents += 1
            if status == "indexed":
                indexed += 1
                increment(indexed_by_extension, ext)
            elif content_version == CONTENT_INDEX_VERSION and status in CONTENT_TERMINAL_STATUSES:
                terminal += 1
                increment(terminal_by_status, status)
            else:
                pending += 1
                increment(pending_by_extension, ext)
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
        now = datetime.now(MOSCOW_TZ).isoformat()
        normalized_body = clean_text(body)
        with self._sync_connect() as db:
            row = db.execute(
                f"SELECT title FROM {_TABLE} WHERE entity_type = %s AND entity_id = %s",
                (entity_type, str(entity_id)),
            ).fetchone()
            if not row:
                return
            search_text = normalize_search_text(" ".join([entity_type, str(entity_id), row["title"], normalized_body]))
            db.execute(
                f"""
                UPDATE {_TABLE}
                SET body = %s, search_text = %s, metadata_json = %s, indexed_at = %s
                WHERE entity_type = %s AND entity_id = %s
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
        with self._sync_connect() as db:
            total = db.execute(f"SELECT COUNT(*) AS n FROM {_TABLE}").fetchone()["n"]
            by_type_rows = db.execute(
                f"SELECT entity_type, COUNT(*) AS count FROM {_TABLE} GROUP BY entity_type"
            ).fetchall()
            last_row = db.execute(f"SELECT MAX(indexed_at) AS t FROM {_TABLE}").fetchone()
            content_rows = db.execute(
                f"SELECT metadata_json FROM {_TABLE} WHERE entity_type IN ('disk_file', 'task_attachment')"
            ).fetchall()

        content_by_status: dict[str, int] = {}
        for row in content_rows:
            status = safe_json(row["metadata_json"]).get("content_index_status")
            if status:
                k = str(status)
                content_by_status[k] = content_by_status.get(k, 0) + 1

        return PortalIndexStats(
            total_items=int(total),
            by_type={str(r["entity_type"]): int(r["count"]) for r in by_type_rows},
            content_by_status=content_by_status,
            last_indexed_at=str(last_row["t"]) if last_row and last_row["t"] else None,
            path=Path("(postgresql)"),
            exists=True,
        )

    def get_item(self, *, entity_type: str, entity_id: object) -> PortalSearchResult | None:
        with self._sync_connect() as db:
            row = db.execute(
                f"""
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM {_TABLE} WHERE entity_type = %s AND entity_id = %s
                """,
                (entity_type, str(entity_id)),
            ).fetchone()
        return _row_to_search_result(row) if row else None

    def item_snapshot(self, *, entity_type: str, entity_id: object) -> dict[str, Any] | None:
        return self._get_existing_item(entity_type=entity_type, entity_id=entity_id)

    def _select_disk_delta_folder_rows(self, *, cursor_type: str, cursor_id: int, limit: int) -> list[dict]:
        with self._sync_connect() as db:
            return db.execute(
                f"""
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM {_TABLE}
                WHERE entity_type IN ('disk_storage', 'disk_folder')
                  AND (
                    %s = ''
                    OR entity_type > %s
                    OR (entity_type = %s AND CAST(entity_id AS INTEGER) > %s)
                  )
                ORDER BY entity_type, CAST(entity_id AS INTEGER)
                LIMIT %s
                """,
                (cursor_type, cursor_type, cursor_type, cursor_id, limit),
            ).fetchall()

    def _get_existing_item(self, *, entity_type: str, entity_id: object) -> dict[str, Any] | None:
        with self._sync_connect() as db:
            row = db.execute(
                f"""
                SELECT body, metadata_json, source_updated_at FROM {_TABLE}
                WHERE entity_type = %s AND entity_id = %s
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


def _row_to_search_result(row: dict) -> PortalSearchResult:
    return PortalSearchResult(
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        title=row["title"],
        body=row["body"],
        url=row["url"],
        score=0,
        metadata=safe_json(row["metadata_json"]),
    )
