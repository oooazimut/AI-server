from __future__ import annotations

import re
from datetime import UTC, datetime

from .agent_schema import PostgresAgentSchema


def _parse_id_set(raw: str) -> set[int | str]:
    result: set[int | str] = set()
    for item in (raw or "").replace(";", ",").split(","):
        value = item.strip()
        if not value:
            continue
        if value == "*":
            result.add("*")
        else:
            try:
                result.add(int(value))
            except ValueError:
                pass
    return result


def _norm(value: str) -> str:
    return value.casefold().replace("ё", "е")


def _tokens(value: str) -> list[str]:
    return [t for t in re.findall(r"[0-9a-zа-яё_.\-]{2,}", _norm(value)) if t]


def _score(haystack: str, *, query_norm: str, tokens: list[str]) -> int:
    score = 0
    if query_norm and query_norm in haystack:
        score += 10
    for token in tokens:
        if token in haystack:
            score += 1 + min(haystack.count(token), 5)
    return score


def _snippet(content: str, *, query_norm: str, tokens: list[str], max_chars: int = 300) -> str:
    if not content:
        return ""
    normalized = _norm(content)
    positions = [normalized.find(query_norm)] if query_norm else []
    positions += [normalized.find(t) for t in tokens]
    positions = [p for p in positions if p >= 0]
    start = max(0, min(positions) - 80) if positions else 0
    snippet = content[start : start + max_chars].strip()
    if start > 0:
        snippet = "..." + snippet
    if start + max_chars < len(content):
        snippet += "..."
    return snippet


class PostgresKartotekaStore(PostgresAgentSchema):
    """Kartoteka agent store: dialog_history + chunk-based file_index in 'kartoteka' schema."""

    _SCHEMA = "kartoteka"

    def __init__(self, url: str, *, protected_user_ids: str = "", secret_user_ids: str = "") -> None:
        super().__init__(url)
        self._protected_ids = _parse_id_set(protected_user_ids)
        self._secret_ids = _parse_id_set(secret_user_ids)

    def _can_read_protected(self, user_id: int | None) -> bool:
        return "*" in self._protected_ids or (user_id is not None and user_id in self._protected_ids)

    def _can_read_secret(self, user_id: int | None) -> bool:
        return "*" in self._secret_ids or (user_id is not None and user_id in self._secret_ids)

    async def ensure_schema(self) -> None:
        await super().ensure_schema()
        async with await self._connect() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS kartoteka.file_index (
                    id BIGSERIAL PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    chunk_index INT NOT NULL DEFAULT 0,
                    document_id TEXT NOT NULL DEFAULT '',
                    relative_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    extension TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL DEFAULT '',
                    access_level TEXT NOT NULL DEFAULT 'open',
                    group_id TEXT NOT NULL DEFAULT '',
                    group_name TEXT NOT NULL DEFAULT '',
                    search_text TEXT NOT NULL,
                    size_bytes BIGINT NOT NULL DEFAULT 0,
                    modified_time TEXT NOT NULL DEFAULT '',
                    indexed_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kartoteka_search_text
                ON kartoteka.file_index (search_text)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kartoteka_access_level
                ON kartoteka.file_index (access_level)
                """
            )

    async def search(self, query: str, *, user_id: int | None = None, limit: int = 10) -> list[dict]:
        allowed: list[str] = ["open"]
        if self._can_read_protected(user_id):
            allowed.append("protected")

        query_norm = _norm(query)
        tokens = _tokens(query)

        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT chunk_id, relative_path, filename, extension,
                       text, access_level, group_name, chunk_index
                FROM kartoteka.file_index
                WHERE search_text ILIKE %s AND access_level = ANY(%s)
                LIMIT %s
                """,
                (f"%{query}%", allowed, limit * 3),
            )
            rows = await cur.fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            haystack = _norm(" ".join(filter(None, [row["filename"], row["group_name"], row["text"]])))
            s = _score(haystack, query_norm=query_norm, tokens=tokens)
            snippet = _snippet(row["text"], query_norm=query_norm, tokens=tokens)
            scored.append((s, {**row, "snippet": snippet, "text": None}))

        scored.sort(key=lambda x: -x[0])
        return [item for _, item in scored[:limit]]

    async def stats(self) -> dict:
        async with await self._connect() as db:
            cur = await db.execute(
                "SELECT COUNT(*) AS total_chunks, COUNT(DISTINCT document_id) AS total_documents "
                "FROM kartoteka.file_index"
            )
            row = await cur.fetchone()
            if not row:
                return {"total_chunks": 0, "total_documents": 0}
            return {"total_chunks": row["total_chunks"], "total_documents": row["total_documents"]}

    def upsert_chunk(
        self,
        chunk_id: str,
        *,
        chunk_index: int = 0,
        document_id: str = "",
        relative_path: str,
        filename: str,
        extension: str = "",
        text: str = "",
        access_level: str = "open",
        group_id: str = "",
        group_name: str = "",
        size_bytes: int = 0,
        modified_time: str = "",
        indexed_at: str = "",
    ) -> None:
        search_text = " ".join(filter(None, [filename, group_name, text]))
        now = datetime.now(UTC).isoformat()
        with self._sync_connect() as db:
            db.execute(
                """
                INSERT INTO kartoteka.file_index
                    (chunk_id, chunk_index, document_id, relative_path, filename, extension,
                     text, access_level, group_id, group_name, search_text,
                     size_bytes, modified_time, indexed_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    chunk_index = EXCLUDED.chunk_index,
                    relative_path = EXCLUDED.relative_path,
                    filename = EXCLUDED.filename,
                    extension = EXCLUDED.extension,
                    text = EXCLUDED.text,
                    access_level = EXCLUDED.access_level,
                    group_id = EXCLUDED.group_id,
                    group_name = EXCLUDED.group_name,
                    search_text = EXCLUDED.search_text,
                    size_bytes = EXCLUDED.size_bytes,
                    modified_time = EXCLUDED.modified_time,
                    indexed_at = EXCLUDED.indexed_at,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    chunk_id,
                    chunk_index,
                    document_id,
                    relative_path,
                    filename,
                    extension,
                    text,
                    access_level,
                    group_id,
                    group_name,
                    search_text,
                    size_bytes,
                    modified_time,
                    indexed_at,
                    now,
                    now,
                ),
            )
