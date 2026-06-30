from __future__ import annotations

from datetime import UTC, datetime

from .agent_schema import PostgresAgentSchema


class PostgresKartotekaStore(PostgresAgentSchema):
    """Kartoteka agent store: dialog_history + file_index in the 'kartoteka' schema."""

    _SCHEMA = "kartoteka"

    async def ensure_schema(self) -> None:
        await super().ensure_schema()
        async with await self._connect() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS kartoteka.file_index (
                    id BIGSERIAL PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    extension TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '',
                    content_preview TEXT NOT NULL DEFAULT '',
                    search_text TEXT NOT NULL,
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

    async def search(self, query: str, *, limit: int = 10) -> list[dict]:
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT id, path, filename, extension, tags, content_preview
                FROM kartoteka.file_index
                WHERE search_text ILIKE %s
                ORDER BY filename
                LIMIT %s
                """,
                (f"%{query}%", limit),
            )
            return await cur.fetchall()

    async def stats(self) -> dict:
        async with await self._connect() as db:
            cur = await db.execute("SELECT COUNT(*) AS total FROM kartoteka.file_index")
            row = await cur.fetchone()
            return {"total_files": row["total"] if row else 0}

    def upsert_file(
        self,
        path: str,
        filename: str,
        *,
        extension: str = "",
        tags: str = "",
        content_preview: str = "",
    ) -> None:
        search_text = " ".join(filter(None, [filename, tags, content_preview]))
        now = datetime.now(UTC).isoformat()
        with self._sync_connect() as db:
            db.execute(
                """
                INSERT INTO kartoteka.file_index
                    (path, filename, extension, tags, content_preview, search_text, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (path) DO UPDATE SET
                    filename = EXCLUDED.filename,
                    extension = EXCLUDED.extension,
                    tags = EXCLUDED.tags,
                    content_preview = EXCLUDED.content_preview,
                    search_text = EXCLUDED.search_text,
                    updated_at = EXCLUDED.updated_at
                """,
                (path, filename, extension, tags, content_preview, search_text, now, now),
            )

    def delete_file(self, path: str) -> bool:
        with self._sync_connect() as db:
            cur = db.execute(
                "DELETE FROM kartoteka.file_index WHERE path = %s RETURNING id",
                (path,),
            )
            return cur.fetchone() is not None

    def move_file(self, old_path: str, new_path: str, new_filename: str) -> bool:
        now = datetime.now(UTC).isoformat()
        with self._sync_connect() as db:
            cur = db.execute(
                """
                UPDATE kartoteka.file_index
                SET path = %s, filename = %s, updated_at = %s
                WHERE path = %s
                RETURNING id
                """,
                (new_path, new_filename, now, old_path),
            )
            return cur.fetchone() is not None
