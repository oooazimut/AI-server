from __future__ import annotations

from datetime import UTC, datetime

import psycopg
import psycopg.sql as sql
from psycopg.rows import dict_row


class PostgresAgentSchema:
    """Base class for per-agent PostgreSQL schemas.

    Subclasses set ``_SCHEMA`` (e.g. "bitrix24") and call ``super().ensure_schema()``
    before creating their own tables. Provides async dialog_history methods so that
    BaseSpecialist.handle() never blocks the event loop.

    Synchronous compatibility methods in subclasses use ``_sync_connect()``
    (existing psycopg sync API — acceptable for infrequent writes).
    """

    _SCHEMA: str  # set in each concrete subclass

    def __init__(self, url: str) -> None:
        self._url = url

    async def _connect(self) -> psycopg.AsyncConnection:
        return await psycopg.AsyncConnection.connect(self._url, row_factory=dict_row)

    def _sync_connect(self) -> psycopg.Connection:
        return psycopg.connect(self._url, row_factory=dict_row)

    async def ensure_schema(self) -> None:
        """Create the agent schema and dialog_history table. Subclasses call super() first."""
        schema = self._SCHEMA
        async with await self._connect() as db:
            await db.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
            await db.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {schema}.dialog_history (
                    id BIGSERIAL PRIMARY KEY,
                    dialog_key TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            await db.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{schema}_dh_key
                ON {schema}.dialog_history (dialog_key, id)
                """
            )

    async def load_turns(self, dialog_key: str, *, limit: int = 20) -> list[dict[str, str]]:
        schema = self._SCHEMA
        async with await self._connect() as db:
            cur = await db.execute(
                f"""
                SELECT role, content FROM (
                    SELECT id, role, content
                    FROM {schema}.dialog_history
                    WHERE dialog_key = %s
                    ORDER BY id DESC
                    LIMIT %s
                ) sub
                ORDER BY id ASC
                """,
                (dialog_key, limit),
            )
            rows = await cur.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    async def append_turn(self, dialog_key: str, user_text: str, agent_response: str) -> None:
        schema = self._SCHEMA
        now = datetime.now(UTC).isoformat()
        async with await self._connect() as db:
            for role, content in (("user", user_text), ("assistant", agent_response)):
                await db.execute(
                    f"""
                    INSERT INTO {schema}.dialog_history (dialog_key, role, content, created_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (dialog_key, role, content, now),
                )
            # Keep last 40 rows per dialog_key (= 20 exchanges)
            await db.execute(
                f"""
                DELETE FROM {schema}.dialog_history
                WHERE dialog_key = %s AND id NOT IN (
                    SELECT id FROM {schema}.dialog_history
                    WHERE dialog_key = %s
                    ORDER BY id DESC
                    LIMIT 40
                )
                """,
                (dialog_key, dialog_key),
            )
