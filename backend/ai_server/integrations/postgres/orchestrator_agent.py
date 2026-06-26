from __future__ import annotations

from datetime import UTC, datetime

from .agent_schema import PostgresAgentSchema

_KV_FIELD_MAX = 64
_KV_VALUE_MAX = 256


class PostgresOrchestratorStore(PostgresAgentSchema):
    """История диалогов и KV-состояние оркестратора (схема internal_orchestrator)."""

    _SCHEMA = "internal_orchestrator"

    async def ensure_schema(self) -> None:
        await super().ensure_schema()
        async with await self._connect() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS internal_orchestrator.dialog_kv (
                    dialog_key TEXT NOT NULL,
                    field      TEXT NOT NULL,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (dialog_key, field)
                )
                """
            )

    async def get_kv(self, dialog_key: str, field: str) -> str | None:
        async with await self._connect() as db:
            cur = await db.execute(
                "SELECT value FROM internal_orchestrator.dialog_kv WHERE dialog_key=%s AND field=%s",
                (dialog_key, field),
            )
            row = await cur.fetchone()
        return str(row["value"]) if row else None

    async def set_kv(self, dialog_key: str, field: str, value: str) -> None:
        now = datetime.now(UTC).isoformat()
        async with await self._connect() as db:
            await db.execute(
                """
                INSERT INTO internal_orchestrator.dialog_kv (dialog_key, field, value, updated_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (dialog_key, field)
                DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                """,
                (dialog_key, field, value[:_KV_VALUE_MAX], now),
            )

    async def delete_kv(self, dialog_key: str, field: str) -> None:
        async with await self._connect() as db:
            await db.execute(
                "DELETE FROM internal_orchestrator.dialog_kv WHERE dialog_key=%s AND field=%s",
                (dialog_key, field),
            )
