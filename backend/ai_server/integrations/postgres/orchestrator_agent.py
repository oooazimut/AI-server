from __future__ import annotations

from datetime import UTC, datetime, timedelta

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
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS internal_orchestrator.replacement_candidates (
                    dialog_key TEXT PRIMARY KEY,
                    request_text TEXT NOT NULL,
                    draft_id TEXT NOT NULL,
                    draft_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
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

    async def save_replacement_candidate(
        self, dialog_key: str, *, request_text: str, draft_id: str, draft_type: str, ttl_minutes: int = 15
    ) -> dict[str, str]:
        now = datetime.now(UTC)
        expires_at = now + timedelta(minutes=ttl_minutes)
        async with await self._connect() as db:
            await db.execute(
                "DELETE FROM internal_orchestrator.replacement_candidates WHERE expires_at <= %s", (now.isoformat(),)
            )
            cur = await db.execute(
                "SELECT request_text, draft_id, draft_type, created_at, expires_at "
                "FROM internal_orchestrator.replacement_candidates WHERE dialog_key=%s",
                (dialog_key,),
            )
            current = await cur.fetchone()
            if current:
                return {key: str(current[key]) for key in current}
            await db.execute(
                """
                INSERT INTO internal_orchestrator.replacement_candidates
                    (dialog_key, request_text, draft_id, draft_type, created_at, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (dialog_key, request_text, draft_id, draft_type, now.isoformat(), expires_at.isoformat()),
            )
        return {
            "request_text": request_text,
            "draft_id": draft_id,
            "draft_type": draft_type,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
        }

    async def get_replacement_candidate(self, dialog_key: str) -> dict[str, str] | None:
        now = datetime.now(UTC).isoformat()
        async with await self._connect() as db:
            await db.execute(
                "DELETE FROM internal_orchestrator.replacement_candidates WHERE dialog_key=%s AND expires_at <= %s",
                (dialog_key, now),
            )
            cur = await db.execute(
                "SELECT request_text, draft_id, draft_type, created_at, expires_at "
                "FROM internal_orchestrator.replacement_candidates WHERE dialog_key=%s",
                (dialog_key,),
            )
            row = await cur.fetchone()
        return {key: str(row[key]) for key in row} if row else None

    async def delete_replacement_candidate(self, dialog_key: str) -> None:
        async with await self._connect() as db:
            await db.execute(
                "DELETE FROM internal_orchestrator.replacement_candidates WHERE dialog_key=%s", (dialog_key,)
            )
