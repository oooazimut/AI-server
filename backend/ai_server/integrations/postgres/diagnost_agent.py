from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ai_server.models import AgentResult, AgentTask

from .agent_schema import PostgresAgentSchema

logger = logging.getLogger(__name__)


class PostgresDiagnostStore(PostgresAgentSchema):
    """Diagnost specialist store: dialog_history + events + incidents in the 'diagnost' schema."""

    _SCHEMA = "diagnost"

    async def ensure_schema(self) -> None:
        await super().ensure_schema()
        async with await self._connect() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnost.events (
                    id BIGSERIAL PRIMARY KEY,
                    event_id TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    agent_id TEXT NOT NULL DEFAULT '',
                    user_id TEXT,
                    channel TEXT,
                    request TEXT NOT NULL DEFAULT '',
                    response TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT '',
                    confidence FLOAT,
                    handoff_to TEXT[],
                    actions JSONB,
                    model_usage JSONB,
                    metadata JSONB
                )
                """
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_diagnost_events_status ON diagnost.events (status)")
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_diagnost_events_created ON diagnost.events (created_at DESC)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_diagnost_events_confidence ON diagnost.events (confidence)"
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnost.incidents (
                    id BIGSERIAL PRIMARY KEY,
                    incident_id TEXT UNIQUE NOT NULL DEFAULT gen_random_uuid()::TEXT,
                    event_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    reason TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'open',
                    comment TEXT NOT NULL DEFAULT '',
                    metadata JSONB
                )
                """
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_diagnost_incidents_status ON diagnost.incidents (status)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_diagnost_incidents_event ON diagnost.incidents (event_id)")

    async def save_event(self, task: AgentTask, result: AgentResult) -> None:
        user_id = str(task.user.id) if task.user and task.user.id is not None else None
        channel = task.user.channel if task.user else None
        async with await self._connect() as db:
            await db.execute(
                """
                INSERT INTO diagnost.events
                    (event_id, agent_id, user_id, channel, request, response,
                     status, confidence, handoff_to, actions, model_usage, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING
                """,
                (
                    task.task_id,
                    result.agent_id or "",
                    user_id,
                    channel,
                    task.request or "",
                    result.answer or "",
                    result.status or "",
                    result.confidence,
                    result.handoff_to or [],
                    _jsonb([a.model_dump() for a in result.actions_taken]),
                    _jsonb([u.model_dump() for u in result.model_usage]),
                    _jsonb({"dialog_key": task.context.get("dialog_key", "")}),
                ),
            )

    async def save_incident(self, event_id: str, *, reason: str, comment: str = "") -> str:
        incident_id = uuid4().hex
        async with await self._connect() as db:
            await db.execute(
                """
                INSERT INTO diagnost.incidents (incident_id, event_id, reason, comment)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (incident_id) DO NOTHING
                """,
                (incident_id, event_id, reason, comment),
            )
        return incident_id

    async def search_events(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        q = f"%{query}%"
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT event_id, created_at, agent_id, user_id, channel,
                       request, response, status, confidence, handoff_to
                FROM diagnost.events
                WHERE request ILIKE %s OR response ILIKE %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (q, q, limit),
            )
            rows = await cur.fetchall()
        return [_row(r) for r in rows]

    async def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT i.incident_id, i.event_id, i.created_at, i.reason, i.status, i.comment,
                       e.request, e.response, e.agent_id, e.confidence
                FROM diagnost.incidents i
                LEFT JOIN diagnost.events e ON e.event_id = i.event_id
                WHERE i.incident_id = %s
                """,
                (incident_id,),
            )
            row = await cur.fetchone()
        return _row(row) if row else None

    async def list_incidents(self, *, status: str = "", limit: int = 50) -> list[dict[str, Any]]:
        async with await self._connect() as db:
            if status:
                cur = await db.execute(
                    """
                    SELECT incident_id, event_id, created_at, reason, status, comment
                    FROM diagnost.incidents
                    WHERE status = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (status, limit),
                )
            else:
                cur = await db.execute(
                    """
                    SELECT incident_id, event_id, created_at, reason, status, comment
                    FROM diagnost.incidents
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
            rows = await cur.fetchall()
        return [_row(r) for r in rows]

    async def error_report(self, *, since_hours: int = 24) -> dict[str, Any]:
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT reason, COUNT(*) AS cnt
                FROM diagnost.incidents
                WHERE created_at >= NOW() - INTERVAL '%s hours'
                GROUP BY reason
                ORDER BY cnt DESC
                """,
                (since_hours,),
            )
            by_reason = await cur.fetchall()

            cur2 = await db.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_count
                FROM diagnost.incidents
                WHERE created_at >= NOW() - INTERVAL '%s hours'
                """,
                (since_hours,),
            )
            totals = await cur2.fetchone()

            cur3 = await db.execute(
                """
                SELECT i.incident_id, i.event_id, i.reason, i.created_at, e.request, e.confidence
                FROM diagnost.incidents i
                LEFT JOIN diagnost.events e ON e.event_id = i.event_id
                WHERE i.created_at >= NOW() - INTERVAL '%s hours'
                ORDER BY i.created_at DESC
                LIMIT 10
                """,
                (since_hours,),
            )
            recent = await cur3.fetchall()

        return {
            "since_hours": since_hours,
            "generated_at": datetime.now(UTC).isoformat(),
            "total_incidents": int((totals or {}).get("total") or 0),
            "open_incidents": int((totals or {}).get("open_count") or 0),
            "by_reason": [{"reason": r["reason"], "count": int(r["cnt"])} for r in by_reason],
            "recent_incidents": [_row(r) for r in recent],
        }


def _jsonb(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def _row(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if hasattr(row, "keys"):
        return dict(row)
    return row
