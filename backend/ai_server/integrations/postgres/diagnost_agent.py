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
                    metadata JSONB,
                    source TEXT NOT NULL DEFAULT 'orchestrator',
                    trace_snapshot JSONB,
                    trace_captured_at TIMESTAMPTZ
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
            await db.execute("CREATE INDEX IF NOT EXISTS idx_diagnost_events_source ON diagnost.events (source)")
            # ALTER for existing tables that predate the source column
            await db.execute(
                "ALTER TABLE diagnost.events ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'orchestrator'"
            )
            await db.execute("ALTER TABLE diagnost.events ADD COLUMN IF NOT EXISTS trace_snapshot JSONB")
            await db.execute("ALTER TABLE diagnost.events ADD COLUMN IF NOT EXISTS trace_captured_at TIMESTAMPTZ")
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
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnost.pending_feedback (
                    id BIGSERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    dialog_key TEXT NOT NULL,
                    channel TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    prompt_sent_at TIMESTAMPTZ,
                    status TEXT NOT NULL DEFAULT 'pending'
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_fb_user ON diagnost.pending_feedback (user_id, status)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_fb_status ON diagnost.pending_feedback (status, created_at)"
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnost.feedback (
                    id BIGSERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    dialog_key TEXT,
                    rating SMALLINT,
                    raw_text TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_event ON diagnost.feedback (event_id)")

    async def save_trace_snapshot(self, event_id: str, trace: list[dict[str, Any]]) -> None:
        """Persist the single canonical long-lived copy of a task's sanitized trace."""
        async with await self._connect() as db:
            await db.execute(
                """
                UPDATE diagnost.events
                SET trace_snapshot = %s, trace_captured_at = NOW()
                WHERE event_id = %s
                """,
                (_jsonb(trace), event_id),
            )

    async def cancel_pending_feedback(self) -> int:
        """Prevent old unsent prompts from resurfacing while feedback is disabled."""
        async with await self._connect() as db:
            cur = await db.execute("UPDATE diagnost.pending_feedback SET status = 'cancelled' WHERE status = 'pending'")
            return int(cur.rowcount or 0)

    async def save_event(self, task: AgentTask, result: AgentResult, *, source: str = "orchestrator") -> None:
        user_id = str(task.user.id) if task.user and task.user.id is not None else None
        channel = task.user.channel if task.user else None
        async with await self._connect() as db:
            await db.execute(
                """
                INSERT INTO diagnost.events
                    (event_id, agent_id, user_id, channel, request, response,
                     status, confidence, handoff_to, actions, model_usage, metadata, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                    _jsonb({"dialog_key": task.context.get("dialog_key", ""), **(result.metadata or {})}),
                    source,
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
                       request, response, status, confidence, handoff_to, source
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

            cur4 = await db.execute(
                """
                SELECT e.agent_id, COUNT(*) AS cnt
                FROM diagnost.incidents i
                LEFT JOIN diagnost.events e ON e.event_id = i.event_id
                WHERE i.created_at >= NOW() - INTERVAL '%s hours'
                GROUP BY e.agent_id
                ORDER BY cnt DESC
                """,
                (since_hours,),
            )
            by_specialist = await cur4.fetchall()

            cur5 = await db.execute(
                """
                SELECT AVG(rating) AS avg_rating, COUNT(*) AS feedback_count
                FROM diagnost.feedback
                WHERE created_at >= NOW() - INTERVAL '%s hours'
                """,
                (since_hours,),
            )
            fb_totals = await cur5.fetchone()

        return {
            "since_hours": since_hours,
            "generated_at": datetime.now(UTC).isoformat(),
            "total_incidents": int((totals or {}).get("total") or 0),
            "open_incidents": int((totals or {}).get("open_count") or 0),
            "by_reason": [{"reason": r["reason"], "count": int(r["cnt"])} for r in by_reason],
            "by_specialist": [{"agent_id": r["agent_id"] or "unknown", "count": int(r["cnt"])} for r in by_specialist],
            "avg_rating": float((fb_totals or {}).get("avg_rating") or 0) or None,
            "feedback_count": int((fb_totals or {}).get("feedback_count") or 0),
            "recent_incidents": [_row(r) for r in recent],
        }

    # ------------------------------------------------------------------
    # FeedbackLoop methods
    # ------------------------------------------------------------------

    async def create_pending_feedback(
        self, event_id: str, user_id: str, dialog_key: str, *, channel: str | None = None
    ) -> None:
        async with await self._connect() as db:
            await db.execute(
                """
                INSERT INTO diagnost.pending_feedback (event_id, user_id, dialog_key, channel)
                VALUES (%s, %s, %s, %s)
                """,
                (event_id, user_id, dialog_key, channel),
            )

    async def get_pending_feedback_for_user(self, user_id: str) -> dict[str, Any] | None:
        """Return the most recent 'sent' pending_feedback entry for the user, or None."""
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT id, event_id, user_id, dialog_key, channel, created_at
                FROM diagnost.pending_feedback
                WHERE user_id = %s AND status = 'sent'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (user_id,),
            )
            row = await cur.fetchone()
        return _row(row) if row else None

    async def get_unsent_feedback_requests(self, *, delay_seconds: int = 10, limit: int = 20) -> list[dict[str, Any]]:
        """Return pending_feedback rows whose dialog turn is old enough to send a prompt."""
        async with await self._connect() as db:
            cur = await db.execute(
                """
                SELECT id, event_id, user_id, dialog_key, channel, created_at
                FROM diagnost.pending_feedback
                WHERE status = 'pending'
                  AND created_at <= NOW() - INTERVAL '%s seconds'
                ORDER BY created_at ASC
                LIMIT %s
                """,
                (delay_seconds, limit),
            )
            rows = await cur.fetchall()
        return [_row(r) for r in rows]

    async def save_feedback(
        self,
        event_id: str,
        user_id: str,
        *,
        rating: int | None,
        raw_text: str,
        dialog_key: str = "",
    ) -> None:
        async with await self._connect() as db:
            await db.execute(
                """
                INSERT INTO diagnost.feedback (event_id, user_id, dialog_key, rating, raw_text)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (event_id, user_id, dialog_key or None, rating, raw_text),
            )

    async def mark_pending_sent(self, pending_id: int) -> None:
        async with await self._connect() as db:
            await db.execute(
                "UPDATE diagnost.pending_feedback SET status = 'sent', prompt_sent_at = NOW() WHERE id = %s",
                (pending_id,),
            )

    async def mark_pending_received(self, pending_id: int) -> None:
        async with await self._connect() as db:
            await db.execute(
                "UPDATE diagnost.pending_feedback SET status = 'received' WHERE id = %s",
                (pending_id,),
            )


def _jsonb(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, default=str)


def _row(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if hasattr(row, "keys"):
        return dict(row)
    return row
