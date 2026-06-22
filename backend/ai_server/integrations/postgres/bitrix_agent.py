from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from .agent_schema import PostgresAgentSchema


class PostgresBitrixAgentStore(PostgresAgentSchema):
    """Bitrix24 agent store: dialog_history + incomplete_proposals in the 'bitrix24' schema.

    Async methods (ensure_schema, load_turns, append_turn) satisfy AgentDialogStorePort.
    Sync proposal methods satisfy BitrixAgentStorePort via structural typing.
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
