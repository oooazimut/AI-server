from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_server.agent_store import AgentStore


class BitrixAgentStore(AgentStore):
    def __init__(self, path: Path | None = None) -> None:
        super().__init__("bitrix24", path=path)

    def ensure_schema(self) -> None:
        super().ensure_schema()
        with self._connection() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS incomplete_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
        self.ensure_schema()
        with self._connection() as db:
            cur = db.execute(
                """
                INSERT INTO incomplete_proposals
                    (task_id, task_title, missing_parts, responsible_id, responsible_dialog_id,
                     status, created_at, scheduled_for)
                VALUES (?, ?, ?, ?, ?, 'awaiting_response', ?, ?)
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
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_proposal_by_id(self, proposal_id: int) -> dict[str, Any] | None:
        self.ensure_schema()
        with self._connection() as db:
            row = db.execute("SELECT * FROM incomplete_proposals WHERE id = ?", (proposal_id,)).fetchone()
        return dict(row) if row else None

    def get_proposals_for_manager(self) -> list[dict[str, Any]]:
        self.ensure_schema()
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM incomplete_proposals WHERE status IN ('awaiting_response', 'proposed') ORDER BY created_at"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_pending_for_responsible(self, responsible_id: int) -> dict[str, Any] | None:
        self.ensure_schema()
        with self._connection() as db:
            row = db.execute(
                """
                SELECT * FROM incomplete_proposals
                WHERE responsible_id = ? AND status = 'awaiting_response'
                ORDER BY created_at LIMIT 1
                """,
                (responsible_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_responsible_response(self, proposal_id: int, response_text: str) -> None:
        self.ensure_schema()
        with self._connection() as db:
            db.execute(
                "UPDATE incomplete_proposals SET responsible_response = ? WHERE id = ?",
                (response_text, proposal_id),
            )

    def mark_status(self, proposal_id: int, status: str) -> None:
        self.ensure_schema()
        with self._connection() as db:
            db.execute(
                "UPDATE incomplete_proposals SET status = ? WHERE id = ?",
                (status, proposal_id),
            )

    def delete_proposal(self, proposal_id: int) -> None:
        self.ensure_schema()
        with self._connection() as db:
            db.execute("DELETE FROM incomplete_proposals WHERE id = ?", (proposal_id,))
