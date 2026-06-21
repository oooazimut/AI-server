from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from ai_server.settings import get_settings


class SqliteStore:
    """Minimal SQLite connection mixin. Subclasses must set self.path: Path."""

    path: Path

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        """Return a raw connection for callers that manage the lifecycle themselves."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


class AgentStore(SqliteStore):
    """Base SQLite store for specialist agents. Provides connection management and schema helpers."""

    def __init__(self, agent_id: str, path: Path | None = None) -> None:
        self.agent_id = agent_id
        self.path = path or (get_settings().var_dir / agent_id / f"{agent_id}.sqlite")

    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ensure_column(db: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        columns = {str(row["name"]) for row in db.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
