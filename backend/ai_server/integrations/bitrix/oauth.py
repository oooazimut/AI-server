from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from typing import Any

import httpx

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixClient, BitrixConfigError
from ai_server.settings import get_settings


class BitrixOAuthError(RuntimeError):
    pass


class BitrixOAuthTokenMissing(BitrixOAuthError):
    def __init__(self, user_id: int) -> None:
        self.user_id = user_id
        super().__init__(f"OAuth token for Bitrix user #{user_id} is not linked")


@dataclass(frozen=True)
class BitrixOAuthToken:
    user_id: int
    access_token: str
    refresh_token: str
    client_endpoint: str
    server_endpoint: str
    domain: str
    member_id: str
    scope: str
    expires_at: datetime
    updated_at: datetime

    @property
    def expires_soon(self) -> bool:
        return self.expires_at <= datetime.now(timezone.utc) + timedelta(minutes=5)


class BitrixOAuthService:
    def __init__(self, db_path: Path | str | None = None) -> None:
        settings = get_settings()
        self.db_path = Path(db_path or settings.bitrix_oauth_db_path)

    def ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix_oauth_tokens (
                    user_id INTEGER PRIMARY KEY,
                    domain TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    client_endpoint TEXT NOT NULL,
                    server_endpoint TEXT NOT NULL,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    scope TEXT NOT NULL DEFAULT '',
                    expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS bitrix_oauth_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    event_type TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT ''
                )
                """
            )

    def get_token(self, user_id: int) -> BitrixOAuthToken | None:
        self.ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, domain, member_id, client_endpoint, server_endpoint,
                       access_token, refresh_token, scope, expires_at, updated_at
                FROM bitrix_oauth_tokens
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return _row_to_token(row) if row else None

    async def client_for_user(self, user_id: int) -> BitrixClient:
        token = self.get_token(user_id)
        if token is None:
            raise BitrixOAuthTokenMissing(user_id)
        if token.expires_soon:
            token = await self.refresh_token(token)
        return BitrixClient(access_token=token.access_token, client_endpoint=token.client_endpoint)

    async def refresh_token(self, token: BitrixOAuthToken) -> BitrixOAuthToken:
        settings = get_settings()
        if not settings.bitrix_oauth_client_id or not settings.bitrix_oauth_client_secret:
            raise BitrixConfigError("OAuth token expired, but client credentials are not configured")
        payload = await self._request_token(
            {
                "grant_type": "refresh_token",
                "client_id": settings.bitrix_oauth_client_id,
                "client_secret": settings.bitrix_oauth_client_secret,
                "refresh_token": token.refresh_token,
            },
            endpoint=_token_endpoint_from_server(token.server_endpoint),
        )
        refreshed = BitrixOAuthToken(
            user_id=token.user_id,
            access_token=str(payload.get("access_token") or token.access_token),
            refresh_token=str(payload.get("refresh_token") or token.refresh_token),
            client_endpoint=str(payload.get("client_endpoint") or token.client_endpoint),
            server_endpoint=str(payload.get("server_endpoint") or token.server_endpoint),
            domain=str(payload.get("domain") or token.domain),
            member_id=str(payload.get("member_id") or token.member_id),
            scope=str(payload.get("scope") or token.scope),
            expires_at=_expires_at(payload.get("expires"), payload.get("expires_in")),
            updated_at=datetime.now(timezone.utc),
        )
        self.save_token(refreshed, source="refresh")
        return refreshed

    def save_token(self, token: BitrixOAuthToken, *, source: str) -> None:
        self.ensure_schema()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO bitrix_oauth_tokens (
                    user_id, domain, member_id, client_endpoint, server_endpoint,
                    access_token, refresh_token, scope, expires_at, updated_at, source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    domain = excluded.domain,
                    member_id = excluded.member_id,
                    client_endpoint = excluded.client_endpoint,
                    server_endpoint = excluded.server_endpoint,
                    access_token = excluded.access_token,
                    refresh_token = excluded.refresh_token,
                    scope = excluded.scope,
                    expires_at = excluded.expires_at,
                    updated_at = excluded.updated_at,
                    source = excluded.source
                """,
                (
                    token.user_id,
                    token.domain,
                    token.member_id,
                    token.client_endpoint,
                    token.server_endpoint,
                    token.access_token,
                    token.refresh_token,
                    token.scope,
                    token.expires_at.isoformat(),
                    token.updated_at.isoformat(),
                    source,
                ),
            )

    def public_status(self) -> dict[str, Any]:
        settings = get_settings()
        self.ensure_schema()
        with self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM bitrix_oauth_tokens").fetchone()[0])
            rows = connection.execute(
                """
                SELECT user_id, domain, scope, expires_at, updated_at
                FROM bitrix_oauth_tokens
                ORDER BY updated_at DESC
                LIMIT 20
                """
            ).fetchall()
        return {
            "enabled": settings.bitrix_oauth_enabled,
            "configured": settings.bitrix_oauth_configured,
            "required_for_writes": settings.bitrix_oauth_required_for_writes,
            "db_path": str(self.db_path),
            "linked_users_count": count,
            "linked_users": [
                {
                    "user_id": int(row["user_id"]),
                    "domain": row["domain"],
                    "scope": row["scope"],
                    "expires_at": row["expires_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ],
        }

    async def _request_token(self, data: dict[str, str], *, endpoint: str | None = None) -> dict[str, Any]:
        settings = get_settings()
        url = endpoint or settings.bitrix_oauth_token_endpoint
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), trust_env=False) as client:
            response = await client.post(url, data=data)
            response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise BitrixApiError(
                "oauth.token",
                str(payload.get("error", "")),
                str(payload.get("error_description", "")),
            )
        return payload

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection


def _row_to_token(row: sqlite3.Row) -> BitrixOAuthToken:
    return BitrixOAuthToken(
        user_id=int(row["user_id"]),
        access_token=str(row["access_token"]),
        refresh_token=str(row["refresh_token"]),
        client_endpoint=str(row["client_endpoint"]),
        server_endpoint=str(row["server_endpoint"]),
        domain=str(row["domain"]),
        member_id=str(row["member_id"]),
        scope=str(row["scope"] or ""),
        expires_at=_parse_datetime(str(row["expires_at"])),
        updated_at=_parse_datetime(str(row["updated_at"])),
    )


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _expires_at(expires: Any, expires_in: Any) -> datetime:
    now = datetime.now(timezone.utc)
    if expires:
        try:
            return datetime.fromtimestamp(int(expires), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            pass
    try:
        return now + timedelta(seconds=int(expires_in))
    except (TypeError, ValueError):
        return now + timedelta(hours=1)


def _token_endpoint_from_server(server_endpoint: str) -> str:
    endpoint = server_endpoint.strip()
    if endpoint.endswith("/oauth/token/") or endpoint.endswith("/oauth/token"):
        return endpoint
    return endpoint.rstrip("/") + "/oauth/token/"

