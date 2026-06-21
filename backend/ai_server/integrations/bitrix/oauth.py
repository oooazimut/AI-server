from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from ai_server.agent_store import SqliteStore
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixClient, BitrixConfigError
from ai_server.settings import Settings, get_settings
from ai_server.utils import optional_int


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
        return self.expires_at <= datetime.now(UTC) + timedelta(minutes=5)


@dataclass(frozen=True)
class BitrixOAuthSaveResult:
    user_id: int
    domain: str
    member_id: str
    scope: str
    expires_at: datetime
    source: str


class BitrixOAuthService(SqliteStore):
    def __init__(self, settings: Settings | None = None, db_path: Path | str | None = None) -> None:
        self._settings = settings or get_settings()
        self.path = Path(db_path or self._settings.bitrix_oauth_db_path)

    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
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

    async def save_from_payload(
        self,
        payload: dict[str, Any],
        *,
        source: str,
    ) -> BitrixOAuthSaveResult:
        auth = _extract_auth(payload, self._settings)
        if not auth.access_token:
            raise BitrixOAuthError("OAuth access token is missing in Bitrix payload")
        if not auth.refresh_token:
            raise BitrixOAuthError("OAuth refresh token is missing in Bitrix payload")

        user_id = await self._resolve_user_id(auth)
        token = BitrixOAuthToken(
            user_id=user_id,
            access_token=auth.access_token,
            refresh_token=auth.refresh_token,
            client_endpoint=auth.client_endpoint,
            server_endpoint=auth.server_endpoint,
            domain=auth.domain,
            member_id=auth.member_id,
            scope=auth.scope,
            expires_at=auth.expires_at,
            updated_at=datetime.now(UTC),
        )
        self.save_token(token, source=source)
        return BitrixOAuthSaveResult(
            user_id=user_id,
            domain=token.domain,
            member_id=token.member_id,
            scope=token.scope,
            expires_at=token.expires_at,
            source=source,
        )

    async def exchange_authorization_code(
        self,
        *,
        code: str,
        source: str,
    ) -> BitrixOAuthSaveResult:
        settings = self._settings
        if not settings.bitrix_oauth_client_id or not settings.bitrix_oauth_client_secret:
            raise BitrixConfigError("BITRIX_OAUTH_CLIENT_ID and BITRIX_OAUTH_CLIENT_SECRET are required")
        payload = await self._request_token(
            {
                "grant_type": "authorization_code",
                "client_id": settings.bitrix_oauth_client_id,
                "client_secret": settings.bitrix_oauth_client_secret,
                "code": code,
            }
        )
        return await self.save_from_payload({"auth": payload}, source=source)

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
        return BitrixClient(
            settings=self._settings, access_token=token.access_token, client_endpoint=token.client_endpoint
        )

    async def refresh_token(self, token: BitrixOAuthToken) -> BitrixOAuthToken:
        settings = self._settings
        if not settings.bitrix_oauth_client_id or not settings.bitrix_oauth_client_secret:
            raise BitrixConfigError("OAuth token expired, but client credentials are not configured")
        payload = await self._request_token(
            {
                "grant_type": "refresh_token",
                "client_id": settings.bitrix_oauth_client_id,
                "client_secret": settings.bitrix_oauth_client_secret,
                "refresh_token": token.refresh_token,
            },
            endpoint=_token_endpoint_from_server(token.server_endpoint, settings),
        )
        normalized = _normalize_auth(payload, fallback=token, settings=settings)
        refreshed = BitrixOAuthToken(
            user_id=token.user_id,
            access_token=normalized.access_token,
            refresh_token=normalized.refresh_token,
            client_endpoint=normalized.client_endpoint,
            server_endpoint=normalized.server_endpoint,
            domain=normalized.domain,
            member_id=normalized.member_id,
            scope=normalized.scope,
            expires_at=normalized.expires_at,
            updated_at=datetime.now(UTC),
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
            connection.execute(
                """
                INSERT INTO bitrix_oauth_events (user_id, event_type, source, created_at, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    token.user_id,
                    "token_saved",
                    source,
                    datetime.now(UTC).isoformat(),
                    f"domain={token.domain}; scope={token.scope}",
                ),
            )

    def public_status(self) -> dict[str, Any]:
        settings = self._settings
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
            "db_path": str(self.path),
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
            "authorization": self.authorization_hint(),
        }

    def authorization_hint(self, user_id: int | None = None) -> dict[str, Any]:
        settings = self._settings
        return {
            "user_id": user_id,
            "app_url": settings.resolved_bitrix_app_url,
            "marketplace_app_url": settings.resolved_bitrix_marketplace_app_url,
            "marketplace_app_path": settings.resolved_bitrix_marketplace_app_path,
            "oauth_start_url": settings.resolved_bitrix_oauth_start_url,
            "message": (
                "Откройте локальное приложение AI-помощника в Bitrix24 один раз, "
                "чтобы помощник получил OAuth-доступ от вашего имени."
            ),
        }

    async def _resolve_user_id(self, auth: _NormalizedAuth) -> int:
        if auth.user_id:
            return auth.user_id
        client = BitrixClient(
            settings=self._settings, access_token=auth.access_token, client_endpoint=auth.client_endpoint
        )
        result = await client.result("user.current", {})
        if not isinstance(result, dict):
            raise BitrixOAuthError("user.current did not return user data")
        raw_id = result.get("ID") or result.get("id")
        if not str(raw_id or "").isdigit():
            raise BitrixOAuthError("Could not resolve Bitrix user id from OAuth token")
        return int(raw_id)

    async def _request_token(self, data: dict[str, str], *, endpoint: str | None = None) -> dict[str, Any]:
        url = endpoint or self._settings.bitrix_oauth_token_endpoint
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0), trust_env=False) as client:
            response = await client.post(url, data=data)
        payload = _response_json(response)
        if "error" in payload:
            raise BitrixApiError(
                "oauth.token",
                str(payload.get("error", "")),
                str(payload.get("error_description", "")),
            )
        if response.is_error:
            raise BitrixApiError(
                "oauth.token",
                f"HTTP_{response.status_code}",
                _response_text(response),
            ) from None
        return payload


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


@dataclass(frozen=True)
class _NormalizedAuth:
    access_token: str
    refresh_token: str
    client_endpoint: str
    server_endpoint: str
    domain: str
    member_id: str
    scope: str
    expires_at: datetime
    user_id: int | None = None


def _extract_auth(payload: dict[str, Any], settings: Settings) -> _NormalizedAuth:
    auth = payload.get("auth") if isinstance(payload.get("auth"), dict) else {}
    values = {**payload, **auth}
    if "AUTH_ID" in payload:
        values["access_token"] = payload.get("AUTH_ID")
    if "REFRESH_ID" in payload:
        values["refresh_token"] = payload.get("REFRESH_ID")
    if "AUTH_EXPIRES" in payload:
        values["expires_in"] = payload.get("AUTH_EXPIRES")
    if "DOMAIN" in payload and "domain" not in values:
        values["domain"] = payload.get("DOMAIN")
    return _normalize_auth(values, settings=settings)


def _normalize_auth(
    value: dict[str, Any], *, fallback: BitrixOAuthToken | None = None, settings: Settings
) -> _NormalizedAuth:
    access_token = str(value.get("access_token") or (fallback.access_token if fallback else "") or "")
    refresh_token = str(value.get("refresh_token") or (fallback.refresh_token if fallback else "") or "")
    domain = _domain(str(value.get("domain") or (fallback.domain if fallback else "") or settings.bitrix_domain))
    client_endpoint = str(value.get("client_endpoint") or (fallback.client_endpoint if fallback else "") or "")
    if not client_endpoint and domain:
        client_endpoint = f"https://{domain}/rest/"
    server_endpoint = str(
        value.get("server_endpoint")
        or (fallback.server_endpoint if fallback else "")
        or "https://oauth.bitrix.info/rest/"
    )
    member_id = str(value.get("member_id") or (fallback.member_id if fallback else "") or "")
    scope = str(value.get("scope") or (fallback.scope if fallback else "") or "")
    user_id = optional_int(value.get("user_id") or value.get("USER_ID") or value.get("member_user_id"))

    expires_in = optional_int(value.get("expires_in") or value.get("expires") or value.get("AUTH_EXPIRES"))
    if expires_in is None and fallback is not None:
        expires_at = fallback.expires_at
    else:
        expires_at = datetime.now(UTC) + timedelta(seconds=max(60, expires_in or 3600))

    return _NormalizedAuth(
        access_token=access_token,
        refresh_token=refresh_token,
        client_endpoint=client_endpoint.rstrip("/") + "/" if client_endpoint else "",
        server_endpoint=server_endpoint,
        domain=domain,
        member_id=member_id,
        scope=scope,
        expires_at=expires_at,
        user_id=user_id,
    )


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _expires_at(expires: Any, expires_in: Any) -> datetime:
    now = datetime.now(UTC)
    if expires:
        try:
            return datetime.fromtimestamp(int(expires), tz=UTC)
        except (TypeError, ValueError, OSError):
            pass
    try:
        return now + timedelta(seconds=int(expires_in))
    except (TypeError, ValueError):
        return now + timedelta(hours=1)


def _token_endpoint_from_server(server_endpoint: str, settings: Settings) -> str:
    if settings.bitrix_oauth_token_endpoint:
        return settings.bitrix_oauth_token_endpoint
    endpoint = server_endpoint.strip()
    if endpoint.endswith("/oauth/token/") or endpoint.endswith("/oauth/token"):
        return endpoint
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/rest"):
        endpoint = endpoint[:-5]
    return endpoint.rstrip("/") + "/oauth/token/"


def _domain(value: str) -> str:
    return value.strip().removeprefix("https://").removeprefix("http://").rstrip("/")


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _response_text(response: httpx.Response) -> str:
    text = response.text.strip()
    if len(text) > 500:
        return text[:500] + "..."
    return text
