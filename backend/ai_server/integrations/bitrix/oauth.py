from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import redis.asyncio as aioredis

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixClient, BitrixConfigError
from ai_server.settings import Settings, get_settings
from ai_server.utils import optional_int

_KEY_PREFIX = "oauth:bitrix:"


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


class BitrixOAuthService:
    def __init__(self, settings: Settings | None = None, redis_url: str | None = None) -> None:
        self._settings = settings or get_settings()
        self._redis = aioredis.from_url(
            redis_url or self._settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
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
        await self.save_token(token, source=source)
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

    async def get_token(self, user_id: int) -> BitrixOAuthToken | None:
        raw = await self._redis.get(f"{_KEY_PREFIX}{user_id}")
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return _dict_to_token(data)

    async def client_for_user(self, user_id: int) -> BitrixClient:
        token = await self.get_token(user_id)
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
        await self.save_token(refreshed, source="refresh")
        return refreshed

    async def save_token(self, token: BitrixOAuthToken, *, source: str) -> None:
        data = {
            "user_id": token.user_id,
            "access_token": token.access_token,
            "refresh_token": token.refresh_token,
            "client_endpoint": token.client_endpoint,
            "server_endpoint": token.server_endpoint,
            "domain": token.domain,
            "member_id": token.member_id,
            "scope": token.scope,
            "expires_at": token.expires_at.isoformat(),
            "updated_at": token.updated_at.isoformat(),
            "source": source,
        }
        await self._redis.set(f"{_KEY_PREFIX}{token.user_id}", json.dumps(data, ensure_ascii=False))

    async def public_status(self) -> dict[str, Any]:
        settings = self._settings
        keys = await self._redis.keys(f"{_KEY_PREFIX}*")
        linked_users = []
        for key in sorted(keys)[:20]:
            raw = await self._redis.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            linked_users.append(
                {
                    "user_id": int(data.get("user_id", 0)),
                    "domain": data.get("domain", ""),
                    "scope": data.get("scope", ""),
                    "expires_at": data.get("expires_at", ""),
                    "updated_at": data.get("updated_at", ""),
                }
            )
        return {
            "enabled": settings.bitrix_oauth_enabled,
            "configured": settings.bitrix_oauth_configured,
            "required_for_writes": settings.bitrix_oauth_required_for_writes,
            "linked_users_count": len(keys),
            "linked_users": linked_users,
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


def _dict_to_token(data: dict[str, Any]) -> BitrixOAuthToken:
    return BitrixOAuthToken(
        user_id=int(data["user_id"]),
        access_token=str(data["access_token"]),
        refresh_token=str(data["refresh_token"]),
        client_endpoint=str(data["client_endpoint"]),
        server_endpoint=str(data["server_endpoint"]),
        domain=str(data["domain"]),
        member_id=str(data.get("member_id") or ""),
        scope=str(data.get("scope") or ""),
        expires_at=_parse_datetime(str(data["expires_at"])),
        updated_at=_parse_datetime(str(data["updated_at"])),
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
