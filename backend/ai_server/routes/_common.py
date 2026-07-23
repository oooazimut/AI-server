from __future__ import annotations

from datetime import UTC, datetime

from fastapi import HTTPException, Request, status

from ..settings import Settings


def request_secret(request: Request, header_value: str | None = None) -> str | None:
    return (
        header_value
        or request.query_params.get("secret")
        or request.query_params.get("agent_secret")
        or request.query_params.get("token")
    )


def validate_webhook_secret(settings: Settings, value: str | None) -> None:
    if settings.webhook_secret and value != settings.webhook_secret:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid webhook secret")


def validate_admin_secret(settings: Settings, value: str | None) -> None:
    if not settings.admin_api_secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="admin API is not configured")
    if value != settings.admin_api_secret:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid admin secret")


def now_ts() -> str:
    return datetime.now(UTC).isoformat()
