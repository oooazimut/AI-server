from __future__ import annotations

import json
from typing import Annotated, Any
from urllib.parse import parse_qsl

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from ..integrations.bitrix.events import MESSAGE_EVENTS, payload_event_type
from ..integrations.bitrix.portal_search import (
    PortalSearchIndex,
    entity_types_for_scope,
    format_portal_content_sync_stats,
    format_portal_delta_sync_stats,
    format_portal_index_stats,
    format_portal_search_results,
    format_portal_sync_stats,
)
from ._common import now_ts, validate_admin_secret, validate_webhook_secret

router = APIRouter()


# ---------------------------------------------------------------------------
# Status endpoints
# ---------------------------------------------------------------------------


@router.get("/bitrix/status")
async def bitrix_status(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    queue_stats = await request.app.state.webhook_event_queue.stats()
    worker_status = _webhook_worker_status(
        request.app.state.webhook_event_queue_status,
        queue_stats.pop("worker_heartbeat", None),
    )
    return {
        "configured": settings.bitrix_configured,
        "bot_id": settings.bitrix_bot_id,
        "bot_auth_mode": settings.bitrix_bot_auth_mode,
        "webhook_url_configured": bool(settings.resolved_bot_webhook_url),
        "oauth": await request.app.state.bitrix_oauth.public_status(),
        "portal_search": _portal_search_status(request.app.state.portal_search, request.app.state.settings),
        "portal_search_indexer": request.app.state.portal_search_indexer.public_status(),
        "search_webhook_indexer": dict(request.app.state.search_webhook_indexer_status),
        "reconciler": dict(request.app.state.reconciler_status),
        "webhook_events": dict(request.app.state.webhook_event_status),
        "webhook_event_queue": {**worker_status, **queue_stats},
        "outbound_queue": await request.app.state.outbound_queue.stats(),
    }


@router.get("/bitrix/oauth/status")
async def bitrix_oauth_status(request: Request) -> dict[str, Any]:
    return await request.app.state.bitrix_oauth.public_status()


# ---------------------------------------------------------------------------
# OAuth / app install
# ---------------------------------------------------------------------------


@router.api_route("/bitrix/app", methods=["GET", "POST"], response_class=HTMLResponse)
async def bitrix_app(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    payload = await _read_bitrix_event_payload(request)
    if request.query_params:
        payload = {**dict(request.query_params), **payload}
    if request.query_params.get("code"):
        result = await request.app.state.bitrix_oauth.exchange_authorization_code(
            code=str(request.query_params["code"]),
            source="oauth_callback",
        )
        return _oauth_success_page(result.user_id, result.expires_at.isoformat())

    if _payload_has_oauth(payload):
        result = await request.app.state.bitrix_oauth.save_from_payload(payload, source="bitrix_app")
        return _oauth_success_page(result.user_id, result.expires_at.isoformat())

    if settings.resolved_bitrix_oauth_start_url:
        return RedirectResponse(settings.resolved_bitrix_oauth_start_url)

    return HTMLResponse(
        _html_page(
            "AI-помощник",
            "<p>OAuth пока не настроен: не задан `BITRIX_OAUTH_CLIENT_ID` или публичный `PUBLIC_BASE_URL`.</p>",
        )
    )


@router.post("/bitrix/install", response_class=HTMLResponse)
async def bitrix_install(request: Request) -> HTMLResponse:
    payload = await _read_bitrix_event_payload(request)
    if request.query_params:
        payload = {**dict(request.query_params), **payload}
    result = await request.app.state.bitrix_oauth.save_from_payload(payload, source="bitrix_install")
    return _oauth_success_page(result.user_id, result.expires_at.isoformat())


@router.get("/bitrix/oauth/callback", response_class=HTMLResponse)
async def bitrix_oauth_callback(request: Request) -> HTMLResponse:
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth code is missing")
    result = await request.app.state.bitrix_oauth.exchange_authorization_code(
        code=str(code),
        source="oauth_callback",
    )
    return _oauth_success_page(result.user_id, result.expires_at.isoformat())


@router.get("/bitrix/oauth/start")
def bitrix_oauth_start(request: Request) -> RedirectResponse:
    settings = request.app.state.settings
    if not settings.resolved_bitrix_oauth_start_url:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Bitrix OAuth client id or domain is not configured",
        )
    return RedirectResponse(settings.resolved_bitrix_oauth_start_url)


# ---------------------------------------------------------------------------
# Webhook events
# ---------------------------------------------------------------------------


@router.get("/bitrix/webhook-events/status")
async def bitrix_webhook_events_status(request: Request) -> dict[str, Any]:
    queue_stats = await request.app.state.webhook_event_queue.stats()
    worker_status = _webhook_worker_status(
        request.app.state.webhook_event_queue_status,
        queue_stats.pop("worker_heartbeat", None),
    )
    return {
        "worker": worker_status,
        "queue": queue_stats,
        "outbound_queue": await request.app.state.outbound_queue.stats(),
        "latest_events": await request.app.state.webhook_event_queue.latest(limit=20),
    }


@router.get("/admin/conversation-trace/recent")
async def conversation_trace_recent(
    request: Request,
    x_trace_secret: Annotated[str | None, Header(alias="X-Trace-Secret")] = None,
    secret: str = "",
    hours: int = Query(default=24, ge=1, le=48),
    limit: int = Query(default=100, ge=1, le=500),
    user_id: str = "",
    dialog_key: str = "",
    message_id: str = "",
    task_id: str = "",
) -> dict[str, Any]:
    _validate_trace_secret(request, provided=x_trace_secret or secret)
    trace = request.app.state.conversation_trace
    if message_id:
        events = await trace.by_message(message_id, limit=limit, hours=hours)
    elif task_id:
        events = await trace.by_task(task_id, limit=limit, hours=hours)
    elif dialog_key:
        events = await trace.by_dialog(dialog_key, limit=limit, hours=hours)
    elif user_id:
        events = await trace.by_user(user_id, limit=limit, hours=hours)
    else:
        events = await trace.recent(limit=limit, hours=hours)
    return {
        "enabled": trace.enabled,
        "hours": hours,
        "limit": limit,
        "count": len(events),
        "events": events,
    }


@router.get("/admin/outbound-queue/status")
async def outbound_queue_admin_status(
    request: Request,
    x_trace_secret: Annotated[str | None, Header(alias="X-Trace-Secret")] = None,
    secret: str = "",
) -> dict[str, Any]:
    _validate_trace_secret(request, provided=x_trace_secret or secret)
    return await request.app.state.outbound_queue.public_status()


def _webhook_worker_status(local_status: dict[str, Any], heartbeat: dict[str, Any] | None) -> dict[str, Any]:
    status = dict(local_status)
    if not heartbeat:
        return status
    if heartbeat.get("running"):
        status.update({key: value for key, value in heartbeat.items() if value is not None})
        return status
    status["heartbeat"] = heartbeat
    return status


@router.post("/bitrix/events")
async def bitrix_events(
    request: Request,
    x_agent_secret: Annotated[str | None, Header(alias="X-Agent-Secret")] = None,
) -> dict[str, Any]:
    settings = request.app.state.settings
    payload = await _read_bitrix_event_payload(request)
    validate_webhook_secret(
        settings,
        x_agent_secret
        or request.query_params.get("secret")
        or request.query_params.get("agent_secret")
        or request.query_params.get("token")
        or _payload_secret(payload),
    )

    event_type = payload_event_type(payload)
    webhook_status = request.app.state.webhook_event_status
    webhook_status["last_received_at"] = now_ts()
    webhook_status["events_seen"] = int(webhook_status.get("events_seen") or 0) + 1
    webhook_status["last_event"] = event_type

    if settings.webhook_event_queue_enabled:
        event_id, inserted = await request.app.state.webhook_event_queue.enqueue(
            payload,
            event_type=event_type,
        )
        await request.app.state.conversation_trace.record_ingress(
            event_id=event_id,
            event_type=event_type,
            payload=payload,
            inserted=inserted,
        )
        if event_type in MESSAGE_EVENTS:
            await request.app.state.conversation_trace.record_inbound(
                event_id=event_id,
                event_type=event_type,
                payload=payload,
                inserted=inserted,
            )
        queue_status = request.app.state.webhook_event_queue_status
        queue_status["last_enqueued_at"] = now_ts()
        queue_status["last_enqueued_event_id"] = event_id
        queue_status["last_enqueued_event"] = event_type
        queue_status["enqueued"] = int(queue_status.get("enqueued") or 0) + int(inserted)
        if not inserted:
            queue_status["duplicates_seen"] = int(queue_status.get("duplicates_seen") or 0) + 1
            webhook_status["duplicates_seen"] = int(webhook_status.get("duplicates_seen") or 0) + 1
        return {
            "ok": True,
            "queued": inserted,
            "duplicate": not inserted,
            "event": event_type,
            "event_id": event_id,
        }

    return {"ok": True, "skipped": True, "reason": "webhook_queue_disabled"}


def _validate_trace_secret(request: Request, *, provided: str | None) -> None:
    settings = request.app.state.settings
    if not settings.conversation_trace_secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation trace is not configured")
    if provided != settings.conversation_trace_secret:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid conversation trace secret")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@router.get("/bitrix/search/status")
def bitrix_search_status(request: Request) -> dict[str, Any]:
    return {
        **_portal_search_status(request.app.state.portal_search, request.app.state.settings),
        "indexer": request.app.state.portal_search_indexer.public_status(),
        "webhook_indexer": dict(request.app.state.search_webhook_indexer_status),
    }


@router.get("/bitrix/search/indexer/status")
def bitrix_search_indexer_status(request: Request) -> dict[str, Any]:
    return request.app.state.portal_search_indexer.public_status()


@router.get("/bitrix/search/webhook-indexer/status")
def bitrix_search_webhook_indexer_status(request: Request) -> dict[str, Any]:
    return dict(request.app.state.search_webhook_indexer_status)


@router.post("/admin/bitrix/search/reindex")
async def bitrix_search_reindex(
    request: Request,
    x_admin_secret: Annotated[str | None, Header(alias="X-Admin-Secret")] = None,
) -> dict[str, Any]:
    validate_admin_secret(request.app.state.settings, x_admin_secret)
    try:
        stats = await request.app.state.portal_search_indexer.run_metadata_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "summary": format_portal_sync_stats(stats),
        "stats": stats,
        "indexer": request.app.state.portal_search_indexer.public_status(),
    }


@router.post("/admin/bitrix/search/reindex-delta")
async def bitrix_search_reindex_delta(
    request: Request,
    x_admin_secret: Annotated[str | None, Header(alias="X-Admin-Secret")] = None,
) -> dict[str, Any]:
    validate_admin_secret(request.app.state.settings, x_admin_secret)
    try:
        stats = await request.app.state.portal_search_indexer.run_delta_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "summary": format_portal_delta_sync_stats(stats),
        "stats": stats,
        "indexer": request.app.state.portal_search_indexer.public_status(),
    }


@router.post("/admin/bitrix/search/reindex-content")
async def bitrix_search_reindex_content(
    request: Request,
    extensions: str | None = None,
    x_admin_secret: Annotated[str | None, Header(alias="X-Admin-Secret")] = None,
) -> dict[str, Any]:
    validate_admin_secret(request.app.state.settings, x_admin_secret)
    try:
        stats = await request.app.state.portal_search_indexer.run_content_once(
            extensions=_extension_set(extensions),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "summary": format_portal_content_sync_stats(stats),
        "stats": stats,
        "indexer": request.app.state.portal_search_indexer.public_status(),
    }


@router.get("/admin/bitrix/search")
def bitrix_search(
    request: Request,
    q: str = Query(..., min_length=1),
    scope: str = Query(default="all"),
    limit: int = Query(default=10, ge=1, le=30),
    x_admin_secret: Annotated[str | None, Header(alias="X-Admin-Secret")] = None,
) -> dict[str, Any]:
    validate_admin_secret(request.app.state.settings, x_admin_secret)
    index: PortalSearchIndex = request.app.state.portal_search
    entity_types = entity_types_for_scope(scope)
    if entity_types is None and scope.strip().lower() not in {"", "all"}:
        raise HTTPException(status_code=400, detail=f"unknown portal search scope: {scope}")
    stats = index.stats()
    if not stats.exists:
        raise HTTPException(status_code=409, detail=f"portal search index is missing: {stats.path}")
    results = index.search(q, entity_types=entity_types, limit=limit)
    return {
        "summary": format_portal_search_results(results, query=q),
        "query": q,
        "scope": scope,
        "limit": limit,
        "results": [result.as_dict() for result in results],
    }


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

@router.get("/bitrix/reconciler/status")
def bitrix_reconciler_status(request: Request) -> dict[str, Any]:
    return dict(request.app.state.reconciler_status)


@router.post("/admin/bitrix/reconciler/run-once")
async def bitrix_reconciler_run_once(
    request: Request,
    x_admin_secret: Annotated[str | None, Header(alias="X-Admin-Secret")] = None,
) -> dict[str, Any]:
    validate_admin_secret(request.app.state.settings, x_admin_secret)
    result = await request.app.state.reconcile_fn(status=request.app.state.reconciler_status)
    return {"ok": True, "result": result, "status": dict(request.app.state.reconciler_status)}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _read_bitrix_event_payload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        payload = await request.json()
        return payload if isinstance(payload, dict) else {}

    body = await request.body()
    if not body:
        return {}

    if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        pairs = parse_qsl(body.decode("utf-8"), keep_blank_values=True)
        return _expand_form_pairs(pairs)

    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        pairs = parse_qsl(body.decode("utf-8"), keep_blank_values=True)
        return _expand_form_pairs(pairs)
    return payload if isinstance(payload, dict) else {}


def _expand_form_pairs(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if "[" not in key:
            result[key] = value
            continue
        _assign_bracketed(result, key, value)
    return result


def _assign_bracketed(target: dict[str, Any], key: str, value: str) -> None:
    head, *raw_parts = key.replace("]", "").split("[")
    current = target.setdefault(head, {})
    for part in raw_parts[:-1]:
        if not isinstance(current, dict):
            return
        current = current.setdefault(part, {})
    if isinstance(current, dict) and raw_parts:
        current[raw_parts[-1]] = value


def _payload_secret(payload: dict[str, Any]) -> str | None:
    for key in ("secret", "agent_secret", "token", "WEBHOOK_SECRET"):
        value = payload.get(key)
        if value:
            return str(value)
    auth = payload.get("auth")
    if isinstance(auth, dict):
        value = auth.get("application_token") or auth.get("APPLICATION_TOKEN")
        if value:
            return str(value)
    return None


def _payload_has_oauth(payload: dict[str, Any]) -> bool:
    auth = payload.get("auth")
    if isinstance(auth, dict) and (auth.get("access_token") or auth.get("refresh_token")):
        return True
    return bool(payload.get("AUTH_ID") or payload.get("REFRESH_ID"))


def _oauth_success_page(user_id: int, expires_at: str) -> HTMLResponse:
    return HTMLResponse(
        _html_page(
            "OAuth подключён",
            (
                f"<p>Готово. OAuth-доступ для пользователя Bitrix #{user_id} сохранён.</p>"
                f"<p>Текущий access token действует примерно до: <code>{expires_at}</code>.</p>"
                "<p>Теперь AI-помощник сможет выполнять разрешённые действия от имени этого пользователя.</p>"
            ),
        )
    )


def _html_page(title: str, body: str) -> str:
    return (
        "<!doctype html>"
        '<html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#1f2937}"
        "main{max-width:760px}"
        "code{background:#f3f4f6;padding:2px 5px;border-radius:4px}"
        "</style></head><body><main>"
        f"<h1>{title}</h1>{body}"
        "</main></body></html>"
    )


def _portal_search_status(index: PortalSearchIndex, settings: Any) -> dict[str, Any]:
    stats = index.stats()
    content = (
        index.content_readiness(
            allowed_extensions=settings.resolved_search_content_allowed_extensions,
        ).as_dict()
        if stats.exists
        else {
            "total_documents": 0,
            "supported_documents": 0,
            "indexed": 0,
            "pending": 0,
            "terminal": 0,
            "unsupported": 0,
            "indexed_by_extension": {},
            "pending_by_extension": {},
            "pending_by_status": {},
            "terminal_by_status": {},
            "unsupported_by_extension": {},
        }
    )
    return {
        "exists": stats.exists,
        "path": str(stats.path),
        "summary": format_portal_index_stats(stats),
        "total_items": stats.total_items,
        "by_type": stats.by_type,
        "content_by_status": stats.content_by_status,
        "content": content,
        "last_indexed_at": stats.last_indexed_at,
    }


def _extension_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    result = {
        item.strip().lower() if item.strip().startswith(".") else f".{item.strip().lower()}"
        for item in value.replace(";", ",").split(",")
        if item.strip()
    }
    return result or None
