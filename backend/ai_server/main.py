import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated, Any
from urllib.parse import parse_qsl
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Query, Request, status

from .channels.bitrix import BitrixWebhookProcessor
from .integrations.bitrix.client import BitrixClient
from .integrations.bitrix.events import payload_event_type
from .integrations.bitrix.oauth import BitrixOAuthService
from .integrations.bitrix.portal_search import (
    PortalSearchIndex,
    entity_types_for_scope,
    format_portal_index_stats,
    format_portal_search_results,
)
from .knowledge import MarkdownKnowledgeBase
from .models import AgentTask, AgentTestRequest, UserContext
from .retrieval import HybridKnowledgeRetriever
from .orchestrator import suggest_agents
from .orchestrators.internal import InternalOrchestrator
from .registry import get_agent_manifest, load_agent_manifests, summarize_agents
from .runtime import ensure_runtime_dirs
from .settings import get_settings
from .skills import SkillStore
from .workers.bitrix.webhook_event_queue import WebhookEventQueue, run_webhook_event_worker
from .workers.registry import (
    get_automation_manifest,
    load_automation_manifests,
    summarize_automations,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ensure_runtime_dirs()
    bitrix = BitrixClient()
    bitrix_oauth = BitrixOAuthService()
    bitrix_oauth.ensure_schema()
    portal_search = PortalSearchIndex()
    webhook_event_queue = WebhookEventQueue(settings.webhook_event_queue_path)
    webhook_event_queue.ensure_schema()

    app.state.bitrix = bitrix
    app.state.bitrix_oauth = bitrix_oauth
    app.state.portal_search = portal_search
    app.state.webhook_event_queue = webhook_event_queue
    app.state.webhook_event_status = {
        "enabled": True,
        "mode": "webhook",
        "webhook_url_configured": bool(settings.resolved_bot_webhook_url),
        "secret_required": bool(settings.webhook_secret),
        "last_received_at": None,
        "last_event": None,
        "events_seen": 0,
        "duplicates_seen": 0,
    }
    app.state.webhook_event_queue_status = {
        "enabled": settings.webhook_event_queue_enabled,
        "running": False,
        "path": str(settings.webhook_event_queue_path),
        "worker_enabled": settings.webhook_event_worker_enabled,
        "worker_count": settings.webhook_event_queue_worker_count,
        "last_enqueued_at": None,
        "last_enqueued_event_id": None,
        "last_enqueued_event": None,
        "enqueued": 0,
        "duplicates_seen": 0,
        "processed": 0,
        "errors": 0,
        "last_error": None,
    }

    worker_task: asyncio.Task | None = None
    if settings.webhook_event_queue_enabled and settings.webhook_event_worker_enabled:
        processor = BitrixWebhookProcessor(bitrix=bitrix)
        worker_task = asyncio.create_task(
            run_webhook_event_worker(
                webhook_event_queue,
                processor.process,
                status=app.state.webhook_event_queue_status,
            )
        )

    try:
        yield
    finally:
        if worker_task:
            app.state.webhook_event_queue_status["running"] = False
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="AI Server", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    manifests = load_agent_manifests()
    settings = get_settings()
    return {
        "status": "ok",
        "architecture": "orchestrator_plus_modular_specialists",
        "agent_count": len(manifests),
        "agents": [agent.id for agent in manifests],
        "bitrix_configured": settings.bitrix_configured,
        "bitrix_webhook_queue_enabled": settings.webhook_event_queue_enabled,
        "bitrix_webhook_worker_enabled": settings.webhook_event_worker_enabled,
    }


@app.get("/agents")
def agents():
    manifests = load_agent_manifests()
    return summarize_agents(manifests)


@app.get("/agents/{agent_id}")
def agent_detail(agent_id: str):
    manifest = get_agent_manifest(agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return manifest


@app.get("/agents/{agent_id}/skills")
def agent_skills(agent_id: str):
    manifest = get_agent_manifest(agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return SkillStore().list_skills(manifest)


@app.get("/agents/{agent_id}/knowledge/topics")
def agent_knowledge_topics(agent_id: str):
    manifest = get_agent_manifest(agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return MarkdownKnowledgeBase().list_topics(manifest)


@app.get("/agents/{agent_id}/knowledge/search")
def agent_knowledge_search(
    agent_id: str,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
    topic: str | None = None,
):
    manifest = get_agent_manifest(agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return HybridKnowledgeRetriever().search(manifest, q, limit=limit, topic=topic)


@app.get("/agents/{agent_id}/automations")
def agent_automations(agent_id: str):
    manifest = get_agent_manifest(agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return summarize_automations(load_automation_manifests(agent_id=agent_id))


@app.get("/automations")
def automations(agent_id: str | None = None):
    if agent_id is not None and get_agent_manifest(agent_id) is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return summarize_automations(load_automation_manifests(agent_id=agent_id))


@app.get("/automations/{automation_id}")
def automation_detail(automation_id: str):
    automation = get_automation_manifest(automation_id)
    if automation is None:
        raise HTTPException(status_code=404, detail="automation not found")
    return automation


@app.get("/bitrix/status")
def bitrix_status(request: Request) -> dict[str, Any]:
    settings = get_settings()
    return {
        "configured": settings.bitrix_configured,
        "bot_id": settings.bitrix_bot_id,
        "bot_auth_mode": settings.bitrix_bot_auth_mode,
        "webhook_url_configured": bool(settings.resolved_bot_webhook_url),
        "oauth": request.app.state.bitrix_oauth.public_status(),
        "portal_search": _portal_search_status(request.app.state.portal_search),
        "webhook_events": dict(request.app.state.webhook_event_status),
        "webhook_event_queue": {
            **dict(request.app.state.webhook_event_queue_status),
            **request.app.state.webhook_event_queue.stats(),
        },
    }


@app.get("/bitrix/oauth/status")
def bitrix_oauth_status(request: Request) -> dict[str, Any]:
    return request.app.state.bitrix_oauth.public_status()


@app.get("/bitrix/webhook-events/status")
def bitrix_webhook_events_status(request: Request) -> dict[str, Any]:
    return {
        "worker": dict(request.app.state.webhook_event_queue_status),
        "queue": request.app.state.webhook_event_queue.stats(),
        "latest_events": request.app.state.webhook_event_queue.latest(limit=20),
    }


@app.get("/bitrix/search/status")
def bitrix_search_status(request: Request) -> dict[str, Any]:
    return _portal_search_status(request.app.state.portal_search)


@app.get("/bitrix/search")
def bitrix_search(
    request: Request,
    q: str = Query(..., min_length=1),
    scope: str = Query(default="all"),
    limit: int = Query(default=10, ge=1, le=30),
) -> dict[str, Any]:
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


@app.post("/bitrix/events")
async def bitrix_events(
    request: Request,
    x_agent_secret: Annotated[str | None, Header(alias="X-Agent-Secret")] = None,
) -> dict[str, Any]:
    settings = get_settings()
    payload = await _read_bitrix_event_payload(request)
    _validate_webhook_secret(
        settings,
        x_agent_secret
        or request.query_params.get("secret")
        or request.query_params.get("agent_secret")
        or request.query_params.get("token")
        or _payload_secret(payload),
    )

    event_type = payload_event_type(payload)
    webhook_status = request.app.state.webhook_event_status
    webhook_status["last_received_at"] = _now_ts()
    webhook_status["events_seen"] = int(webhook_status.get("events_seen") or 0) + 1
    webhook_status["last_event"] = event_type

    if settings.webhook_event_queue_enabled:
        event_id, inserted = request.app.state.webhook_event_queue.enqueue(
            payload,
            event_type=event_type,
        )
        queue_status = request.app.state.webhook_event_queue_status
        queue_status["last_enqueued_at"] = _now_ts()
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

    processor = BitrixWebhookProcessor(bitrix=request.app.state.bitrix)
    result = await processor.process(payload)
    return {"ok": True, **result}


@app.get("/route-preview")
def route_preview(q: str = Query(..., min_length=1)):
    manifests = load_agent_manifests()
    matches = suggest_agents(q, manifests)
    return summarize_agents(matches)


@app.post("/orchestrator/test")
async def orchestrator_test(body: AgentTestRequest):
    manifests = load_agent_manifests()
    task = AgentTask(
        task_id=str(uuid4()),
        source="local_test",
        user=UserContext(id=body.user_id, channel=body.channel, raw={"dialog_id": body.dialog_id}),
        request=body.text,
    )
    return await InternalOrchestrator(manifests).handle(task)


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


def _validate_webhook_secret(settings, value: str | None) -> None:
    if settings.webhook_secret and value != settings.webhook_secret:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid webhook secret")


def _now_ts() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _portal_search_status(index: PortalSearchIndex) -> dict[str, Any]:
    stats = index.stats()
    return {
        "exists": stats.exists,
        "path": str(stats.path),
        "summary": format_portal_index_stats(stats),
        "total_items": stats.total_items,
        "by_type": stats.by_type,
        "content_by_status": stats.content_by_status,
        "last_indexed_at": stats.last_indexed_at,
    }

