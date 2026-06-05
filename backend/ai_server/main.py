import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated, Any
from urllib.parse import parse_qsl
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from .channels.bitrix import BitrixWebhookProcessor
from .integrations.bitrix.client import BitrixClient
from .integrations.bitrix.events import payload_event_type
from .integrations.bitrix.oauth import BitrixOAuthService
from .integrations.bitrix.portal_search import (
    PortalSearchIndex,
    entity_types_for_scope,
    format_portal_content_sync_stats,
    format_portal_delta_sync_stats,
    format_portal_index_stats,
    format_portal_search_results,
    format_portal_sync_stats,
)
from .knowledge import MarkdownKnowledgeBase
from .learning import LearningEventRecorder
from .models import AgentTask, AgentTestRequest, LearningFeedbackRequest, UserContext
from .retrieval import HybridKnowledgeRetriever
from .orchestrators.internal import InternalOrchestrator
from .registry import get_agent_manifest, load_agent_manifests
from .runtime import ensure_runtime_dirs
from .settings import get_settings
from .skills import SkillStore
from .workers.bitrix.webhook_event_queue import WebhookEventQueue, run_webhook_event_worker
from .workers.bitrix.search_indexer import PortalSearchIndexerWorker
from .workers.bitrix.reconciler import reconcile_once, run_reconciler
from .workers.bitrix.supervisor import run_task_supervisor, run_task_supervisor_once
from .workers.logistics.vehicle_usage import run_vehicle_usage_once, run_vehicle_usage_worker
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
    portal_search.ensure_schema()
    portal_search_indexer = PortalSearchIndexerWorker(bitrix, portal_search)
    learning_recorder = LearningEventRecorder()
    webhook_event_queue = WebhookEventQueue(settings.webhook_event_queue_path)
    webhook_event_queue.ensure_schema()

    app.state.bitrix = bitrix
    app.state.bitrix_oauth = bitrix_oauth
    app.state.portal_search = portal_search
    app.state.portal_search_indexer = portal_search_indexer
    app.state.learning_recorder = learning_recorder
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
    app.state.search_webhook_indexer_status = {
        "enabled": settings.search_webhook_indexer_enabled,
        "events_seen": 0,
        "processed": 0,
        "errors": 0,
        "last_received_at": None,
        "last_event": None,
        "last_file_id": None,
        "last_action": None,
        "last_reason": None,
        "last_error": None,
        "last_result": None,
    }
    app.state.quality_control_webhook_status = {
        "enabled": settings.quality_control_webhook_enabled,
        "auto_managed_only": settings.quality_control_webhook_auto_managed_only,
        "auto_manage_project_id": settings.quality_control_auto_manage_project_id,
        "dry_run": settings.quality_control_dry_run,
        "actor_user_id": settings.quality_control_actor_user_id,
        "last_received_at": None,
        "last_event": None,
        "last_task_id": None,
        "last_reason": None,
        "last_error": None,
        "last_actions": [],
        "events_seen": 0,
        "tasks_processed": 0,
        "duplicates_seen": 0,
        "ignored": 0,
        "errors": 0,
    }
    app.state.task_supervisor_status = {
        "enabled": settings.supervisor_enabled,
        "running": False,
        "dry_run": settings.supervisor_dry_run,
        "interval_seconds": settings.supervisor_interval_seconds,
        "last_check_at": None,
        "last_success_at": None,
        "last_error": None,
        "next_check_at": None,
        "runs": 0,
        "errors": 0,
    }
    app.state.reconciler_status = {
        "enabled": settings.reconcile_enabled,
        "running": False,
        "interval_seconds": settings.reconcile_interval_seconds,
        "task_lookback_hours": settings.reconcile_task_lookback_hours,
        "last_check_at": None,
        "last_success_at": None,
        "last_error": None,
        "next_check_at": None,
        "runs": 0,
        "errors": 0,
    }
    app.state.vehicle_usage_status = {
        "enabled": settings.vehicle_usage_enabled,
        "running": False,
        "dry_run": settings.vehicle_usage_dry_run,
        "interval_seconds": settings.vehicle_usage_interval_seconds,
        "dialog_id": settings.vehicle_usage_dialog_id,
        "manager_user_id": settings.vehicle_usage_manager_user_id,
        "admin_notify_user_ids": settings.resolved_vehicle_usage_admin_notify_user_ids,
        "request_time": settings.vehicle_usage_request_time,
        "request_times": settings.vehicle_usage_request_times,
        "escalation_time": settings.vehicle_usage_escalation_time,
        "last_check_at": None,
        "last_sent_at": None,
        "last_escalated_at": None,
        "last_error": None,
        "runs": 0,
        "errors": 0,
    }

    webhook_worker_task: asyncio.Task | None = None
    search_indexer_task: asyncio.Task | None = None
    supervisor_task: asyncio.Task | None = None
    reconciler_task: asyncio.Task | None = None
    vehicle_usage_task: asyncio.Task | None = None
    if settings.webhook_event_queue_enabled and settings.webhook_event_worker_enabled:
        processor = BitrixWebhookProcessor(
            bitrix=bitrix,
            portal_search=portal_search,
            bitrix_oauth=bitrix_oauth,
            search_webhook_status=app.state.search_webhook_indexer_status,
            quality_control_status=app.state.quality_control_webhook_status,
            learning_recorder=learning_recorder,
        )
        webhook_worker_task = asyncio.create_task(
            run_webhook_event_worker(
                webhook_event_queue,
                processor.process,
                status=app.state.webhook_event_queue_status,
            )
        )
    if settings.search_background_indexer_enabled:
        search_indexer_task = asyncio.create_task(portal_search_indexer.run())
    if settings.supervisor_enabled:
        supervisor_task = asyncio.create_task(
            run_task_supervisor(bitrix, status=app.state.task_supervisor_status)
        )
    if settings.reconcile_enabled:
        reconciler_task = asyncio.create_task(
            run_reconciler(
                bitrix,
                webhook_event_queue,
                portal_search_indexer,
                status=app.state.reconciler_status,
            )
        )
    if settings.vehicle_usage_enabled:
        vehicle_usage_task = asyncio.create_task(
            run_vehicle_usage_worker(bitrix, status=app.state.vehicle_usage_status)
        )

    try:
        yield
    finally:
        if webhook_worker_task:
            app.state.webhook_event_queue_status["running"] = False
            webhook_worker_task.cancel()
            try:
                await webhook_worker_task
            except asyncio.CancelledError:
                pass
        if search_indexer_task:
            app.state.portal_search_indexer.status["running"] = False
            search_indexer_task.cancel()
            try:
                await search_indexer_task
            except asyncio.CancelledError:
                pass
        if supervisor_task:
            app.state.task_supervisor_status["running"] = False
            supervisor_task.cancel()
            try:
                await supervisor_task
            except asyncio.CancelledError:
                pass
        if reconciler_task:
            app.state.reconciler_status["running"] = False
            reconciler_task.cancel()
            try:
                await reconciler_task
            except asyncio.CancelledError:
                pass
        if vehicle_usage_task:
            app.state.vehicle_usage_status["running"] = False
            vehicle_usage_task.cancel()
            try:
                await vehicle_usage_task
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
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_configured": settings.llm_configured,
        "tech_footer_enabled": settings.tech_footer_enabled,
        "tech_footer_allowed_user_ids": settings.resolved_tech_footer_allowed_user_ids,
        "deepseek_balance_configured": bool(settings.deepseek_api_key),
        "learning_events_enabled": settings.learning_events_enabled,
        "learning_events_path": str(settings.learning_events_path),
        "bitrix_webhook_queue_enabled": settings.webhook_event_queue_enabled,
        "bitrix_webhook_worker_enabled": settings.webhook_event_worker_enabled,
        "bitrix_search_indexer_enabled": settings.search_background_indexer_enabled,
        "bitrix_quality_control_enabled": settings.quality_control_webhook_enabled,
        "bitrix_quality_control_dry_run": settings.quality_control_dry_run,
        "bitrix_task_supervisor_enabled": settings.supervisor_enabled,
        "bitrix_reconciler_enabled": settings.reconcile_enabled,
        "logistics_vehicle_usage_enabled": settings.vehicle_usage_enabled,
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


@app.get("/learning/status")
def learning_status(request: Request) -> dict[str, Any]:
    return request.app.state.learning_recorder.stats()


@app.get("/learning/events")
def learning_events(
    request: Request,
    x_agent_secret: Annotated[str | None, Header(alias="X-Agent-Secret")] = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    _validate_webhook_secret(get_settings(), _request_secret(request, x_agent_secret))
    recorder: LearningEventRecorder = request.app.state.learning_recorder
    return {"events": recorder.latest(limit=limit), "status": recorder.stats()}


@app.post("/learning/feedback")
def learning_feedback(
    request: Request,
    body: LearningFeedbackRequest,
    x_agent_secret: Annotated[str | None, Header(alias="X-Agent-Secret")] = None,
) -> dict[str, Any]:
    _validate_webhook_secret(get_settings(), _request_secret(request, x_agent_secret))
    recorder: LearningEventRecorder = request.app.state.learning_recorder
    return recorder.record_feedback(
        event_id=body.event_id,
        rating=body.rating,
        corrected_answer=body.corrected_answer,
        comment=body.comment,
        tags=body.tags,
        user_id=body.user_id,
        channel=body.channel,
    )


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
        "portal_search_indexer": request.app.state.portal_search_indexer.public_status(),
        "search_webhook_indexer": dict(request.app.state.search_webhook_indexer_status),
        "quality_control": dict(request.app.state.quality_control_webhook_status),
        "task_supervisor": dict(request.app.state.task_supervisor_status),
        "reconciler": dict(request.app.state.reconciler_status),
        "webhook_events": dict(request.app.state.webhook_event_status),
        "webhook_event_queue": {
            **dict(request.app.state.webhook_event_queue_status),
            **request.app.state.webhook_event_queue.stats(),
        },
    }


@app.get("/agent/status")
def legacy_agent_status(request: Request) -> dict[str, Any]:
    return {
        **bitrix_status(request),
        "agent_runtime": "multi_agent",
        "vehicle_usage": dict(request.app.state.vehicle_usage_status),
    }


@app.get("/agent/vehicles/status")
def legacy_vehicle_usage_status(request: Request) -> dict[str, Any]:
    return logistics_vehicle_usage_status(request)


@app.get("/bitrix/oauth/status")
def bitrix_oauth_status(request: Request) -> dict[str, Any]:
    return request.app.state.bitrix_oauth.public_status()


@app.api_route("/bitrix/app", methods=["GET", "POST"], response_class=HTMLResponse)
async def bitrix_app(request: Request):
    settings = get_settings()
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
            (
                "<p>OAuth пока не настроен: не задан `BITRIX_OAUTH_CLIENT_ID` "
                "или публичный `PUBLIC_BASE_URL`.</p>"
            ),
        )
    )


@app.post("/bitrix/install", response_class=HTMLResponse)
async def bitrix_install(request: Request) -> HTMLResponse:
    payload = await _read_bitrix_event_payload(request)
    if request.query_params:
        payload = {**dict(request.query_params), **payload}
    result = await request.app.state.bitrix_oauth.save_from_payload(payload, source="bitrix_install")
    return _oauth_success_page(result.user_id, result.expires_at.isoformat())


@app.get("/bitrix/oauth/callback", response_class=HTMLResponse)
async def bitrix_oauth_callback(request: Request) -> HTMLResponse:
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth code is missing")
    result = await request.app.state.bitrix_oauth.exchange_authorization_code(
        code=str(code),
        source="oauth_callback",
    )
    return _oauth_success_page(result.user_id, result.expires_at.isoformat())


@app.get("/bitrix/oauth/start")
def bitrix_oauth_start() -> RedirectResponse:
    settings = get_settings()
    if not settings.resolved_bitrix_oauth_start_url:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Bitrix OAuth client id or domain is not configured",
        )
    return RedirectResponse(settings.resolved_bitrix_oauth_start_url)


@app.get("/bitrix/webhook-events/status")
def bitrix_webhook_events_status(request: Request) -> dict[str, Any]:
    return {
        "worker": dict(request.app.state.webhook_event_queue_status),
        "queue": request.app.state.webhook_event_queue.stats(),
        "latest_events": request.app.state.webhook_event_queue.latest(limit=20),
    }


@app.get("/agent/webhook-events/status")
def legacy_webhook_events_status(request: Request) -> dict[str, Any]:
    return bitrix_webhook_events_status(request)


@app.get("/bitrix/search/status")
def bitrix_search_status(request: Request) -> dict[str, Any]:
    return {
        **_portal_search_status(request.app.state.portal_search),
        "indexer": request.app.state.portal_search_indexer.public_status(),
        "webhook_indexer": dict(request.app.state.search_webhook_indexer_status),
    }


@app.get("/agent/search/status")
def legacy_bitrix_search_status(request: Request) -> dict[str, Any]:
    return bitrix_search_status(request)


@app.get("/bitrix/search/indexer/status")
def bitrix_search_indexer_status(request: Request) -> dict[str, Any]:
    return request.app.state.portal_search_indexer.public_status()


@app.get("/agent/search/indexer/status")
def legacy_bitrix_search_indexer_status(request: Request) -> dict[str, Any]:
    return bitrix_search_indexer_status(request)


@app.get("/bitrix/search/webhook-indexer/status")
def bitrix_search_webhook_indexer_status(request: Request) -> dict[str, Any]:
    return dict(request.app.state.search_webhook_indexer_status)


@app.get("/bitrix/quality-control/status")
def bitrix_quality_control_status(request: Request) -> dict[str, Any]:
    return dict(request.app.state.quality_control_webhook_status)


@app.get("/bitrix/supervisor/status")
def bitrix_supervisor_status(request: Request) -> dict[str, Any]:
    return dict(request.app.state.task_supervisor_status)


@app.post("/bitrix/supervisor/run-once")
async def bitrix_supervisor_run_once(request: Request) -> dict[str, Any]:
    result = await run_task_supervisor_once(
        request.app.state.bitrix,
        status=request.app.state.task_supervisor_status,
    )
    return {"ok": True, **result, "status": dict(request.app.state.task_supervisor_status)}


@app.get("/bitrix/reconciler/status")
def bitrix_reconciler_status(request: Request) -> dict[str, Any]:
    return dict(request.app.state.reconciler_status)


@app.get("/agent/reconcile/status")
def legacy_reconciler_status(request: Request) -> dict[str, Any]:
    return {"enabled": get_settings().reconcile_enabled, "status": dict(request.app.state.reconciler_status)}


@app.post("/bitrix/reconciler/run-once")
async def bitrix_reconciler_run_once(request: Request) -> dict[str, Any]:
    result = await reconcile_once(
        request.app.state.bitrix,
        request.app.state.webhook_event_queue,
        request.app.state.portal_search_indexer,
        status=request.app.state.reconciler_status,
    )
    return {"ok": True, "result": result, "status": dict(request.app.state.reconciler_status)}


@app.get("/logistics/vehicle-usage/status")
def logistics_vehicle_usage_status(request: Request) -> dict[str, Any]:
    from .tools.vehicle_usage import VehicleUsageStore

    store = VehicleUsageStore()
    return {
        "status": dict(request.app.state.vehicle_usage_status),
        "latest_requests": store.latest_requests(limit=10),
    }


@app.post("/logistics/vehicle-usage/run-once")
async def logistics_vehicle_usage_run_once(request: Request) -> dict[str, Any]:
    result = await run_vehicle_usage_once(
        request.app.state.bitrix,
        status=request.app.state.vehicle_usage_status,
    )
    return {"ok": True, "result": result, "status": dict(request.app.state.vehicle_usage_status)}


@app.post("/bitrix/search/reindex")
async def bitrix_search_reindex(request: Request) -> dict[str, Any]:
    try:
        stats = await request.app.state.portal_search_indexer.run_metadata_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "summary": format_portal_sync_stats(stats),
        "stats": stats,
        "indexer": request.app.state.portal_search_indexer.public_status(),
    }


@app.post("/agent/search/reindex")
async def legacy_bitrix_search_reindex(request: Request) -> dict[str, Any]:
    return await bitrix_search_reindex(request)


@app.post("/bitrix/search/reindex-delta")
async def bitrix_search_reindex_delta(request: Request) -> dict[str, Any]:
    try:
        stats = await request.app.state.portal_search_indexer.run_delta_once()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "summary": format_portal_delta_sync_stats(stats),
        "stats": stats,
        "indexer": request.app.state.portal_search_indexer.public_status(),
    }


@app.post("/bitrix/search/reindex-content")
async def bitrix_search_reindex_content(
    request: Request,
    extensions: str | None = None,
) -> dict[str, Any]:
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


@app.post("/agent/search/reindex-content")
async def legacy_bitrix_search_reindex_content(
    request: Request,
    extensions: str | None = None,
) -> dict[str, Any]:
    return await bitrix_search_reindex_content(request, extensions=extensions)


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


@app.get("/agent/search")
def legacy_bitrix_search(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=30),
) -> dict[str, Any]:
    return bitrix_search(request, q=q, scope="all", limit=limit)


@app.get("/agent/tools")
def legacy_agent_tools() -> dict[str, Any]:
    manifests = load_agent_manifests()
    return {
        "tools": [
            {"agent_id": agent.id, "tools": agent.tools, "capabilities": agent.capabilities}
            for agent in manifests
        ]
    }


@app.get("/agent/search/readiness")
def legacy_search_readiness(request: Request) -> dict[str, Any]:
    status_data = bitrix_search_status(request)
    return {"summary": status_data.get("summary", ""), **status_data}


@app.get("/agent/search/production-status")
def legacy_search_production_status(request: Request) -> dict[str, Any]:
    status_data = bitrix_search_status(request)
    return {"summary": status_data.get("summary", ""), **status_data}


@app.post("/agent/documents/compare")
def legacy_documents_compare() -> dict[str, Any]:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Document comparison now belongs to the PTO LLM specialist; use the orchestrator/chat flow.",
    )


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

    processor = BitrixWebhookProcessor(
        bitrix=request.app.state.bitrix,
        portal_search=request.app.state.portal_search,
        bitrix_oauth=request.app.state.bitrix_oauth,
        search_webhook_status=request.app.state.search_webhook_indexer_status,
        quality_control_status=request.app.state.quality_control_webhook_status,
        learning_recorder=request.app.state.learning_recorder,
    )
    result = await processor.process(payload)
    return {"ok": True, **result}


@app.post("/orchestrator/test")
async def orchestrator_test(request: Request, body: AgentTestRequest):
    manifests = load_agent_manifests()
    task = AgentTask(
        task_id=str(uuid4()),
        source="local_test",
        user=UserContext(id=body.user_id, channel=body.channel, raw={"dialog_id": body.dialog_id}),
        request=body.text,
    )
    result = await InternalOrchestrator(manifests).handle(task)
    request.app.state.learning_recorder.record_agent_result(
        task,
        result,
        metadata={"endpoint": "/orchestrator/test", "dialog_id": body.dialog_id},
    )
    return result


@app.post("/agent/test")
async def legacy_agent_test(request: Request, body: AgentTestRequest):
    return await orchestrator_test(request, body)


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
        "<html lang=\"ru\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{title}</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:32px;line-height:1.45;color:#1f2937}"
        "main{max-width:760px}"
        "code{background:#f3f4f6;padding:2px 5px;border-radius:4px}"
        "</style></head><body><main>"
        f"<h1>{title}</h1>{body}"
        "</main></body></html>"
    )


def _request_secret(request: Request, header_value: str | None = None) -> str | None:
    return (
        header_value
        or request.query_params.get("secret")
        or request.query_params.get("agent_secret")
        or request.query_params.get("token")
    )


def _validate_webhook_secret(settings, value: str | None) -> None:
    if settings.webhook_secret and value != settings.webhook_secret:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="invalid webhook secret")


def _now_ts() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _portal_search_status(index: PortalSearchIndex) -> dict[str, Any]:
    settings = get_settings()
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

