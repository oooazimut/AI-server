from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from .agent_scheduler import AgentScheduler
from .agents.bitrix24 import BitrixLLMService
from .agents.logistics import LogisticsLLMService
from .agents.logistics.specialist import VehicleUsageSettings
from .agents.pto import PtoLLMService
from .channels.bitrix import BitrixWebhookProcessor, build_orchestrator
from .integrations.bitrix.bitrix_store import BitrixAgentStore
from .integrations.bitrix.client import BitrixClient
from .integrations.bitrix.dialog_state import BitrixPendingActionService, DialogStateStore
from .integrations.bitrix.oauth import BitrixOAuthService
from .integrations.bitrix.portal_search import PortalSearchIndex
from .integrations.bitrix.ports import BitrixAgentStorePort
from .integrations.ports import VehicleUsageStorePort
from .integrations.postgres.bitrix_agent import PostgresBitrixAgentStore
from .integrations.postgres.orchestrator_agent import PostgresOrchestratorStore
from .integrations.postgres.portal_search import PostgresPortalSearchIndex
from .integrations.postgres.pto_agent import PostgresPtoAgentStore
from .integrations.postgres.vehicle_usage import PostgresVehicleUsageStore
from .integrations.redis.event_queue import RedisEventQueue
from .learning import LearningEventRecorder
from .orchestrators.orchestrator_llm import OrchestratorLLMService
from .registry import load_agent_manifests
from .runtime import ensure_runtime_dirs
from .settings import Settings, get_settings
from .specialists import SpecialistDeps
from .tools.vehicle_usage import VehicleUsageStore
from .workers.bitrix.quality_control_adapter import QualityControlHandlerAdapter
from .workers.bitrix.reconciler import reconcile_once, run_reconciler
from .workers.bitrix.search_indexer import PortalSearchIndexerWorker
from .workers.bitrix.search_webhook_adapter import SearchWebhookHandlerAdapter
from .workers.bitrix.supervisor import run_task_supervisor, run_task_supervisor_once
from .workers.bitrix.webhook_event_queue import WebhookEventQueue, run_webhook_event_worker
from .workers.logistics.staff_sync import run_staff_sync


async def _make_portal_search(settings: Settings) -> PortalSearchIndex:
    if settings.database_url:
        index = PostgresPortalSearchIndex(settings.database_url)
        await index.ensure_schema()
        return index  # type: ignore[return-value]
    index = PortalSearchIndex()
    index.ensure_schema()
    return index


def _make_event_queue(settings: Settings) -> WebhookEventQueue | RedisEventQueue:
    if settings.redis_url:
        return RedisEventQueue(settings.redis_url)
    queue = WebhookEventQueue(settings.webhook_event_queue_path, settings=settings)
    queue.ensure_schema()
    return queue


async def _make_vehicle_store(settings: Settings) -> VehicleUsageStorePort:
    if settings.database_url:
        store = PostgresVehicleUsageStore(settings.database_url)
        await store.ensure_schema()
        return store
    store = VehicleUsageStore(settings.vehicle_usage_db_path)
    store.bootstrap_reference_data()
    return store


async def _make_bitrix_store(settings: Settings) -> BitrixAgentStorePort:
    if settings.database_url:
        store = PostgresBitrixAgentStore(settings.database_url)
        await store.ensure_schema()
        return store
    store = BitrixAgentStore()
    store.ensure_schema()
    return store


async def _make_pto_store(settings: Settings) -> Any:
    if not settings.database_url:
        return None
    store = PostgresPtoAgentStore(settings.database_url)
    await store.ensure_schema()
    return store


async def _make_orchestrator_store(settings: Settings) -> Any:
    if not settings.database_url:
        return None
    store = PostgresOrchestratorStore(settings.database_url)
    await store.ensure_schema()
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ensure_runtime_dirs()
    manifests = load_agent_manifests()
    bitrix_oauth = BitrixOAuthService()
    bitrix = BitrixClient(settings=settings, oauth_service=bitrix_oauth)
    bitrix_oauth.ensure_schema()
    portal_search = await _make_portal_search(settings)
    portal_search_indexer = PortalSearchIndexerWorker(bitrix, portal_search, settings=settings)
    learning_recorder = LearningEventRecorder()
    webhook_event_queue = _make_event_queue(settings)
    bitrix_store = await _make_bitrix_store(settings)
    pto_store = await _make_pto_store(settings)
    orchestrator_store = await _make_orchestrator_store(settings)

    app.state.settings = settings
    app.state.manifests = manifests
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
        "dry_run": settings.vehicle_usage_dry_run,
        "dialog_id": settings.vehicle_usage_dialog_id,
        "manager_user_id": settings.vehicle_usage_manager_user_id,
        "admin_notify_user_ids": settings.resolved_vehicle_usage_admin_notify_user_ids,
        "request_time": settings.vehicle_usage_request_time,
        "reminder_interval_minutes": settings.vehicle_usage_reminder_interval_minutes,
        "max_reminders": settings.vehicle_usage_max_reminders,
    }

    async def _reconcile_fn(*, status: dict) -> dict:
        return await reconcile_once(bitrix, webhook_event_queue, portal_search_indexer, status=status)

    async def _supervisor_fn(*, status: dict) -> dict:
        return await run_task_supervisor_once(bitrix, status=status)

    app.state.reconcile_fn = _reconcile_fn
    app.state.supervisor_fn = _supervisor_fn

    scheduler = AgentScheduler()
    scheduler.start()
    app.state.scheduler = scheduler

    webhook_worker_task: asyncio.Task | None = None
    search_indexer_task: asyncio.Task | None = None
    supervisor_task: asyncio.Task | None = None
    reconciler_task: asyncio.Task | None = None
    staff_sync_task: asyncio.Task | None = None
    vehicle_usage_store = None
    if settings.vehicle_usage_enabled:
        vehicle_usage_store = await _make_vehicle_store(settings)

    logistics_llm_svc = LogisticsLLMService()

    if settings.webhook_event_queue_enabled and settings.webhook_event_worker_enabled:
        bitrix_llm_svc = BitrixLLMService(settings=settings)

        vu_settings = (
            VehicleUsageSettings(
                dialog_id=settings.vehicle_usage_dialog_id or "",
                manager_user_id=settings.vehicle_usage_manager_user_id,
                max_reminders=settings.vehicle_usage_max_reminders,
                reminder_interval_minutes=settings.vehicle_usage_reminder_interval_minutes,
                dry_run=settings.vehicle_usage_dry_run,
                request_time=settings.vehicle_usage_request_time,
            )
            if settings.vehicle_usage_enabled
            else None
        )
        specialist_deps = SpecialistDeps(
            settings=settings,
            scheduler=scheduler,
            orchestrator_llm=OrchestratorLLMService(),
            orchestrator_store=orchestrator_store,
            bitrix_llm=bitrix_llm_svc,
            pto_llm=PtoLLMService(),
            pto_store=pto_store,
            logistics_llm=logistics_llm_svc,
            vehicle_usage_store=vehicle_usage_store,
            logistics_vu_settings=vu_settings,
        )
        pending_actions = BitrixPendingActionService(
            store=DialogStateStore(settings.dialog_state_path),
            bitrix=bitrix,
            bitrix_oauth=bitrix_oauth,
            audit_log_path=settings.bitrix_write_audit_log_path,
            dry_run=settings.agent_dry_run,
            settings=settings,
        )
        orchestrator = build_orchestrator(
            manifests,
            specialist_deps,
            bitrix=bitrix,
            portal_search=portal_search,
            pending_actions=pending_actions,
            bot=bitrix,
        )

        if (logistics_spec := orchestrator.specialists.get("logistics")) is not None:
            admin_ids = settings.resolved_vehicle_usage_admin_notify_user_ids

            async def _logistics_to_orchestrator(task):
                enriched = task
                if task.context.get("event") == "vehicle_usage_escalation" and admin_ids:
                    enriched = task.model_copy(update={"context": {**task.context, "notify_user_ids": admin_ids}})
                return await orchestrator.handle(enriched)

            logistics_spec._output_port = _logistics_to_orchestrator
            if settings.vehicle_usage_enabled and not settings.redis_url:
                logistics_spec.start()
            app.state.logistics_specialist = logistics_spec

        search_webhook_handler = SearchWebhookHandlerAdapter(
            bitrix=bitrix,
            index=portal_search,
            settings=settings,
        )
        quality_control_handler = QualityControlHandlerAdapter(
            bitrix=bitrix,
            bitrix_oauth=bitrix_oauth,
            manifests=manifests,
            bitrix_llm=bitrix_llm_svc,
            scheduler=scheduler,
            bitrix_store=bitrix_store,
            settings=settings,
        )
        processor = BitrixWebhookProcessor(
            settings=settings,
            manifests=manifests,
            bitrix=bitrix,
            portal_search=portal_search,
            bitrix_oauth=bitrix_oauth,
            pending_actions=pending_actions,
            orchestrator=orchestrator,
            search_webhook_status=app.state.search_webhook_indexer_status,
            quality_control_status=app.state.quality_control_webhook_status,
            learning_recorder=learning_recorder,
            scheduler=scheduler,
            specialist_deps=specialist_deps,
            bitrix_store=bitrix_store,
            search_webhook_handler=search_webhook_handler,
            quality_control_handler=quality_control_handler,
        )

        async def _dispatch_processor(payload: dict) -> dict:
            return await processor.process(payload)

        webhook_worker_task = asyncio.create_task(
            run_webhook_event_worker(
                webhook_event_queue,
                _dispatch_processor,
                status=app.state.webhook_event_queue_status,
                settings=settings,
            )
        )
    if settings.vehicle_usage_enabled and vehicle_usage_store is not None:
        staff_sync_task = asyncio.create_task(run_staff_sync(bitrix, vehicle_usage_store, settings=settings))
    if settings.search_background_indexer_enabled:
        search_indexer_task = asyncio.create_task(portal_search_indexer.run())
    if settings.supervisor_enabled:
        supervisor_task = asyncio.create_task(
            run_task_supervisor(bitrix, status=app.state.task_supervisor_status, settings=settings)
        )
    if settings.reconcile_enabled:
        reconciler_task = asyncio.create_task(
            run_reconciler(
                bitrix,
                webhook_event_queue,
                portal_search_indexer,
                status=app.state.reconciler_status,
                settings=settings,
            )
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
        if staff_sync_task:
            staff_sync_task.cancel()
            try:
                await staff_sync_task
            except asyncio.CancelledError:
                pass
        if (spec := getattr(app.state, "logistics_specialist", None)) is not None:
            spec.stop()
        scheduler.stop()
