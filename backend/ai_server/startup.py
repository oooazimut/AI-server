from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .agent_scheduler import AgentScheduler
from .agents.logistics import LogisticsSpecialist
from .agents.logistics_llm import LogisticsLLMService
from .channels.bitrix import BitrixWebhookProcessor
from .integrations.bitrix.client import BitrixClient
from .integrations.bitrix.oauth import BitrixOAuthService
from .integrations.bitrix.portal_search import PortalSearchIndex
from .learning import LearningEventRecorder
from .registry import load_agent_manifests
from .runtime import ensure_runtime_dirs
from .settings import get_settings
from .specialists import manifest_by_id
from .tools.vehicle_usage import VehicleUsageStore, VehicleUsageToolset
from .workers.bitrix.reconciler import run_reconciler
from .workers.bitrix.search_indexer import PortalSearchIndexerWorker
from .workers.bitrix.supervisor import run_task_supervisor
from .workers.bitrix.webhook_event_queue import WebhookEventQueue, run_webhook_event_worker
from .workers.logistics.staff_sync import run_staff_sync


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ensure_runtime_dirs()
    manifests = load_agent_manifests()
    bitrix_oauth = BitrixOAuthService()
    bitrix = BitrixClient(settings=settings, oauth_service=bitrix_oauth)
    bitrix_oauth.ensure_schema()
    portal_search = PortalSearchIndex()
    portal_search.ensure_schema()
    portal_search_indexer = PortalSearchIndexerWorker(bitrix, portal_search)
    learning_recorder = LearningEventRecorder()
    webhook_event_queue = WebhookEventQueue(settings.webhook_event_queue_path)
    webhook_event_queue.ensure_schema()

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

    scheduler = AgentScheduler()
    scheduler.start()
    app.state.scheduler = scheduler

    webhook_worker_task: asyncio.Task | None = None
    search_indexer_task: asyncio.Task | None = None
    supervisor_task: asyncio.Task | None = None
    reconciler_task: asyncio.Task | None = None
    staff_sync_task: asyncio.Task | None = None
    logistics_specialist: LogisticsSpecialist | None = None

    if settings.webhook_event_queue_enabled and settings.webhook_event_worker_enabled:
        processor = BitrixWebhookProcessor(
            settings=settings,
            manifests=manifests,
            bitrix=bitrix,
            portal_search=portal_search,
            bitrix_oauth=bitrix_oauth,
            search_webhook_status=app.state.search_webhook_indexer_status,
            quality_control_status=app.state.quality_control_webhook_status,
            learning_recorder=learning_recorder,
            scheduler=scheduler,
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
        supervisor_task = asyncio.create_task(run_task_supervisor(bitrix, status=app.state.task_supervisor_status))
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
        vehicle_usage_store = VehicleUsageStore()
        vehicle_usage_store.bootstrap_reference_data()

        async def _vehicle_deliver(dialog_id: str, message: str) -> None:
            await bitrix.send_bot_message(dialog_id, message)

        async def _vehicle_notify(user_id: int, message: str) -> None:
            await bitrix.notify_user(user_id=user_id, message=message, tag="vehicle_usage_escalation")

        logistics_manifest = manifest_by_id(manifests, "logistics")
        if logistics_manifest is not None:
            logistics_specialist = LogisticsSpecialist(
                logistics_manifest,
                llm=LogisticsLLMService(),
                scheduler=scheduler,
                deliver_fn=_vehicle_deliver,
                notify_fn=_vehicle_notify,
                settings=settings,
                tools=VehicleUsageToolset(
                    store=vehicle_usage_store,
                    user_id=settings.vehicle_usage_manager_user_id,
                    dialog_id=settings.vehicle_usage_dialog_id,
                ),
            )
            logistics_specialist.start()
            app.state.logistics_specialist = logistics_specialist

        staff_sync_task = asyncio.create_task(run_staff_sync(bitrix, vehicle_usage_store))

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
        if logistics_specialist is not None:
            logistics_specialist.stop()
        scheduler.stop()
