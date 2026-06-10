from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .channels.bitrix import BitrixWebhookProcessor
from .integrations.bitrix.client import BitrixClient
from .integrations.bitrix.oauth import BitrixOAuthService
from .integrations.bitrix.portal_search import PortalSearchIndex
from .learning import LearningEventRecorder
from .registry import load_agent_manifests
from .runtime import ensure_runtime_dirs
from .settings import get_settings
from .workers.bitrix.reconciler import run_reconciler
from .workers.bitrix.search_indexer import PortalSearchIndexerWorker
from .workers.bitrix.supervisor import run_task_supervisor
from .workers.bitrix.webhook_event_queue import WebhookEventQueue, run_webhook_event_worker
from .workers.logistics.vehicle_usage import run_vehicle_usage_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ensure_runtime_dirs()
    manifests = load_agent_manifests()
    bitrix = BitrixClient()
    bitrix_oauth = BitrixOAuthService()
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
            settings=settings,
            manifests=manifests,
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
