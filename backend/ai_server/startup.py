from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .integrations.bitrix.client import BitrixClient
from .integrations.bitrix.oauth import BitrixOAuthService
from .integrations.postgres.bitrix_agent import PostgresBitrixAgentStore
from .integrations.postgres.vehicle_usage import PostgresVehicleUsageStore
from .integrations.redis.conversation_trace import RedisConversationTrace
from .integrations.redis.event_queue import RedisEventQueue
from .integrations.redis.orchestrator_catalog_health import RedisOrchestratorCatalogHealth
from .integrations.redis.outbound_queue import RedisOutboundQueue
from .registry import load_agent_manifests
from .runtime import ensure_runtime_dirs
from .settings import Settings, get_settings
from .workers.bitrix.reconciler import reconcile_once
from .workers.bitrix.search_indexer import PortalSearchIndexerWorker

logger = logging.getLogger(__name__)


def _make_event_queue(settings: Settings) -> RedisEventQueue:
    return RedisEventQueue(settings.redis_url)


def _make_conversation_trace(settings: Settings) -> RedisConversationTrace:
    return RedisConversationTrace(settings.redis_url, settings=settings)


def _make_outbound_queue(settings: Settings) -> RedisOutboundQueue:
    return RedisOutboundQueue(settings.redis_url)


async def _make_vehicle_store(settings: Settings) -> PostgresVehicleUsageStore:
    store = PostgresVehicleUsageStore(settings.database_url)
    await store.ensure_schema()
    return store


async def _make_bitrix_store(settings: Settings) -> PostgresBitrixAgentStore:
    store = PostgresBitrixAgentStore(settings.database_url)
    await store.ensure_schema()
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required — SQLite support has been removed")
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required — in-memory/SQLite fallbacks have been removed")

    ensure_runtime_dirs()
    manifests = load_agent_manifests()
    bitrix_oauth = BitrixOAuthService(settings=settings)
    bitrix = BitrixClient(settings=settings, oauth_service=bitrix_oauth)
    bitrix_store = await _make_bitrix_store(settings)
    portal_search = bitrix_store
    portal_search_indexer = PortalSearchIndexerWorker(
        bitrix,
        portal_search,
        settings=settings,
        bitrix_oauth=bitrix_oauth,
    )
    webhook_event_queue = _make_event_queue(settings)
    conversation_trace = _make_conversation_trace(settings)
    outbound_queue = _make_outbound_queue(settings)
    orchestrator_catalog_health = RedisOrchestratorCatalogHealth(
        settings.redis_url,
        ttl_seconds=(settings.orchestrator_entity_catalog_refresh_seconds * 2) + 60,
    )

    app.state.settings = settings
    app.state.manifests = manifests
    app.state.bitrix = bitrix
    app.state.bitrix_oauth = bitrix_oauth
    app.state.portal_search = portal_search
    app.state.portal_search_indexer = portal_search_indexer
    app.state.webhook_event_queue = webhook_event_queue
    app.state.conversation_trace = conversation_trace
    app.state.outbound_queue = outbound_queue
    app.state.orchestrator_catalog_health = orchestrator_catalog_health
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
        "admin_user_ids": sorted(settings.resolved_vehicle_usage_admin_user_ids),
        "allowed_user_ids": sorted(settings.resolved_vehicle_usage_allowed_user_ids),
        "staff_sync_enabled": settings.vehicle_usage_staff_sync_enabled,
        "request_time": settings.vehicle_usage_request_time,
        "reminder_interval_minutes": settings.vehicle_usage_reminder_interval_minutes,
        "max_reminders": settings.vehicle_usage_max_reminders,
    }

    async def _reconcile_fn(*, status: dict) -> dict:
        return await reconcile_once(
            bitrix, webhook_event_queue, portal_search_indexer, status=status, settings=settings
        )  # type: ignore[return-value]

    app.state.reconcile_fn = _reconcile_fn

    vehicle_usage_store = None
    if settings.vehicle_usage_enabled:
        vehicle_usage_store = await _make_vehicle_store(settings)

    app.state.vehicle_usage_store = vehicle_usage_store

    try:
        yield
    finally:
        await orchestrator_catalog_health.close()
        await outbound_queue.close()
