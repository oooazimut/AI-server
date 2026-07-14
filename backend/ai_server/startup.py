from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .agent_scheduler import AgentScheduler
from .agents.bitrix24 import BitrixLLMService
from .agents.kartoteka import KartotekaLLMService
from .agents.logistics import LogisticsLLMService
from .agents.logistics.specialist import VehicleUsageSettings
from .agents.pto import PtoLLMService
from .channels.bitrix import BitrixChatChannel
from .integrations.bitrix.client import BitrixClient
from .integrations.bitrix.oauth import BitrixOAuthService
from .integrations.postgres.bitrix_agent import PostgresBitrixAgentStore
from .integrations.postgres.kartoteka_agent import PostgresKartotekaStore
from .integrations.postgres.orchestrator_agent import PostgresOrchestratorStore
from .integrations.postgres.pto_agent import PostgresPtoAgentStore
from .integrations.postgres.vehicle_usage import PostgresVehicleUsageStore
from .integrations.redis.agent_queue import RedisAgentQueue
from .integrations.redis.conversation_trace import RedisConversationTrace
from .integrations.redis.dialog_guard import RedisDialogGuard
from .integrations.redis.event_queue import RedisEventQueue
from .llm import build_orchestrator_llm_client
from .orchestrators.internal import InternalOrchestrator
from .orchestrators.orchestrator_llm import OrchestratorLLMService
from .registry import load_agent_manifests
from .runtime import ensure_runtime_dirs
from .settings import Settings, get_settings
from .specialists import SpecialistDeps
from .technical_footer import TechnicalFooterService
from .workers.bitrix.reconciler import reconcile_once
from .workers.bitrix.search_indexer import PortalSearchIndexerWorker
from .workers.bitrix.supervisor import run_task_supervisor_once

logger = logging.getLogger(__name__)


def _make_event_queue(settings: Settings) -> RedisEventQueue:
    return RedisEventQueue(settings.redis_url)


def _make_agent_queue(settings: Settings) -> RedisAgentQueue:
    return RedisAgentQueue(settings.redis_url)


def _make_conversation_trace(settings: Settings) -> RedisConversationTrace:
    return RedisConversationTrace(settings.redis_url, settings=settings)


def _make_dialog_guard(settings: Settings) -> RedisDialogGuard:
    return RedisDialogGuard(settings.redis_url, settings=settings)


async def _make_vehicle_store(settings: Settings) -> PostgresVehicleUsageStore:
    store = PostgresVehicleUsageStore(settings.database_url)
    await store.ensure_schema()
    return store


async def _make_bitrix_store(settings: Settings) -> PostgresBitrixAgentStore:
    store = PostgresBitrixAgentStore(settings.database_url)
    await store.ensure_schema()
    return store


async def _make_pto_store(settings: Settings) -> PostgresPtoAgentStore:
    store = PostgresPtoAgentStore(settings.database_url)
    await store.ensure_schema()
    return store


async def _make_orchestrator_store(settings: Settings) -> PostgresOrchestratorStore:
    store = PostgresOrchestratorStore(settings.database_url)
    await store.ensure_schema()
    return store


async def _make_kartoteka_store(settings: Settings) -> PostgresKartotekaStore:
    store = PostgresKartotekaStore(
        settings.database_url,
        protected_user_ids=settings.kartoteka_protected_user_ids,
        secret_user_ids=settings.kartoteka_secret_user_ids,
    )
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
    dialog_guard = _make_dialog_guard(settings)
    pto_store = await _make_pto_store(settings)
    orchestrator_store = await _make_orchestrator_store(settings)
    kartoteka_store = await _make_kartoteka_store(settings)

    app.state.settings = settings
    app.state.manifests = manifests
    app.state.bitrix = bitrix
    app.state.bitrix_oauth = bitrix_oauth
    app.state.portal_search = portal_search
    app.state.portal_search_indexer = portal_search_indexer
    app.state.webhook_event_queue = webhook_event_queue
    app.state.conversation_trace = conversation_trace
    app.state.dialog_guard = dialog_guard
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

    async def _supervisor_fn(*, status: dict) -> dict:
        return await run_task_supervisor_once(bitrix, status=status, settings=settings)  # type: ignore[return-value]

    app.state.reconcile_fn = _reconcile_fn
    app.state.supervisor_fn = _supervisor_fn

    scheduler = AgentScheduler()
    scheduler.start()
    app.state.scheduler = scheduler

    vehicle_usage_store = None
    if settings.vehicle_usage_enabled:
        vehicle_usage_store = await _make_vehicle_store(settings)

    app.state.vehicle_usage_store = vehicle_usage_store

    logistics_llm_svc = LogisticsLLMService()

    if settings.webhook_event_queue_enabled and settings.webhook_event_worker_enabled:
        bitrix_llm_svc = BitrixLLMService(settings=settings)

        vu_settings = (
            VehicleUsageSettings(
                manager_user_id=settings.vehicle_usage_manager_user_id,
                max_reminders=settings.vehicle_usage_max_reminders,
                reminder_interval_minutes=settings.vehicle_usage_reminder_interval_minutes,
                allowed_user_ids=frozenset(settings.resolved_vehicle_usage_allowed_user_ids),
                admin_user_ids=frozenset(settings.resolved_vehicle_usage_admin_user_ids),
                dry_run=settings.vehicle_usage_dry_run,
                request_time=settings.vehicle_usage_request_time,
            )
            if settings.vehicle_usage_enabled
            else None
        )

        bitrix_channel = BitrixChatChannel(settings=settings, bitrix=bitrix)
        app.state.bitrix_channel = bitrix_channel

        specialist_deps = SpecialistDeps(
            settings=settings,
            manifests=manifests,
            bitrix_client=bitrix,
            portal_search_index=portal_search,
            bitrix_oauth=bitrix_oauth,
            bitrix_bot=bitrix,
            scheduler=scheduler,
            orchestrator_llm=OrchestratorLLMService(build_orchestrator_llm_client(settings)),
            orchestrator_store=orchestrator_store,
            bitrix_llm=bitrix_llm_svc,
            bitrix_store=bitrix_store,
            pto_llm=PtoLLMService(),
            pto_store=pto_store,
            logistics_llm=logistics_llm_svc,
            vehicle_usage_store=vehicle_usage_store,
            logistics_vu_settings=vu_settings,
            kartoteka_store=kartoteka_store,
            kartoteka_llm=KartotekaLLMService(),
            channels={"bitrix24": bitrix_channel},
            footer_service=TechnicalFooterService(settings=settings),
            conversation_trace=conversation_trace,
            dialog_guard=dialog_guard,
        )
        orch_manifest = next((m for m in manifests if m.kind == "orchestrator"), None)
        orchestrator = InternalOrchestrator.build(
            orch_manifest,
            **specialist_deps.as_build_kwargs(),
        )
        app.state.orchestrator = orchestrator

        agent_queue = _make_agent_queue(settings)
        app.state.agent_queue = agent_queue

        app.state.search_webhook_indexer_status = portal_search_indexer.event_status

    try:
        yield
    finally:
        scheduler.stop()
