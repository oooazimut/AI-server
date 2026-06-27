from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI

from .agent_scheduler import AgentScheduler
from .agents.bitrix24 import BitrixLLMService
from .agents.logistics import LogisticsLLMService
from .agents.logistics.specialist import VehicleUsageSettings
from .agents.pto import PtoLLMService
from .attachments import AttachmentService
from .channels.bitrix import BitrixChatChannel
from .integrations.bitrix.client import BitrixClient
from .integrations.bitrix.oauth import BitrixOAuthService
from .integrations.postgres.bitrix_agent import PostgresBitrixAgentStore
from .integrations.postgres.orchestrator_agent import PostgresOrchestratorStore
from .integrations.postgres.pto_agent import PostgresPtoAgentStore
from .integrations.postgres.vehicle_usage import PostgresVehicleUsageStore
from .integrations.redis.agent_queue import RedisAgentQueue
from .integrations.redis.event_queue import RedisEventQueue
from .learning import LearningEventRecorder
from .models import AgentTask
from .orchestrators.internal import InternalOrchestrator
from .orchestrators.orchestrator_llm import OrchestratorLLMService
from .registry import load_agent_manifests
from .runtime import ensure_runtime_dirs
from .settings import Settings, get_settings
from .specialists import SpecialistDeps
from .technical_footer import TechnicalFooterService
from .tools.vehicle_usage import SentRequestData
from .transcription import build_transcriber
from .utils import MOSCOW_TZ
from .workers.bitrix.reconciler import reconcile_once, run_reconciler
from .workers.bitrix.search_indexer import PortalSearchIndexerWorker
from .workers.bitrix.supervisor import run_task_supervisor, run_task_supervisor_once
from .workers.bitrix.webhook_event_queue import run_webhook_event_worker
from .workers.logistics.staff_sync import run_staff_sync

logger = logging.getLogger(__name__)


def _make_event_queue(settings: Settings) -> RedisEventQueue:
    return RedisEventQueue(settings.redis_url)


def _make_agent_queue(settings: Settings) -> RedisAgentQueue:
    return RedisAgentQueue(settings.redis_url)


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


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 8, 0


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
    portal_search_indexer = PortalSearchIndexerWorker(bitrix, portal_search, settings=settings)
    learning_recorder = LearningEventRecorder()
    webhook_event_queue = _make_event_queue(settings)
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
    agent_worker_tasks: list[asyncio.Task] = []
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
            orchestrator_llm=OrchestratorLLMService(),
            orchestrator_store=orchestrator_store,
            bitrix_llm=bitrix_llm_svc,
            bitrix_store=bitrix_store,
            pto_llm=PtoLLMService(),
            pto_store=pto_store,
            logistics_llm=logistics_llm_svc,
            vehicle_usage_store=vehicle_usage_store,
            logistics_vu_settings=vu_settings,
            channels={"bitrix24": bitrix_channel},
            footer_service=TechnicalFooterService(settings=settings),
            learning_recorder=learning_recorder,
        )
        orch_manifest = next((m for m in manifests if m.kind == "orchestrator"), None)
        orchestrator = InternalOrchestrator.build(
            orch_manifest,
            **specialist_deps.as_build_kwargs(),
        )
        app.state.orchestrator = orchestrator

        agent_queue = _make_agent_queue(settings)
        app.state.agent_queue = agent_queue

        # Morning vehicle-usage cron
        if settings.vehicle_usage_enabled:
            dialog_id = settings.vehicle_usage_dialog_id or ""

            _vu_store_ref = vehicle_usage_store
            _orch_ref = orchestrator
            _mgr_id = settings.vehicle_usage_manager_user_id
            _dry_run = settings.vehicle_usage_dry_run

            async def _run_morning() -> None:
                task = AgentTask(
                    task_id=f"scheduled_vu_{uuid4().hex[:8]}",
                    request="Сгенерируй утренний отчёт по использованию служебных автомобилей.",
                    context={
                        "channel_id": "bitrix24",
                        "recipient_id": dialog_id,
                        "event": "vehicle_usage_morning",
                    },
                )
                result = await _orch_ref.handle(task)
                if result.answer and _vu_store_ref is not None and not _dry_run:
                    _vu_store_ref.create_sent_request(
                        SentRequestData(
                            request_date=datetime.now(MOSCOW_TZ).date().isoformat(),
                            user_id=_mgr_id,
                            dialog_id=dialog_id,
                            message=result.answer,
                            sent_at=datetime.now(MOSCOW_TZ).isoformat(),
                            reminder_count=1,
                        )
                    )

            hour, minute = _parse_hhmm(settings.vehicle_usage_request_time)
            scheduler.add_job_cron(
                "logistics",
                "morning_report",
                _run_morning,
                hour,
                minute,
                replace_existing=False,
            )
            logger.info("Morning vehicle-usage cron scheduled at %02d:%02d МСК", hour, minute)

        # Morning proposals cron: publishes trigger to bitrix24 queue at 08:30
        manager_id = settings.task_proposal_manager_bitrix_id

        _aq_ref = agent_queue

        async def _run_morning_proposals() -> None:
            if not manager_id:
                return
            proposal_task = AgentTask(
                task_id=f"morning_proposals_{uuid4().hex[:8]}",
                request="morning_proposals",
                context={
                    "channel_id": "bitrix24",
                    "recipient_id": str(manager_id),
                },
            )
            await _aq_ref.publish(
                {
                    "to": "bitrix24",
                    "from": "scheduler",
                    "type": "task",
                    "payload": proposal_task.model_dump(),
                    "reply_to": "orchestrator",
                }
            )

        scheduler.add_job_cron(
            "bitrix24",
            "morning_proposals",
            _run_morning_proposals,
            8,
            30,
            replace_existing=False,
        )

        attachment_service = AttachmentService(bitrix)
        transcriber = build_transcriber()

        webhook_worker_task = asyncio.create_task(
            run_webhook_event_worker(
                webhook_event_queue,
                agent_queue=agent_queue,
                attachment_service=attachment_service,
                transcriber=transcriber,
                status=app.state.webhook_event_queue_status,
                settings=settings,
            )
        )

        # Start per-agent run() loops
        agent_worker_tasks = [
            asyncio.create_task(orchestrator.run(agent_queue)),
            *[asyncio.create_task(specialist.run(agent_queue)) for specialist in orchestrator.specialists.values()],
            asyncio.create_task(portal_search_indexer.run(agent_queue)),
        ]
        app.state.search_webhook_indexer_status = portal_search_indexer.event_status

    if settings.vehicle_usage_enabled and vehicle_usage_store is not None:
        staff_sync_task = asyncio.create_task(run_staff_sync(bitrix, vehicle_usage_store, settings=settings))
    if settings.search_background_indexer_enabled:
        search_indexer_task = asyncio.create_task(portal_search_indexer.run_periodic())
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
        for agent_task in agent_worker_tasks:
            agent_task.cancel()
        if agent_worker_tasks:
            await asyncio.gather(*agent_worker_tasks, return_exceptions=True)
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
        scheduler.stop()
