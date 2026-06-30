"""Agent worker process.

Run as a separate systemd unit so consumer loops fire exactly once
even when uvicorn runs with --workers N.

Usage:
    python -m ai_server.agent_worker
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime
from uuid import uuid4

from ai_server.agent_scheduler import AgentScheduler
from ai_server.agents.bitrix24 import BitrixLLMService
from ai_server.agents.kartoteka.llm import KartotekaLLMService
from ai_server.agents.logistics import LogisticsLLMService
from ai_server.agents.logistics.specialist import VehicleUsageSettings
from ai_server.agents.pto import PtoLLMService
from ai_server.attachments import AttachmentService
from ai_server.channels.bitrix import BitrixChatChannel
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.integrations.postgres.bitrix_agent import PostgresBitrixAgentStore
from ai_server.integrations.postgres.kartoteka_agent import PostgresKartotekaStore
from ai_server.integrations.postgres.orchestrator_agent import PostgresOrchestratorStore
from ai_server.integrations.postgres.pto_agent import PostgresPtoAgentStore
from ai_server.integrations.postgres.vehicle_usage import PostgresVehicleUsageStore
from ai_server.integrations.redis.agent_queue import RedisAgentQueue
from ai_server.integrations.redis.event_queue import RedisEventQueue
from ai_server.learning import LearningEventRecorder
from ai_server.llm import build_orchestrator_llm_client
from ai_server.models import AgentTask
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.orchestrators.orchestrator_llm import OrchestratorLLMService
from ai_server.registry import load_agent_manifests
from ai_server.runtime import ensure_runtime_dirs
from ai_server.settings import get_settings
from ai_server.specialists import SpecialistDeps
from ai_server.technical_footer import TechnicalFooterService
from ai_server.tools.vehicle_usage import SentRequestData
from ai_server.transcription import build_transcriber
from ai_server.utils import MOSCOW_TZ
from ai_server.workers.bitrix.reconciler import run_reconciler
from ai_server.workers.bitrix.search_indexer import PortalSearchIndexerWorker
from ai_server.workers.bitrix.staff_roster_publisher import publish_staff_roster
from ai_server.workers.bitrix.supervisor import run_task_supervisor
from ai_server.workers.bitrix.webhook_event_queue import run_webhook_event_worker
from ai_server.workers.logistics.staff_sync import run_staff_sync

logger = logging.getLogger(__name__)


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = value.strip().split(":")
    try:
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return 8, 0


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    settings = get_settings()

    if not settings.database_url:
        logger.error("DATABASE_URL is required — agent_worker requires PostgreSQL. Exiting.")
        sys.exit(1)
    if not settings.redis_url:
        logger.error("REDIS_URL is required — agent_worker requires Redis. Exiting.")
        sys.exit(1)

    ensure_runtime_dirs()
    manifests = load_agent_manifests()
    bitrix_oauth = BitrixOAuthService(settings=settings)
    bitrix = BitrixClient(settings=settings, oauth_service=bitrix_oauth)

    bitrix_store = PostgresBitrixAgentStore(settings.database_url)
    await bitrix_store.ensure_schema()
    portal_search = bitrix_store

    pto_store = PostgresPtoAgentStore(settings.database_url)
    await pto_store.ensure_schema()

    kartoteka_store = PostgresKartotekaStore(
        settings.database_url,
        protected_user_ids=settings.kartoteka_protected_user_ids,
        secret_user_ids=settings.kartoteka_secret_user_ids,
    )
    await kartoteka_store.ensure_schema()

    orchestrator_store = PostgresOrchestratorStore(settings.database_url)
    await orchestrator_store.ensure_schema()

    portal_search_indexer = PortalSearchIndexerWorker(bitrix, portal_search, settings=settings)
    learning_recorder = LearningEventRecorder()
    webhook_event_queue = RedisEventQueue(settings.redis_url)

    vehicle_usage_store = None
    if settings.vehicle_usage_enabled:
        vehicle_usage_store = PostgresVehicleUsageStore(settings.database_url)
        await vehicle_usage_store.ensure_schema()

    scheduler = AgentScheduler()
    scheduler.start()

    agent_tasks: list[asyncio.Task] = []

    if settings.webhook_event_queue_enabled and settings.webhook_event_worker_enabled:
        bitrix_llm_svc = BitrixLLMService(settings=settings)
        logistics_llm_svc = LogisticsLLMService()

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
            kartoteka_llm=KartotekaLLMService(),
            kartoteka_store=kartoteka_store,
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

        agent_queue = RedisAgentQueue(settings.redis_url)

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

        _webhook_status: dict = {
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
        agent_tasks.append(
            asyncio.create_task(
                run_webhook_event_worker(
                    webhook_event_queue,
                    agent_queue=agent_queue,
                    attachment_service=attachment_service,
                    transcriber=transcriber,
                    status=_webhook_status,
                    settings=settings,
                )
            )
        )
        agent_tasks.append(asyncio.create_task(orchestrator.run(agent_queue)))
        for sp in orchestrator.specialists.values():
            agent_tasks.append(asyncio.create_task(sp.run(agent_queue)))  # type: ignore[union-attr]
        agent_tasks.append(asyncio.create_task(portal_search_indexer.run(agent_queue)))

    if settings.vehicle_usage_enabled and vehicle_usage_store is not None:
        _bitrix_ref = bitrix
        _redis_url = settings.redis_url

        async def _publish_roster() -> None:
            await publish_staff_roster(_bitrix_ref, _redis_url, settings=settings)

        scheduler.add_job_cron(
            "bitrix",
            "staff_roster_sync",
            _publish_roster,
            3,
            0,
            day_of_week="tue",
            replace_existing=False,
        )
        agent_tasks.append(asyncio.create_task(_publish_roster()))
        agent_tasks.append(asyncio.create_task(run_staff_sync(vehicle_usage_store, settings.redis_url)))
    if settings.search_background_indexer_enabled:
        agent_tasks.append(asyncio.create_task(portal_search_indexer.run_periodic()))
    if settings.supervisor_enabled:
        _supervisor_status: dict = {
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
        agent_tasks.append(
            asyncio.create_task(run_task_supervisor(bitrix, status=_supervisor_status, settings=settings))
        )
    if settings.reconcile_enabled:
        _reconciler_status: dict = {
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
        agent_tasks.append(
            asyncio.create_task(
                run_reconciler(
                    bitrix,
                    webhook_event_queue,
                    portal_search_indexer,
                    status=_reconciler_status,
                    settings=settings,
                )
            )
        )

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _on_signal(*_: object) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    logger.info("Agent worker started — %d background task(s)", len(agent_tasks))
    await stop.wait()
    logger.info("Agent worker stopping...")
    for t in agent_tasks:
        t.cancel()
    await asyncio.gather(*agent_tasks, return_exceptions=True)
    scheduler.stop()
    logger.info("Agent worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
