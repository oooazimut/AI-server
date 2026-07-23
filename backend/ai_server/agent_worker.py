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
from datetime import datetime, timedelta

from ai_server.agent_scheduler import AgentScheduler
from ai_server.agents.logistics.tools.vehicle_save import DEFAULT_START_MESSAGE
from ai_server.attachments import AttachmentService
from ai_server.channels.bitrix import BitrixChatChannel
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.integrations.postgres.bitrix_agent import PostgresBitrixAgentStore
from ai_server.integrations.postgres.diagnost_agent import PostgresDiagnostStore
from ai_server.integrations.postgres.orchestrator_agent import PostgresOrchestratorStore
from ai_server.integrations.postgres.vehicle_usage import PostgresVehicleUsageStore
from ai_server.integrations.redis.agent_queue import RedisAgentQueue
from ai_server.integrations.redis.conversation_trace import RedisConversationTrace
from ai_server.integrations.redis.diagnost_queue import RedisDiagnostQueue
from ai_server.integrations.redis.dialog_guard import RedisDialogGuard
from ai_server.integrations.redis.event_queue import RedisEventQueue
from ai_server.integrations.redis.orchestrator_catalog_health import RedisOrchestratorCatalogHealth
from ai_server.integrations.redis.outbound_queue import RedisOutboundQueue
from ai_server.llm import build_orchestrator_llm_client
from ai_server.orchestrators.bitrix_formatter import format_task_close_report, format_task_close_result_text
from ai_server.orchestrators.draft_confirmation import draft_confirmation_phrase, matches_draft_confirmation
from ai_server.orchestrators.entity_catalog import OrchestratorEntityCatalog
from ai_server.orchestrators.plan_authoritative import DeepSeekPlanService, PlanAuthoritativeOrchestrator
from ai_server.registry import load_agent_manifests
from ai_server.runtime import ensure_runtime_dirs
from ai_server.settings import get_settings
from ai_server.specialists import SpecialistDeps
from ai_server.technical_footer import TechnicalFooterService
from ai_server.tools.vehicle_usage import SentRequestData, resolve_vehicle_usage_operator_ids
from ai_server.transcription import build_transcriber
from ai_server.utils import MOSCOW_TZ
from ai_server.workers.bitrix.reconciler import run_reconciler
from ai_server.workers.bitrix.search_indexer import PortalSearchIndexerWorker
from ai_server.workers.bitrix.staff_roster_publisher import publish_staff_roster
from ai_server.workers.bitrix.task_close_direct_dispatcher import run_task_close_direct_control_worker
from ai_server.workers.bitrix.webhook_event_queue import run_webhook_event_worker
from ai_server.workers.diagnost.event_worker import run_diagnost_event_worker
from ai_server.workers.logistics.staff_sync import run_staff_sync
from ai_server.workers.orchestrator.outbound_delivery import run_outbound_delivery_worker
from ai_server.workers.orchestrator.result_publisher import OrchestratorResultPublisher, SpecialistResultPublisher

logger = logging.getLogger(__name__)


async def _run_entity_catalog_refresh(
    entity_catalog: OrchestratorEntityCatalog,
    catalog_health: RedisOrchestratorCatalogHealth,
    *,
    refresh_interval_seconds: int,
) -> None:
    while True:
        await asyncio.sleep(max(60, int(refresh_interval_seconds)))
        snapshot = await entity_catalog.refresh()
        await catalog_health.publish(snapshot)


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

    orchestrator_store = PostgresOrchestratorStore(settings.database_url)
    await orchestrator_store.ensure_schema()

    diagnost_store = PostgresDiagnostStore(settings.database_url)
    await diagnost_store.ensure_schema()
    portal_search_indexer = PortalSearchIndexerWorker(
        bitrix,
        portal_search,
        settings=settings,
        bitrix_oauth=bitrix_oauth,
    )
    diagnost_queue = RedisDiagnostQueue(settings.redis_url)
    conversation_trace = RedisConversationTrace(settings.redis_url, settings=settings)
    outbound_queue = RedisOutboundQueue(settings.redis_url)
    dialog_guard = RedisDialogGuard(settings.redis_url, settings=settings)
    result_publisher = OrchestratorResultPublisher(diagnost_queue, conversation_trace=conversation_trace)
    specialist_result_publisher = SpecialistResultPublisher(diagnost_queue, conversation_trace=conversation_trace)
    webhook_event_queue = RedisEventQueue(settings.redis_url)
    catalog_health = RedisOrchestratorCatalogHealth(
        settings.redis_url,
        ttl_seconds=(settings.orchestrator_entity_catalog_refresh_seconds * 2) + 60,
    )

    vehicle_usage_store = None
    if settings.vehicle_usage_enabled:
        vehicle_usage_store = PostgresVehicleUsageStore(settings.database_url)
        await vehicle_usage_store.ensure_schema()

    scheduler = AgentScheduler()
    if settings.scheduler_enabled:
        scheduler.start()
    else:
        logger.info("AgentScheduler disabled by AI_SERVER_SCHEDULER_ENABLED=false")

    agent_tasks: list[asyncio.Task] = []
    bitrix_channel = BitrixChatChannel(settings=settings, bitrix=bitrix)
    # The outbox is an independent durable subsystem: it must drain an existing
    # backlog even when inbound webhook processing is intentionally disabled.
    agent_tasks.append(
        asyncio.create_task(
            run_outbound_delivery_worker(
                outbound_queue,
                channels={"bitrix24": bitrix_channel},
                conversation_trace=conversation_trace,
                incident_queue=diagnost_queue,
            )
        )
    )

    orchestrator: PlanAuthoritativeOrchestrator | None = None
    if settings.webhook_event_queue_enabled and settings.webhook_event_worker_enabled:
        entity_catalog = OrchestratorEntityCatalog(
            bitrix,
            refresh_interval_seconds=settings.orchestrator_entity_catalog_refresh_seconds,
            user_limit=settings.orchestrator_entity_catalog_user_limit,
            project_limit=settings.orchestrator_entity_catalog_project_limit,
            warehouse_limit=settings.orchestrator_entity_catalog_warehouse_limit,
        )
        await catalog_health.publish(await entity_catalog.refresh())

        specialist_deps = SpecialistDeps(
            settings=settings,
            manifests=manifests,
            bitrix_client=bitrix,
            portal_search_index=portal_search,
            bitrix_oauth=bitrix_oauth,
            bitrix_bot=bitrix,
            scheduler=scheduler,
            orchestrator_llm=DeepSeekPlanService(build_orchestrator_llm_client(settings)),
            orchestrator_store=orchestrator_store,
            orchestrator_entity_catalog=entity_catalog,
            task_close_report_renderer=format_task_close_report,
            task_close_result_text_renderer=format_task_close_result_text,
            draft_confirmation_phrase_renderer=draft_confirmation_phrase,
            draft_confirmation_matcher=matches_draft_confirmation,
            bitrix_store=bitrix_store,
            specialist_result_publisher=specialist_result_publisher,
            vehicle_usage_store=vehicle_usage_store,
            channels={"bitrix24": bitrix_channel},
            footer_service=TechnicalFooterService(settings=settings),
            conversation_trace=conversation_trace,
            dialog_guard=dialog_guard,
            outbound_queue=outbound_queue,
            result_publisher=result_publisher,
        )
        orch_manifest = next((m for m in manifests if m.kind == "orchestrator"), None)
        orchestrator = PlanAuthoritativeOrchestrator.build(
            orch_manifest,
            **specialist_deps.as_build_kwargs(),
        )
        agent_tasks.append(
            asyncio.create_task(
                _run_entity_catalog_refresh(
                    entity_catalog,
                    catalog_health,
                    refresh_interval_seconds=settings.orchestrator_entity_catalog_refresh_seconds,
                )
            )
        )

        agent_queue = RedisAgentQueue(
            settings.redis_url,
            processing_ttl_seconds=settings.agent_queue_processing_ttl_seconds,
        )

        if settings.scheduler_enabled and settings.vehicle_usage_enabled:
            _vu_store_ref = vehicle_usage_store
            _fallback_operator_ids = frozenset(settings.resolved_vehicle_usage_allowed_user_ids)
            _dry_run = settings.vehicle_usage_dry_run

            def _vehicle_usage_reminder_run_date(started_at: datetime, reminder_count: int) -> datetime:
                delays = settings.resolved_vehicle_usage_reminder_delays_minutes or (
                    settings.vehicle_usage_reminder_interval_minutes,
                )
                index = max(0, min(reminder_count - 1, len(delays) - 1))
                return started_at + timedelta(minutes=delays[index])

            def _vehicle_usage_message(reminder_count: int) -> str:
                if reminder_count <= 0:
                    return DEFAULT_START_MESSAGE
                return (
                    f"Напоминание #{reminder_count}. Пожалуйста, отправьте отчет по машинам и людям за сегодня: "
                    "кто работает, кто выходной/болеет/в отпуске, кто на какой машине, "
                    "и какие машины свободны, на ремонте или не работают."
                )

            def _vehicle_usage_request_is_closed(started_at: datetime, operator_id: int) -> bool:
                if _vu_store_ref is None:
                    return False
                request = _vu_store_ref.get_request(
                    request_date=started_at.date().isoformat(),
                    user_id=operator_id,
                )
                status = str((request or {}).get("status") or "").strip()
                return status in {"answered", "cancelled_day_off"}

            def _schedule_vehicle_usage_reminder(started_at: datetime, operator_id: int, reminder_count: int) -> None:
                if reminder_count > settings.vehicle_usage_max_reminders:
                    return
                run_date = _vehicle_usage_reminder_run_date(started_at, reminder_count)
                job_id = f"vu_reminder_{started_at.date().isoformat()}_{operator_id}"

                async def _run_reminder() -> None:
                    if _vehicle_usage_request_is_closed(started_at, operator_id):
                        return
                    await _send_vehicle_usage_message(started_at, operator_id, reminder_count=reminder_count)
                    _schedule_vehicle_usage_reminder(started_at, operator_id, reminder_count + 1)

                scheduler.add_job_at("logistics", job_id, _run_reminder, run_date, replace_existing=True)

            async def _send_vehicle_usage_message(
                started_at: datetime,
                operator_id: int,
                *,
                reminder_count: int,
            ) -> None:
                recipient_id = str(operator_id)
                message = _vehicle_usage_message(reminder_count)
                await bitrix_channel.send(recipient_id, message)
                if _vu_store_ref is not None and not _dry_run:
                    _vu_store_ref.create_sent_request(
                        SentRequestData(
                            request_date=started_at.date().isoformat(),
                            user_id=operator_id,
                            dialog_id=recipient_id,
                            message=message,
                            sent_at=datetime.now(MOSCOW_TZ).isoformat(),
                            reminder_count=reminder_count,
                            source="scheduled" if reminder_count <= 0 else "scheduled_reminder",
                        )
                    )

            async def _run_morning_for_operator(started_at: datetime, operator_id: int) -> None:
                await _send_vehicle_usage_message(started_at, operator_id, reminder_count=0)
                _schedule_vehicle_usage_reminder(started_at, operator_id, reminder_count=1)

            async def _run_morning() -> None:
                started_at = datetime.now(MOSCOW_TZ)
                operator_ids = resolve_vehicle_usage_operator_ids(_vu_store_ref, _fallback_operator_ids)
                if not operator_ids:
                    logger.warning("Morning vehicle-usage skipped: no configured operators")
                    return
                for operator_id in operator_ids:
                    await _run_morning_for_operator(started_at, operator_id)

            async def _finalize_pending_unknowns() -> None:
                report_date = datetime.now(MOSCOW_TZ).date().isoformat()
                if _vu_store_ref is None:
                    logger.warning("Vehicle-usage unknown finalization skipped: store is not configured")
                    return
                if _dry_run:
                    logger.info("Vehicle-usage unknown finalization dry-run skipped for %s", report_date)
                    return
                result = _vu_store_ref.finalize_pending_unknowns(
                    report_date=report_date,
                    reason="Auto-filled missing vehicle usage data as unknown.",
                )
                logger.info("Vehicle-usage unknown finalization result: %s", result)

            async def _auto_close_unanswered_day() -> None:
                report_date = datetime.now(MOSCOW_TZ).date().isoformat()
                if _vu_store_ref is None:
                    logger.warning("Vehicle-usage day-off auto close skipped: store is not configured")
                    return
                if _dry_run:
                    logger.info("Vehicle-usage day-off auto close dry-run skipped for %s", report_date)
                    return
                result = _vu_store_ref.auto_close_unanswered_day(
                    report_date=report_date,
                    reason="Auto-closed as day off because no useful vehicle usage response was received by cutoff.",
                )
                logger.info("Vehicle-usage day-off auto close result: %s", result)

            hour, minute = _parse_hhmm(settings.vehicle_usage_request_time)
            unknown_hour, unknown_minute = _parse_hhmm(settings.vehicle_usage_unknown_fill_time)
            day_off_hour, day_off_minute = _parse_hhmm(settings.vehicle_usage_auto_day_off_time)
            vehicle_usage_day_of_week = "mon-fri" if settings.vehicle_usage_workday_mode == "weekday" else None
            scheduler.add_job_cron(
                "logistics",
                "morning_report",
                _run_morning,
                hour,
                minute,
                day_of_week=vehicle_usage_day_of_week,
                replace_existing=False,
            )
            logger.info("Morning vehicle-usage cron scheduled at %02d:%02d МСК", hour, minute)

            scheduler.add_job_cron(
                "logistics",
                "vehicle_usage_unknown_finalize",
                _finalize_pending_unknowns,
                unknown_hour,
                unknown_minute,
                day_of_week=vehicle_usage_day_of_week,
                replace_existing=False,
            )
            logger.info(
                "Vehicle-usage unknown finalization cron scheduled at %02d:%02d MSK",
                unknown_hour,
                unknown_minute,
            )
            scheduler.add_job_cron(
                "logistics",
                "vehicle_usage_auto_day_off",
                _auto_close_unanswered_day,
                day_off_hour,
                day_off_minute,
                day_of_week=vehicle_usage_day_of_week,
                replace_existing=False,
            )
            logger.info(
                "Vehicle-usage day-off auto close cron scheduled at %02d:%02d MSK",
                day_off_hour,
                day_off_minute,
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
                    conversation_trace=conversation_trace,
                    dialog_guard=dialog_guard,
                    bitrix_sender=bitrix,
                    orchestrator_store=orchestrator_store,
                )
            )
        )
        agent_task_timeout = settings.agent_task_timeout_seconds
        orchestrator_worker_count = max(1, settings.agent_orchestrator_worker_count)
        bitrix_worker_count = max(1, settings.agent_bitrix_worker_count)
        for index in range(orchestrator_worker_count):
            agent_tasks.append(
                asyncio.create_task(
                    orchestrator.run(
                        agent_queue,
                        worker_name=f"orchestrator-{index + 1}",
                        task_timeout_seconds=agent_task_timeout,
                    )
                )
            )
        for sp in orchestrator.specialists.values():
            specialist_id = str(getattr(getattr(sp, "manifest", None), "id", ""))
            if specialist_id == "bitrix24":
                worker_count = bitrix_worker_count
            else:
                worker_count = 1
            for index in range(worker_count):
                agent_tasks.append(
                    asyncio.create_task(
                        sp.run(  # type: ignore[union-attr]
                            agent_queue,
                            worker_name=f"{specialist_id or 'specialist'}-{index + 1}",
                            task_timeout_seconds=agent_task_timeout,
                        )
                    )
                )
        agent_tasks.append(asyncio.create_task(portal_search_indexer.run(agent_queue)))
        if settings.diagnost_enabled:
            agent_tasks.append(
                asyncio.create_task(
                    run_diagnost_event_worker(
                        diagnost_queue,
                        diagnost_store,
                        conversation_trace=conversation_trace,
                        trace_snapshot_enabled=settings.diagnost_trace_snapshot_enabled,
                        trace_settle_seconds=settings.diagnost_trace_settle_seconds,
                        high_latency_ms=settings.diagnost_high_latency_ms,
                    )
                )
            )
        else:
            logger.info("Diagnost workers disabled by DIAGNOST_ENABLED=false")

    if settings.vehicle_usage_enabled and settings.vehicle_usage_staff_sync_enabled and vehicle_usage_store is not None:
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
    if settings.search_background_periodic_enabled:
        agent_tasks.append(asyncio.create_task(portal_search_indexer.run_periodic()))
    if settings.bitrix_task_close_control_worker_enabled and orchestrator is not None:
        _task_close_direct_status: dict = {
            "enabled": settings.bitrix_task_close_control_worker_enabled,
            "running": False,
            "interval_seconds": settings.bitrix_task_close_control_interval_seconds,
            "last_check_at": None,
            "last_success_at": None,
            "last_error": None,
            "next_check_at": None,
            "runs": 0,
            "errors": 0,
        }
        agent_tasks.append(
            asyncio.create_task(
                run_task_close_direct_control_worker(
                    store=portal_search,
                    settings=settings,
                    status=_task_close_direct_status,
                    orchestrator_handler=orchestrator.handle,
                )
            )
        )
    elif settings.bitrix_task_close_control_worker_enabled:
        logger.error(
            "Task-close control worker disabled: the authoritative orchestrator is unavailable"
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
    await catalog_health.close()
    await outbound_queue.close()
    logger.info("Agent worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
