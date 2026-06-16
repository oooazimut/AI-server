from __future__ import annotations

import logging
from datetime import date, datetime, time
from typing import Any
from uuid import uuid4

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from ai_server.agent_scheduler import AgentScheduler
from ai_server.agents.logistics import LogisticsSpecialist
from ai_server.agents.logistics_llm import LogisticsAgentLLM
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.models import AgentTask, UserContext
from ai_server.registry import get_agent_manifest
from ai_server.settings import Settings, get_settings
from ai_server.tools.vehicle_usage import MOSCOW_TZ, VehicleUsageStore, VehicleUsageToolset, fetch_staff_roster
from ai_server.utils import optional_int

logger = logging.getLogger(__name__)


class VehicleUsageWorker:
    """
    Manages the daily morning vehicle-usage schedule for the logistics specialist.

    Creates one-shot DateTrigger jobs each working day:
      - one job per request time (e.g. 08:30, 09:00, 09:30)
      - one escalation job (e.g. 10:00)

    A midnight CronTrigger recreates jobs for the next day.
    When a confirmed report is saved the specialist cancels remaining jobs via
    ``cancel_daily_jobs(date)``.
    """

    def __init__(
        self,
        scheduler: AgentScheduler,
        bitrix: BitrixClient,
        *,
        settings: Settings | None = None,
        logistics_llm: LogisticsAgentLLM | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.bitrix = bitrix
        self._settings = settings or get_settings()
        self._logistics_llm = logistics_llm

    def setup_today(self, *, today: date | None = None) -> dict[str, Any]:
        """Create jobs for today (or the given date). No-op on non-working days."""
        settings = self._settings
        today = today or datetime.now(MOSCOW_TZ).date()

        if not _is_working_date(today):
            logger.debug("VehicleUsageWorker: %s is not a working day, skipping schedule", today)
            return {"scheduled": False, "reason": "non_working_day", "date": today.isoformat()}

        manager_id = settings.vehicle_usage_manager_user_id
        dialog_id = _vehicle_usage_dialog_id(settings)
        if manager_id is None or not dialog_id:
            return {"scheduled": False, "reason": "manager_or_dialog_not_configured", "date": today.isoformat()}

        date_str = today.isoformat()
        store = VehicleUsageStore()
        jobs_added: list[str] = []

        for i, t in enumerate(_request_times(settings), start=1):
            run_at = _at_datetime(today, t)
            if run_at <= datetime.now(MOSCOW_TZ):
                existing = store.get_request(request_date=date_str, user_id=manager_id)
                if existing:
                    continue  # already sent/answered
            job_id = f"morning_{i}_{date_str}"
            reminder_num = i
            self.scheduler.add_job(
                "logistics",
                job_id,
                self._make_reminder_handler(
                    store=store,
                    manager_id=manager_id,
                    dialog_id=dialog_id,
                    date_str=date_str,
                    reminder_number=reminder_num,
                ),
                DateTrigger(run_date=run_at, timezone=MOSCOW_TZ),
            )
            jobs_added.append(job_id)

        esc_time = _parse_time(settings.vehicle_usage_escalation_time)
        esc_run_at = _at_datetime(today, esc_time)
        if esc_run_at > datetime.now(MOSCOW_TZ):
            esc_job_id = f"escalation_{date_str}"
            self.scheduler.add_job(
                "logistics",
                esc_job_id,
                self._make_escalation_handler(
                    store=store,
                    manager_id=manager_id,
                    date_str=date_str,
                ),
                DateTrigger(run_date=esc_run_at, timezone=MOSCOW_TZ),
            )
            jobs_added.append(esc_job_id)

        logger.info("VehicleUsageWorker: scheduled %d jobs for %s", len(jobs_added), date_str)
        return {"scheduled": True, "date": date_str, "jobs": jobs_added}

    def cancel_daily_jobs(self, date_str: str) -> int:
        """Cancel all remaining morning/escalation jobs for the given date."""
        removed = self.scheduler.remove_jobs_by_prefix("logistics", "morning_")
        removed += self.scheduler.remove_jobs_by_prefix("logistics", "escalation_")
        logger.info("VehicleUsageWorker: cancelled %d jobs after report saved for %s", removed, date_str)
        return removed

    def setup_midnight_cron(self) -> None:
        """Register a nightly CronTrigger that recreates jobs for the next day."""
        self.scheduler.add_job(
            "logistics",
            "midnight_setup",
            self._make_midnight_handler(),
            CronTrigger(hour=0, minute=1, timezone=MOSCOW_TZ),
        )

    # ------------------------------------------------------------------
    # Internal job factories — closures capture all deps at schedule time
    # ------------------------------------------------------------------

    def _make_reminder_handler(
        self,
        *,
        store: VehicleUsageStore,
        manager_id: int,
        dialog_id: str,
        date_str: str,
        reminder_number: int,
    ):
        bitrix = self.bitrix
        llm = self._logistics_llm
        settings = self._settings

        async def handler() -> None:
            existing = store.get_request(request_date=date_str, user_id=manager_id)
            if existing and str(existing.get("status") or "") != "sent":
                logger.debug("Reminder %d skipped: request already %s", reminder_number, existing.get("status"))
                return
            if existing and optional_int(existing.get("reminder_count")) >= reminder_number:
                logger.debug("Reminder %d already sent", reminder_number)
                return

            bot_id = settings.bitrix_bot_id
            exclude = {bot_id} if bot_id else None
            try:
                roster = await fetch_staff_roster(bitrix, exclude_user_ids=exclude)
            except Exception:
                logger.warning("Failed to fetch staff roster from Bitrix, falling back to settings")
                roster = None
            store.bootstrap_reference_data(roster)

            result = await _run_logistics_event(
                bitrix,
                store=store,
                logistics_llm=llm,
                manager_id=manager_id,
                dialog_id=dialog_id,
                request=(
                    f"Scheduler event: пора отправить утренний запрос по сотрудникам и служебным автомобилям "
                    f"за {date_str}. Номер напоминания: {reminder_number}. "
                    "Сформулируй точный текст сообщения для Переговорщика. "
                    "Перед этим прочитай roster/vehicles/latest_request через vehicle_usage_context. "
                    "Сам не отправляй сообщения в Bitrix."
                ),
                context={
                    "event": "vehicle_usage_reminder_due",
                    "request_date": date_str,
                    "reminder_number": reminder_number,
                    "dialog_id": dialog_id,
                    "existing_request": existing,
                },
            )
            await _deliver_reminder(
                bitrix,
                store=store,
                request_date=date_str,
                manager_id=manager_id,
                dialog_id=dialog_id,
                message=result.answer,
                reminder_number=reminder_number,
                now=datetime.now(MOSCOW_TZ),
                settings=settings,
            )

        return handler

    def _make_escalation_handler(
        self,
        *,
        store: VehicleUsageStore,
        manager_id: int,
        date_str: str,
    ):
        bitrix = self.bitrix
        llm = self._logistics_llm
        settings = self._settings

        async def handler() -> None:
            existing = store.get_request(request_date=date_str, user_id=manager_id)
            if not existing or str(existing.get("status") or "") != "sent":
                return
            if existing.get("escalated_at"):
                return

            result = await _run_logistics_event(
                bitrix,
                store=store,
                logistics_llm=llm,
                manager_id=manager_id,
                dialog_id=_vehicle_usage_dialog_id(settings),
                request=(
                    f"Scheduler event: к времени эскалации не получен утренний отчет по "
                    f"сотрудникам и служебным автомобилям за {date_str}. "
                    "Сформулируй точный текст уведомления для Переговорщика. "
                    "Перед этим прочитай контекст через vehicle_usage_context. "
                    "Сам не отправляй сообщения в Bitrix."
                ),
                context={
                    "event": "vehicle_usage_escalation_due",
                    "request_date": date_str,
                    "admin_user_ids": settings.resolved_vehicle_usage_admin_notify_user_ids,
                    "existing_request": existing,
                },
            )
            await _deliver_escalation(
                bitrix,
                store=store,
                request_date=date_str,
                manager_id=manager_id,
                message=result.answer,
                now=datetime.now(MOSCOW_TZ),
                settings=settings,
            )

        return handler

    def _make_midnight_handler(self):
        async def handler() -> None:
            self.setup_today()

        return handler


# ---------------------------------------------------------------------------
# Kept for backwards-compat with existing tests — thin wrapper
# ---------------------------------------------------------------------------


async def run_vehicle_usage_once(
    bitrix: BitrixClient,
    *,
    status: dict[str, Any],
    now: datetime | None = None,
    store: VehicleUsageStore | None = None,
    logistics_llm: LogisticsAgentLLM | None = None,
) -> dict[str, Any]:
    """Single-tick helper retained for unit tests. Not used in production."""
    settings = get_settings()
    selected_store = store or VehicleUsageStore()

    bot_id = settings.bitrix_bot_id
    exclude = {bot_id} if bot_id else None
    try:
        roster = await fetch_staff_roster(bitrix, exclude_user_ids=exclude)
    except Exception:
        logger.warning("Failed to fetch staff roster from Bitrix, falling back to settings")
        roster = None
    selected_store.bootstrap_reference_data(roster)

    now = (now or datetime.now(MOSCOW_TZ)).astimezone(MOSCOW_TZ)
    status["last_check_at"] = now.isoformat()
    status["runs"] = int(status.get("runs") or 0) + 1

    if not settings.vehicle_usage_enabled:
        return _skipped(status, "disabled")
    manager_id = settings.vehicle_usage_manager_user_id
    if manager_id is None:
        return _skipped(status, "manager_user_id_missing")
    dialog_id = _vehicle_usage_dialog_id(settings)
    if not dialog_id:
        return _skipped(status, "dialog_id_missing")
    if not _is_working_date(now.date()):
        return _skipped(status, "non_working_day")

    request_date = now.date().isoformat()
    existing = selected_store.get_request(request_date=request_date, user_id=manager_id)

    if existing and str(existing.get("status") or "") == "sent" and not existing.get("escalated_at"):
        escalation_time = _parse_time(settings.vehicle_usage_escalation_time)
        if now >= _at_datetime(now.date(), escalation_time):
            result = await _run_logistics_event(
                bitrix,
                store=selected_store,
                logistics_llm=logistics_llm,
                manager_id=manager_id,
                dialog_id=dialog_id,
                request=(
                    f"Scheduler event: к времени эскалации не получен утренний отчет по "
                    f"сотрудникам и служебным автомобилям за {request_date}. "
                    "Сформулируй точный текст уведомления для Переговорщика. "
                    "Перед этим прочитай контекст через vehicle_usage_context. "
                    "Сам не отправляй сообщения в Bitrix."
                ),
                context={
                    "event": "vehicle_usage_escalation_due",
                    "request_date": request_date,
                    "admin_user_ids": settings.resolved_vehicle_usage_admin_notify_user_ids,
                    "existing_request": existing,
                },
            )
            delivery = await _deliver_escalation(
                bitrix,
                store=selected_store,
                request_date=request_date,
                manager_id=manager_id,
                message=result.answer,
                now=now,
                settings=settings,
            )
            status["last_escalated_at"] = now.isoformat()
            return {"handled": True, "action": "escalation", "result": result.model_dump(), "delivery": delivery}

    reminder_number = _due_reminder_number(now, settings)
    if reminder_number is None:
        return _skipped(status, "no_due_reminder")
    if existing and str(existing.get("status") or "") != "sent":
        return _skipped(status, "request_already_in_progress")
    if existing and optional_int(existing.get("reminder_count")) >= reminder_number:
        return _skipped(status, "reminder_already_sent")

    result = await _run_logistics_event(
        bitrix,
        store=selected_store,
        logistics_llm=logistics_llm,
        manager_id=manager_id,
        dialog_id=dialog_id,
        request=(
            f"Scheduler event: пора отправить утренний запрос по сотрудникам и служебным автомобилям "
            f"за {request_date}. Номер напоминания: {reminder_number}. "
            "Сформулируй точный текст сообщения для Переговорщика. "
            "Перед этим прочитай roster/vehicles/latest_request через vehicle_usage_context. "
            "Сам не отправляй сообщения в Bitrix."
        ),
        context={
            "event": "vehicle_usage_reminder_due",
            "request_date": request_date,
            "reminder_number": reminder_number,
            "dialog_id": dialog_id,
            "existing_request": existing,
        },
    )
    delivery = await _deliver_reminder(
        bitrix,
        store=selected_store,
        request_date=request_date,
        manager_id=manager_id,
        dialog_id=dialog_id,
        message=result.answer,
        reminder_number=reminder_number,
        now=now,
        settings=settings,
    )
    status["last_sent_at"] = now.isoformat()
    status["last_reminder_number"] = reminder_number
    return {"handled": True, "action": "reminder", "result": result.model_dump(), "delivery": delivery}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _run_logistics_event(
    bitrix: BitrixClient,
    *,
    store: VehicleUsageStore,
    logistics_llm: LogisticsAgentLLM | None,
    manager_id: int,
    dialog_id: str,
    request: str,
    context: dict[str, Any],
) -> Any:
    manifest = get_agent_manifest("logistics")
    if manifest is None:
        raise RuntimeError("logistics manifest is missing")
    return await LogisticsSpecialist(
        manifest,
        tools=VehicleUsageToolset(
            client=bitrix,
            store=store,
            user_id=manager_id,
            dialog_id=dialog_id,
        ),
        llm=logistics_llm,
    ).handle(
        AgentTask(
            task_id=str(uuid4()),
            source="scheduler",
            user=UserContext(id=str(manager_id), channel="scheduler", raw={"dialog_id": dialog_id}),
            request=request,
            context=context,
        )
    )


async def _deliver_reminder(
    bitrix: BitrixClient,
    *,
    store: VehicleUsageStore,
    request_date: str,
    manager_id: int,
    dialog_id: str,
    message: str,
    reminder_number: int,
    now: datetime,
    settings: Settings,
) -> dict[str, Any]:
    cleaned = message.strip()
    if not cleaned:
        return {"sent": False, "reason": "empty_message"}
    sent = False
    if not settings.vehicle_usage_dry_run:
        await bitrix.send_bot_message(dialog_id, cleaned)
        sent = True
    request_id = store.create_sent_request(
        request_date=request_date,
        user_id=manager_id,
        dialog_id=dialog_id,
        message=cleaned,
        sent_at=now.isoformat(),
        reminder_count=reminder_number,
    )
    return {
        "sent": sent,
        "dry_run": settings.vehicle_usage_dry_run,
        "dialog_id": dialog_id,
        "request_id": request_id,
        "message": cleaned,
        "speaker": "negotiator_channel",
    }


async def _deliver_escalation(
    bitrix: BitrixClient,
    *,
    store: VehicleUsageStore,
    request_date: str,
    manager_id: int,
    message: str,
    now: datetime,
    settings: Settings,
) -> dict[str, Any]:
    cleaned = message.strip()
    if not cleaned:
        return {"sent": False, "reason": "empty_message"}
    user_ids = settings.resolved_vehicle_usage_admin_notify_user_ids
    if not user_ids:
        marked = store.mark_escalated(request_date=request_date, user_id=manager_id, escalated_at=now.isoformat())
        return {"sent": False, "marked": marked, "reason": "no_admin_user_ids", "speaker": "negotiator_channel"}
    notified: list[int] = []
    if not settings.vehicle_usage_dry_run:
        for uid in user_ids:
            await bitrix.notify_user(user_id=uid, message=cleaned, tag=f"vehicle_usage_no_response_{request_date}")
            notified.append(uid)
    marked = store.mark_escalated(request_date=request_date, user_id=manager_id, escalated_at=now.isoformat())
    return {
        "sent": bool(notified),
        "dry_run": settings.vehicle_usage_dry_run,
        "notified_user_ids": notified if notified else user_ids,
        "marked": marked,
        "message": cleaned,
        "speaker": "negotiator_channel",
    }


def _skipped(status: dict[str, Any], reason: str) -> dict[str, Any]:
    status["last_skip_reason"] = reason
    return {"handled": False, "reason": reason}


def _due_reminder_number(now: datetime, settings: Settings) -> int | None:
    if now >= _at_datetime(now.date(), _parse_time(settings.vehicle_usage_escalation_time)):
        return None
    due = 0
    for t in _request_times(settings):
        if now >= _at_datetime(now.date(), t):
            due += 1
    return due or None


def _request_times(settings: Settings) -> list[time]:
    raw = settings.vehicle_usage_request_times or settings.vehicle_usage_request_time
    times = [_parse_time(item) for item in raw.replace(";", ",").split(",") if item.strip()]
    return sorted(times or [_parse_time(settings.vehicle_usage_request_time)])


def _parse_time(value: str) -> time:
    try:
        hour, minute = value.strip().split(":", 1)
        return time(hour=int(hour), minute=int(minute))
    except (TypeError, ValueError):
        return time(hour=9, minute=0)


def _at_datetime(d: date, t: time) -> datetime:
    return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=MOSCOW_TZ)


def _is_working_date(value: date) -> bool:
    settings = get_settings()
    if value in _date_set(settings.agent_working_dates):
        return True
    if value in _date_set(settings.agent_non_working_dates):
        return False
    return value.weekday() < 5


def _date_set(raw_dates: str) -> set[date]:
    result: set[date] = set()
    for raw_date in raw_dates.replace(";", ",").split(","):
        cleaned = raw_date.strip()
        if not cleaned:
            continue
        try:
            result.add(date.fromisoformat(cleaned))
        except ValueError:
            continue
    return result


def _vehicle_usage_dialog_id(settings: Settings) -> str:
    configured = settings.vehicle_usage_dialog_id.strip()
    if configured:
        return configured
    manager_id = settings.vehicle_usage_manager_user_id
    return str(manager_id) if manager_id else ""
