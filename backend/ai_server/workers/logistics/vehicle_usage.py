from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time
from typing import Any
from uuid import uuid4

from ai_server.agents.logistics import LogisticsSpecialist
from ai_server.agents.logistics_llm import LogisticsAgentLLM
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.models import AgentTask, UserContext
from ai_server.registry import get_agent_manifest
from ai_server.settings import get_settings
from ai_server.tools.vehicle_usage import MOSCOW_TZ, VehicleUsageStore, VehicleUsageToolset


logger = logging.getLogger(__name__)


async def run_vehicle_usage_worker(
    bitrix: BitrixClient,
    *,
    status: dict[str, Any],
) -> None:
    settings = get_settings()
    status.update(_initial_status())
    status["running"] = True

    try:
        while True:
            try:
                await run_vehicle_usage_once(bitrix, status=status)
                status["last_error"] = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Vehicle usage worker tick failed")
                status["last_error"] = f"{type(exc).__name__}: {exc}"
                status["errors"] = int(status.get("errors") or 0) + 1
            await asyncio.sleep(settings.vehicle_usage_interval_seconds)
    finally:
        status["running"] = False


async def run_vehicle_usage_once(
    bitrix: BitrixClient,
    *,
    status: dict[str, Any],
    now: datetime | None = None,
    store: VehicleUsageStore | None = None,
    logistics_llm: LogisticsAgentLLM | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    selected_store = store or VehicleUsageStore()
    selected_store.bootstrap_reference_data()
    now = (now or datetime.now(MOSCOW_TZ)).astimezone(MOSCOW_TZ)
    status.update(_status_config(selected_store))
    status["last_check_at"] = now.isoformat()
    status["runs"] = int(status.get("runs") or 0) + 1

    if not settings.vehicle_usage_enabled:
        return _skipped(status, "disabled")
    manager_id = settings.vehicle_usage_manager_user_id
    if manager_id is None:
        return _skipped(status, "manager_user_id_missing")
    dialog_id = _vehicle_usage_dialog_id()
    if not dialog_id:
        return _skipped(status, "dialog_id_missing")
    if not _is_working_date(now.date()):
        status["last_skipped_date"] = now.date().isoformat()
        return _skipped(status, "non_working_day")

    request_date = now.date().isoformat()
    existing = selected_store.get_request(request_date=request_date, user_id=manager_id)
    if existing and str(existing.get("status") or "") == "sent" and not existing.get("escalated_at"):
        escalation_time = _parse_time(settings.vehicle_usage_escalation_time)
        if now >= _at_time(now, escalation_time):
            result = await _run_logistics_event(
                bitrix,
                store=selected_store,
                logistics_llm=logistics_llm,
                manager_id=manager_id,
                dialog_id=dialog_id,
                request=(
                    "Scheduler event: к времени эскалации не получен утренний отчет по "
                    f"сотрудникам и служебным автомобилям за {request_date}. "
                    "Сформулируй уведомление администраторам и вызови vehicle_usage_notify_admins. "
                    "Перед этим прочитай контекст через vehicle_usage_context."
                ),
                context={
                    "event": "vehicle_usage_escalation_due",
                    "request_date": request_date,
                    "admin_user_ids": settings.resolved_vehicle_usage_admin_notify_user_ids,
                    "existing_request": existing,
                },
            )
            status["last_escalated_at"] = now.isoformat()
            status["last_result"] = result.model_dump()
            return {"handled": True, "action": "escalation", "result": result.model_dump()}

    reminder_number = _due_reminder_number(now)
    if reminder_number is None:
        return _skipped(status, "no_due_reminder")
    if existing and str(existing.get("status") or "") != "sent":
        return _skipped(status, "request_already_in_progress")
    if existing and _optional_int(existing.get("reminder_count")) >= reminder_number:
        return _skipped(status, "reminder_already_sent")

    result = await _run_logistics_event(
        bitrix,
        store=selected_store,
        logistics_llm=logistics_llm,
        manager_id=manager_id,
        dialog_id=dialog_id,
        request=(
            "Scheduler event: пора отправить утренний запрос по сотрудникам и служебным автомобилям "
            f"за {request_date}. Номер напоминания: {reminder_number}. "
            "Сформулируй сообщение, отправь его в Bitrix через vehicle_usage_send_message, "
            "затем отметь отправку через vehicle_usage_mark_request_sent. "
            "Перед отправкой прочитай roster/vehicles/latest_request через vehicle_usage_context."
        ),
        context={
            "event": "vehicle_usage_reminder_due",
            "request_date": request_date,
            "reminder_number": reminder_number,
            "dialog_id": dialog_id,
            "existing_request": existing,
        },
    )
    status["last_sent_at"] = now.isoformat()
    status["last_request_date"] = request_date
    status["last_reminder_number"] = reminder_number
    status["last_result"] = result.model_dump()
    return {"handled": True, "action": "reminder", "result": result.model_dump()}


def _initial_status() -> dict[str, Any]:
    return {"running": False, **_status_config(VehicleUsageStore()), "runs": 0, "errors": 0, "last_error": None}


def _status_config(store: VehicleUsageStore) -> dict[str, Any]:
    settings = get_settings()
    return {
        "enabled": settings.vehicle_usage_enabled,
        "dry_run": settings.vehicle_usage_dry_run,
        "interval_seconds": settings.vehicle_usage_interval_seconds,
        "dialog_id": settings.vehicle_usage_dialog_id,
        "manager_user_id": settings.vehicle_usage_manager_user_id,
        "admin_notify_user_ids": settings.resolved_vehicle_usage_admin_notify_user_ids,
        "request_time": settings.vehicle_usage_request_time,
        "request_times": settings.vehicle_usage_request_times,
        "escalation_time": settings.vehicle_usage_escalation_time,
        "db_path": str(store.path),
    }


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


def _skipped(status: dict[str, Any], reason: str) -> dict[str, Any]:
    status["last_skip_reason"] = reason
    return {"handled": False, "reason": reason}


def _due_reminder_number(now: datetime) -> int | None:
    settings = get_settings()
    if now >= _at_time(now, _parse_time(settings.vehicle_usage_escalation_time)):
        return None
    due = 0
    for item in _request_times():
        if now >= _at_time(now, item):
            due += 1
    return due or None


def _request_times() -> list[time]:
    settings = get_settings()
    raw = settings.vehicle_usage_request_times or settings.vehicle_usage_request_time
    times = [_parse_time(item) for item in raw.replace(";", ",").split(",") if item.strip()]
    return sorted(times or [_parse_time(settings.vehicle_usage_request_time)])


def _parse_time(value: str) -> time:
    try:
        hour, minute = value.strip().split(":", 1)
        return time(hour=int(hour), minute=int(minute))
    except (TypeError, ValueError):
        return time(hour=9, minute=0)


def _at_time(now: datetime, selected_time: time) -> datetime:
    return now.replace(hour=selected_time.hour, minute=selected_time.minute, second=0, microsecond=0)


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


def _vehicle_usage_dialog_id() -> str:
    settings = get_settings()
    configured = settings.vehicle_usage_dialog_id.strip()
    if configured:
        return configured
    manager_id = settings.vehicle_usage_manager_user_id
    return str(manager_id) if manager_id else ""


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
