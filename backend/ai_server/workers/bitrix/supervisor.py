from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ai_server.integrations.bitrix.ports import BitrixSupervisorPort
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ, optional_int

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverdueTask:
    id: int
    title: str
    responsible_id: int | None
    deadline: str | None
    status: str | None


@dataclass(frozen=True)
class OverdueReport:
    tasks: list[OverdueTask]
    user_names: dict[int, str]
    checked_at: datetime

    @property
    def grouped_tasks(self) -> dict[int | None, list[OverdueTask]]:
        grouped: dict[int | None, list[OverdueTask]] = {}
        for task in self.tasks:
            grouped.setdefault(task.responsible_id, []).append(task)
        return grouped


@dataclass(frozen=True)
class SupervisorRunResult:
    checked_at: str
    overdue_tasks_seen: int
    notifications_sent: int
    notifications_planned: int
    notifications: list[dict[str, Any]]


@dataclass(frozen=True)
class SupervisorNotification:
    recipient_id: int
    dry_run: bool
    sent: bool
    task_count: int
    reason: str = ""


async def run_task_supervisor(
    bitrix: BitrixSupervisorPort,
    *,
    status: dict[str, Any],
    settings: Settings,
) -> None:
    status.update(
        {
            "enabled": settings.supervisor_enabled,
            "running": True,
            "dry_run": settings.supervisor_dry_run,
            "interval_seconds": settings.supervisor_interval_seconds,
            "last_check_at": None,
            "last_success_at": None,
            "last_error": None,
            "next_check_at": None,
            "runs": int(status.get("runs") or 0),
            "errors": int(status.get("errors") or 0),
        }
    )
    if settings.supervisor_initial_delay_seconds:
        await _sleep_until_next(status, settings.supervisor_initial_delay_seconds)

    while True:
        try:
            result = await run_task_supervisor_once(bitrix, status=status, settings=settings)
            status["last_success_at"] = _now().isoformat()
            status["last_error"] = None
            status["runs"] = int(status.get("runs") or 0) + 1
            status["last_result"] = asdict(result)
            await _sleep_until_next(status, settings.supervisor_interval_seconds)
        except asyncio.CancelledError:
            status["running"] = False
            raise
        except Exception as exc:
            logger.exception("Task supervisor tick failed")
            status["last_error"] = f"{type(exc).__name__}: {exc}"
            status["errors"] = int(status.get("errors") or 0) + 1
            await _sleep_until_next(status, min(settings.supervisor_interval_seconds, 300))


async def run_task_supervisor_once(
    bitrix: BitrixSupervisorPort,
    *,
    status: dict[str, Any] | None = None,
    settings: Settings,
) -> SupervisorRunResult:
    report = await build_overdue_report(bitrix, limit=settings.supervisor_max_tasks)
    notifications = await send_overdue_notifications(bitrix, report, settings=settings)
    result = SupervisorRunResult(
        checked_at=report.checked_at.isoformat(),
        overdue_tasks_seen=len(report.tasks),
        notifications_sent=sum(1 for item in notifications if item.sent),
        notifications_planned=len(notifications),
        notifications=[asdict(item) for item in notifications],
    )
    if status is not None:
        status.update(asdict(result))
    return result


async def build_overdue_report(bitrix: BitrixSupervisorPort, *, limit: int = 50) -> OverdueReport:
    tasks = [
        task
        for task in _parse_overdue_tasks(
            await bitrix.list_all_tasks(
                filter_={"<DEADLINE": "now", "!STATUS": 5},
                select=["ID", "TITLE", "RESPONSIBLE_ID", "DEADLINE", "STATUS", "CREATED_BY"],
                order={"DEADLINE": "ASC"},
                limit=limit,
            )
        )
        if str(task.status or "") != "4"
    ][:limit]
    user_names = await _resolve_user_names(
        bitrix,
        sorted({task.responsible_id for task in tasks if task.responsible_id is not None}),
    )
    return OverdueReport(tasks=tasks, user_names=user_names, checked_at=_now())


def format_overdue_report(
    report: OverdueReport,
    *,
    settings: Settings,
    task_url_builder: Callable[[object], str] | None = None,
    max_users: int = 20,
    max_tasks_per_user: int | None = None,
) -> str:
    if not report.tasks:
        return "Просроченных активных задач не нашёл."

    grouped_items = sorted(
        report.grouped_tasks.items(),
        key=lambda item: (-len(item[1]), _user_label(item[0], report.user_names)),
    )
    lines = [
        f"Просроченные задачи: {len(report.tasks)} у {len(grouped_items)} исполнителей.",
        f"Проверка: {report.checked_at.strftime('%d.%m.%Y %H:%M')}",
    ]

    for index, (user_id, tasks) in enumerate(grouped_items[:max_users], start=1):
        lines.append("")
        lines.append(f"{index}. {_user_label(user_id, report.user_names)} - {len(tasks)}")
        for task in tasks[: max_tasks_per_user or settings.supervisor_max_tasks_per_user]:
            lines.append(
                "- "
                + _format_task_link(task.id, task.title, task_url_builder=task_url_builder)
                + f" (статус: {_format_task_status(task.status)}; "
                + f"срок: {_format_deadline(task.deadline)}; "
                + f"просрочка: {_format_overdue_age(task.deadline, now=report.checked_at)})"
            )
        remaining = len(tasks) - (max_tasks_per_user or settings.supervisor_max_tasks_per_user)
        if remaining > 0:
            lines.append(f"- ... ещё {remaining}")

    remaining_users = len(grouped_items) - max_users
    if remaining_users > 0:
        lines.extend(["", f"И ещё {remaining_users} исполнителей в отчёт не поместились."])
    return "\n".join(lines)


async def send_overdue_notifications(
    bitrix: BitrixSupervisorPort,
    report: OverdueReport,
    *,
    settings: Settings,
) -> list[SupervisorNotification]:
    state = _load_state(settings.supervisor_state_path)
    task_url_builder = lambda tid: _task_url(tid, settings=settings)  # noqa: E731
    notifications: list[SupervisorNotification] = []

    if settings.resolved_supervisor_admin_user_ids and report.tasks:
        message = "Контроль срока задач: есть незакрытые задачи после дедлайна.\n\n" + format_overdue_report(
            report,
            settings=settings,
            task_url_builder=task_url_builder,
        )
        for admin_user_id in settings.resolved_supervisor_admin_user_ids:
            notifications.append(
                await _send_notification_with_cooldown(
                    bitrix,
                    state,
                    settings=settings,
                    recipient_id=admin_user_id,
                    state_key=f"admin_digest:{admin_user_id}",
                    message=message,
                    task_count=len(report.tasks),
                )
            )

    if settings.supervisor_notify_responsibles:
        for user_id, tasks in report.grouped_tasks.items():
            if not user_id or not tasks:
                continue
            single_user_report = OverdueReport(
                tasks=tasks,
                user_names=report.user_names,
                checked_at=report.checked_at,
            )
            message = "Напоминание по просроченным задачам.\n\n" + format_overdue_report(
                single_user_report,
                settings=settings,
                task_url_builder=task_url_builder,
                max_users=1,
            )
            notifications.append(
                await _send_notification_with_cooldown(
                    bitrix,
                    state,
                    settings=settings,
                    recipient_id=user_id,
                    state_key=f"responsible:{user_id}",
                    message=message,
                    task_count=len(tasks),
                )
            )

    _save_state(settings.supervisor_state_path, state)
    return notifications


async def _send_notification_with_cooldown(
    bitrix: BitrixSupervisorPort,
    state: dict[str, Any],
    *,
    settings: Settings,
    recipient_id: int,
    state_key: str,
    message: str,
    task_count: int,
) -> SupervisorNotification:
    now = _now()
    last_sent_at = _parse_datetime(state.get(state_key))
    cooldown = timedelta(hours=settings.supervisor_reminder_cooldown_hours)
    if last_sent_at and now - last_sent_at < cooldown:
        return SupervisorNotification(recipient_id, settings.supervisor_dry_run, False, task_count, "cooldown")
    if settings.supervisor_dry_run:
        return SupervisorNotification(recipient_id, True, False, task_count, "dry_run")

    await bitrix.notify_user(
        user_id=recipient_id,
        message=message,
        tag="bitrix_ai_agent_overdue_tasks",
        sub_tag=state_key,
    )
    state[state_key] = now.isoformat()
    return SupervisorNotification(recipient_id, False, True, task_count)


def _parse_overdue_tasks(raw_tasks: object) -> list[OverdueTask]:
    tasks: list[OverdueTask] = []
    for raw in raw_tasks if isinstance(raw_tasks, list) else []:
        if not isinstance(raw, dict):
            continue
        task_id = _first(raw, "id", "ID")
        if task_id is None:
            continue
        tasks.append(
            OverdueTask(
                id=int(task_id),
                title=str(_first(raw, "title", "TITLE") or "Без названия"),
                responsible_id=optional_int(_first(raw, "responsibleId", "RESPONSIBLE_ID")),
                deadline=_optional_str(_first(raw, "deadline", "DEADLINE")),
                status=_optional_str(_first(raw, "status", "STATUS")),
            )
        )
    return tasks


async def _resolve_user_names(bitrix: BitrixSupervisorPort, user_ids: list[int]) -> dict[int, str]:
    names: dict[int, str] = {}
    for user_id in user_ids:
        try:
            user = await bitrix.get_user(user_id)
        except Exception:
            logger.exception("Failed to resolve overdue task responsible name")
            continue
        if user:
            names[user_id] = _format_user_name(user)
    return names


def _format_user_name(user: dict[str, Any]) -> str:
    name = " ".join(
        part
        for part in (
            str(user.get("NAME") or user.get("name") or "").strip(),
            str(user.get("LAST_NAME") or user.get("lastName") or "").strip(),
        )
        if part
    )
    return name or f"#{user.get('ID') or user.get('id')}"


def _user_label(user_id: int | None, user_names: dict[int, str]) -> str:
    if user_id is None:
        return "Без исполнителя"
    return f"{user_names.get(user_id) or f'#{user_id}'} (#{user_id})"


def _format_task_link(task_id: object, title: str, *, task_url_builder: Callable[[object], str] | None) -> str:
    url = task_url_builder(task_id) if task_url_builder else ""
    return f"[{title}]({url})" if url else f"{title} (#{task_id})"


def _format_task_status(status: object) -> str:
    return {
        "1": "новая",
        "2": "ждёт выполнения",
        "3": "в работе",
        "4": "ждёт контроля",
        "5": "завершена",
        "6": "отложена",
    }.get(str(status or ""), str(status or "неизвестно"))


def _format_deadline(value: str | None) -> str:
    parsed = _parse_datetime(value)
    return parsed.astimezone(MOSCOW_TZ).strftime("%d.%m.%Y %H:%M") if parsed else "не указан"


def _format_overdue_age(deadline: str | None, *, now: datetime) -> str:
    parsed = _parse_datetime(deadline)
    if not parsed:
        return "неизвестно"
    delta = now - parsed.astimezone(MOSCOW_TZ)
    if delta.days > 0:
        return f"{delta.days} дн."
    hours = max(1, int(delta.total_seconds() // 3600))
    return f"{hours} ч."


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=MOSCOW_TZ)


def _task_url(task_id: object, *, settings: Settings) -> str:
    domain = settings.bitrix_domain.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if not domain:
        return f"/company/personal/user/0/tasks/task/view/{task_id}/"
    return f"https://{domain}/company/personal/user/0/tasks/task/view/{task_id}/"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load supervisor state")
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


async def _sleep_until_next(status: dict[str, Any], seconds: int) -> None:
    status["next_check_at"] = (_now() + timedelta(seconds=seconds)).isoformat()
    await asyncio.sleep(seconds)


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _optional_str(value: object) -> str | None:
    return None if value in (None, "") else str(value)
