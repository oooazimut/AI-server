from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.settings import get_settings
from ai_server.utils import MOSCOW_TZ, optional_int
from ai_server.workers.bitrix.search_indexer import PortalSearchIndexerWorker
from ai_server.workers.bitrix.webhook_event_queue import WebhookEventQueue

logger = logging.getLogger(__name__)


async def run_reconciler(
    bitrix: BitrixClient,
    queue: WebhookEventQueue,
    search_indexer: PortalSearchIndexerWorker,
    *,
    status: dict[str, Any],
) -> None:
    settings = get_settings()
    status.update(
        {
            "enabled": settings.reconcile_enabled,
            "running": True,
            "interval_seconds": settings.reconcile_interval_seconds,
            "task_lookback_hours": settings.reconcile_task_lookback_hours,
            "last_check_at": None,
            "last_success_at": None,
            "last_error": None,
            "next_check_at": None,
            "runs": int(status.get("runs") or 0),
            "errors": int(status.get("errors") or 0),
        }
    )
    if settings.reconcile_initial_delay_seconds:
        await _sleep_until_next(status, settings.reconcile_initial_delay_seconds)

    while True:
        try:
            result = await reconcile_once(bitrix, queue, search_indexer, status=status)
            status["last_success_at"] = _now().isoformat()
            status["last_error"] = None
            status["runs"] = int(status.get("runs") or 0) + 1
            status["last_result"] = result
            await _sleep_until_next(status, get_settings().reconcile_interval_seconds)
        except asyncio.CancelledError:
            status["running"] = False
            raise
        except Exception as exc:
            logger.exception("Reconcile tick failed")
            status["last_error"] = f"{type(exc).__name__}: {exc}"
            status["errors"] = int(status.get("errors") or 0) + 1
            await _sleep_until_next(status, min(get_settings().reconcile_interval_seconds, 300))


async def reconcile_once(
    bitrix: BitrixClient,
    queue: WebhookEventQueue,
    search_indexer: PortalSearchIndexerWorker,
    *,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    now = _now()
    if status is not None:
        status["last_check_at"] = now.isoformat()

    result: dict[str, Any] = {
        "checked_at": now.isoformat(),
        "tasks": {"enabled": settings.reconcile_tasks_enabled},
        "disk_delta": {"enabled": settings.reconcile_disk_delta_enabled},
    }
    if settings.reconcile_tasks_enabled:
        result["tasks"] = await _reconcile_tasks(bitrix, queue, now=now)
    if settings.reconcile_disk_delta_enabled and settings.search_delta_indexer_enabled:
        result["disk_delta"] = await _reconcile_disk_delta(search_indexer)
    elif settings.reconcile_disk_delta_enabled:
        result["disk_delta"] = {"enabled": False, "reason": "search_delta_indexer_disabled"}
    return result


async def _reconcile_tasks(
    bitrix: BitrixClient,
    queue: WebhookEventQueue,
    *,
    now: datetime,
) -> dict[str, Any]:
    settings = get_settings()
    since = now - timedelta(hours=settings.reconcile_task_lookback_hours)
    tasks = await bitrix.list_all_tasks(
        filter_={">=CHANGED_DATE": since.isoformat(timespec="seconds")},
        select=["ID", "TITLE", "STATUS", "RESPONSIBLE_ID", "GROUP_ID", "CHANGED_DATE", "CLOSED_DATE"],
        order={"CHANGED_DATE": "ASC"},
        limit=settings.reconcile_task_limit,
    )
    enqueued = 0
    duplicates = 0
    seen = 0
    for task in tasks:
        task_id = optional_int(_first(task, "id", "ID")) if isinstance(task, dict) else None
        if task_id is None:
            continue
        seen += 1
        changed_date = _optional_str(_first(task, "changedDate", "CHANGED_DATE")) or ""
        payload = {
            "event": "ONTASKUPDATE",
            "data": {
                "FIELDS_AFTER": {
                    "ID": str(task_id),
                    "STATUS": _optional_str(_first(task, "status", "STATUS")) or "",
                    "CHANGED_DATE": changed_date,
                }
            },
            "reconcile": {
                "source": "task_changed_lookback",
                "task_id": task_id,
                "seen_changed_date": changed_date,
                "seen_at": now.isoformat(),
            },
        }
        key = f"reconcile:task:{task_id}:{changed_date or 'unknown'}"
        _, inserted = queue.enqueue(payload, event_type="ONTASKUPDATE", dedupe_key=key)
        if inserted:
            enqueued += 1
        else:
            duplicates += 1
    return {
        "enabled": True,
        "lookback_hours": settings.reconcile_task_lookback_hours,
        "limit": settings.reconcile_task_limit,
        "seen": seen,
        "enqueued": enqueued,
        "duplicates": duplicates,
    }


async def _reconcile_disk_delta(search_indexer: PortalSearchIndexerWorker) -> dict[str, Any]:
    try:
        stats = await search_indexer.run_delta_once()
    except RuntimeError as exc:
        return {"enabled": True, "handled": False, "reason": "locked", "error": str(exc)}
    return {
        "enabled": True,
        "handled": True,
        "folders_scanned": stats.folders_scanned,
        "items_seen": stats.items_seen,
        "items_changed": stats.items_changed,
        "files_changed": stats.files_changed,
        "folders_changed": stats.folders_changed,
        "deleted": stats.deleted,
    }


async def _sleep_until_next(status: dict[str, Any], seconds: int) -> None:
    status["next_check_at"] = (_now() + timedelta(seconds=seconds)).isoformat()
    await asyncio.sleep(seconds)


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _optional_str(value: object) -> str | None:
    return None if value in (None, "") else str(value)
