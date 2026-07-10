from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any

from ai_server.agents.ports import AgentQueuePort
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search import (
    PortalContentSyncStats,
    PortalDeltaSyncStats,
    PortalSearchIndex,
    PortalSyncStats,
    format_portal_content_sync_stats,
    format_portal_delta_sync_stats,
    format_portal_sync_stats,
    sync_disk_delta_index,
    sync_portal_content_index,
    sync_portal_index,
)
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ
from ai_server.workers.bitrix.search_webhook_indexer import prepare_search_webhook_job, process_search_webhook_job

logger = logging.getLogger(__name__)

PERSISTED_STATUS_KEYS = {
    "last_error",
    "last_error_at",
    "consecutive_errors",
    "last_success_at",
    "content_runs",
    "metadata_runs",
    "delta_runs",
    "last_content_sync_at",
    "last_metadata_sync_at",
    "last_delta_sync_at",
    "last_content_started_at",
    "last_metadata_started_at",
    "last_delta_started_at",
    "last_content_duration_seconds",
    "last_metadata_duration_seconds",
    "last_delta_duration_seconds",
    "last_content_summary",
    "last_metadata_summary",
    "last_delta_summary",
    "last_content_pending_after",
    "last_metadata_prune_skipped",
    "last_metadata_stale_deleted",
    "last_metadata_total",
    "last_delta_folders_scanned",
    "last_delta_items_seen",
    "last_delta_items_changed",
    "last_delta_files_changed",
    "last_delta_folders_changed",
    "last_delta_deleted",
    "delta_folder_cursor_type",
    "delta_folder_cursor_id",
}


class PortalSearchIndexerWorker:
    def __init__(self, bitrix: BitrixClient, index: PortalSearchIndex, *, settings: Settings) -> None:
        self.bitrix = bitrix
        self.index = index
        self._settings = settings
        self._lock = asyncio.Lock()
        self._owns_lock = False
        self.status: dict[str, Any] = {
            "running": False,
            "enabled": self._periodic_enabled(),
            "metadata_enabled": self._metadata_enabled(),
            "content_enabled": self._content_enabled(),
            "delta_enabled": self._delta_enabled(),
            "last_error": None,
            "last_error_at": None,
            "consecutive_errors": 0,
            "last_success_at": None,
            "metadata_running": False,
            "content_running": False,
            "delta_running": False,
            "metadata_runs": 0,
            "content_runs": 0,
            "delta_runs": 0,
            "last_metadata_sync_at": None,
            "last_content_sync_at": None,
            "last_delta_sync_at": None,
            "last_metadata_started_at": None,
            "last_content_started_at": None,
            "last_delta_started_at": None,
            "last_metadata_duration_seconds": None,
            "last_content_duration_seconds": None,
            "last_delta_duration_seconds": None,
            "next_metadata_sync_at": None,
            "next_content_sync_at": None,
            "next_delta_sync_at": None,
            "last_metadata_summary": None,
            "last_content_summary": None,
            "last_delta_summary": None,
            "last_metadata_prune_skipped": [],
            "last_metadata_stale_deleted": 0,
            "last_metadata_total": 0,
            "last_content_pending_after": None,
            "last_delta_folders_scanned": 0,
            "last_delta_items_seen": 0,
            "last_delta_items_changed": 0,
            "last_delta_files_changed": 0,
            "last_delta_folders_changed": 0,
            "last_delta_deleted": 0,
            "delta_folder_cursor_type": None,
            "delta_folder_cursor_id": None,
            "lock_acquired": False,
            "lock_path": str(settings.search_background_lock_path),
            "lock_owner": None,
            "lock_last_heartbeat_at": None,
            "lock_retry_at": None,
        }
        self._load_state()
        self.event_status: dict[str, Any] = {
            "running": False,
            "processed": 0,
            "errors": 0,
            "last_event": None,
            "last_error": None,
        }

    def public_status(self) -> dict[str, Any]:
        return {
            **self.status,
            "enabled": self._periodic_enabled(),
            "metadata_enabled": self._metadata_enabled(),
            "content_enabled": self._content_enabled(),
            "delta_enabled": self._delta_enabled(),
            "lock_path": str(self._settings.search_background_lock_path),
            "state_path": str(self._settings.search_background_state_path),
        }

    async def run(self, queue: AgentQueuePort) -> None:
        """Event-driven loop: consume disk-file events from the agent queue."""
        _poll_interval = 0.5
        self.event_status["running"] = True
        while True:
            message = await queue.claim_next("index_refresher")
            if message is None:
                await asyncio.sleep(_poll_interval)
                continue
            msg_id = str(message.get("id") or "")
            payload = message.get("payload") or {}
            try:
                job, result = prepare_search_webhook_job(payload, settings=self._settings)
                if job:
                    await process_search_webhook_job(
                        self.bitrix,
                        self.index,
                        job,
                        status=self.event_status,
                        settings=self._settings,
                    )
                    self.event_status["processed"] = int(self.event_status.get("processed") or 0) + 1
                else:
                    self.event_status["last_event"] = result.get("event")
                await queue.ack(msg_id)
            except asyncio.CancelledError:
                self.event_status["running"] = False
                raise
            except Exception as exc:
                logger.exception("PortalSearchIndexerWorker: failed processing event message %s", msg_id)
                self.event_status["errors"] = int(self.event_status.get("errors") or 0) + 1
                self.event_status["last_error"] = f"{type(exc).__name__}: {exc}"
                await queue.nack(msg_id, error=f"{type(exc).__name__}: {exc}")

    async def run_periodic(self) -> None:
        """Periodic background sync loop (metadata / content / delta on configurable intervals)."""
        self.status["running"] = True
        initial_delay = timedelta(seconds=self._settings.search_background_initial_delay_seconds)
        now = _now()
        next_metadata_at = (
            self._next_run_at(
                "last_metadata_sync_at",
                self._settings.search_background_metadata_interval_seconds,
                initial_delay=initial_delay,
                now=now,
            )
            if self._metadata_enabled()
            else None
        )
        next_content_at = (
            self._next_run_at(
                "last_content_sync_at",
                self._settings.search_background_content_interval_seconds,
                initial_delay=initial_delay,
                now=now,
            )
            if self._content_enabled()
            else None
        )
        next_delta_at = now + initial_delay if self._delta_enabled() else None
        self._set_next_times(next_metadata_at, next_content_at, next_delta_at)

        try:
            while self.status["running"]:
                self.status["enabled"] = self._periodic_enabled()
                self.status["metadata_enabled"] = self._metadata_enabled()
                self.status["content_enabled"] = self._content_enabled()
                self.status["delta_enabled"] = self._delta_enabled()
                if not self._periodic_enabled():
                    await asyncio.sleep(30)
                    continue

                if not self._owns_lock and not self._acquire_lock():
                    retry_at = _now() + timedelta(seconds=60)
                    self.status["lock_retry_at"] = _format_time(retry_at)
                    self._save_state()
                    await asyncio.sleep(60)
                    continue

                self._heartbeat_lock()
                now = _now()
                if self._metadata_enabled() and next_metadata_at and now >= next_metadata_at:
                    await self.run_metadata_once()
                    self._heartbeat_lock()
                    next_metadata_at = _now() + timedelta(
                        seconds=self._settings.search_background_metadata_interval_seconds
                    )
                    self.status["next_metadata_sync_at"] = _format_time(next_metadata_at)

                now = _now()
                if self._content_enabled() and next_content_at and now >= next_content_at:
                    await self.run_content_once()
                    self._heartbeat_lock()
                    next_content_at = _now() + timedelta(
                        seconds=self._settings.search_background_content_interval_seconds
                    )
                    self.status["next_content_sync_at"] = _format_time(next_content_at)

                now = _now()
                if self._delta_enabled() and next_delta_at and now >= next_delta_at:
                    await self.run_delta_once()
                    self._heartbeat_lock()
                    next_delta_at = _now() + timedelta(seconds=self._settings.search_delta_interval_seconds)
                    self.status["next_delta_sync_at"] = _format_time(next_delta_at)

                sleep_candidates = [30]
                if self._metadata_enabled() and next_metadata_at:
                    sleep_candidates.append(max((next_metadata_at - _now()).total_seconds(), 1))
                if self._content_enabled() and next_content_at:
                    sleep_candidates.append(max((next_content_at - _now()).total_seconds(), 1))
                if self._delta_enabled() and next_delta_at:
                    sleep_candidates.append(max((next_delta_at - _now()).total_seconds(), 1))
                await asyncio.sleep(min(sleep_candidates))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Portal search indexer stopped with error")
            self._record_error(exc)
        finally:
            self._release_lock()
            self.status["running"] = False
            self.status["metadata_running"] = False
            self.status["content_running"] = False
            self.status["delta_running"] = False

    async def run_full_once(self, *, include_content: bool = False) -> PortalSyncStats:
        async with self._lock:
            acquired_for_call = self._ensure_lock_for_operation()
            self.status["metadata_running"] = True
            started_at = _now()
            self.status["last_metadata_started_at"] = _format_time(started_at)
            self.status["last_error"] = None
            self._save_state()
            try:
                stats = await sync_portal_index(
                    self.bitrix,
                    self.index,
                    include_content=include_content,
                    settings=self._settings,
                )
                self.status["metadata_runs"] += 1
                self.status["last_metadata_sync_at"] = _format_time(_now())
                self.status["last_metadata_duration_seconds"] = round(
                    (_now() - started_at).total_seconds(),
                    3,
                )
                self.status["last_metadata_summary"] = format_portal_sync_stats(stats)
                self.status["last_metadata_prune_skipped"] = stats.prune_skipped or []
                self.status["last_metadata_stale_deleted"] = stats.stale_deleted
                self.status["last_metadata_total"] = stats.total
                if stats.content:
                    self._save_content_status(stats.content)
                self._record_success()
                self._save_state()
                return stats
            except Exception as exc:
                self._record_error(exc)
                raise
            finally:
                self.status["metadata_running"] = False
                if acquired_for_call:
                    self._release_lock()
                self._save_state()

    async def run_metadata_once(self) -> PortalSyncStats:
        return await self.run_full_once(include_content=False)

    async def run_content_once(self, *, extensions: set[str] | None = None) -> PortalContentSyncStats:
        async with self._lock:
            acquired_for_call = self._ensure_lock_for_operation()
            self.status["content_running"] = True
            started_at = _now()
            self.status["last_content_started_at"] = _format_time(started_at)
            self.status["last_error"] = None
            self._save_state()
            try:
                stats = await sync_portal_content_index(
                    self.bitrix,
                    self.index,
                    extensions=extensions,
                    settings=self._settings,
                )
                self._save_content_status(stats)
                self.status["last_content_duration_seconds"] = round(
                    (_now() - started_at).total_seconds(),
                    3,
                )
                self._record_success()
                self._save_state()
                return stats
            except Exception as exc:
                self._record_error(exc)
                raise
            finally:
                self.status["content_running"] = False
                if acquired_for_call:
                    self._release_lock()
                self._save_state()

    async def run_delta_once(self) -> PortalDeltaSyncStats:
        async with self._lock:
            acquired_for_call = self._ensure_lock_for_operation()
            self.status["delta_running"] = True
            started_at = _now()
            self.status["last_delta_started_at"] = _format_time(started_at)
            self.status["last_error"] = None
            self._save_state()
            try:
                stats = await sync_disk_delta_index(
                    self.bitrix,
                    self.index,
                    cursor_type=self.status.get("delta_folder_cursor_type"),
                    cursor_id=self.status.get("delta_folder_cursor_id"),
                    folder_limit=self._settings.search_delta_folders_per_run,
                    child_limit=self._settings.search_delta_max_children_per_folder,
                    settings=self._settings,
                )
                self._save_delta_status(stats)
                self.status["last_delta_duration_seconds"] = round(
                    (_now() - started_at).total_seconds(),
                    3,
                )
                self._record_success()
                self._save_state()
                return stats
            except Exception as exc:
                self._record_error(exc)
                raise
            finally:
                self.status["delta_running"] = False
                if acquired_for_call:
                    self._release_lock()
                self._save_state()

    def _save_content_status(self, stats: PortalContentSyncStats) -> None:
        self.status["content_runs"] += 1
        self.status["last_content_sync_at"] = _format_time(_now())
        self.status["last_content_summary"] = format_portal_content_sync_stats(stats)
        self.status["last_content_pending_after"] = self.index.content_readiness(
            allowed_extensions=self._settings.resolved_search_content_allowed_extensions,
        ).pending
        self._save_state()

    def _save_delta_status(self, stats: PortalDeltaSyncStats) -> None:
        self.status["delta_runs"] += 1
        self.status["last_delta_sync_at"] = _format_time(_now())
        self.status["last_delta_summary"] = format_portal_delta_sync_stats(stats)
        self.status["last_delta_folders_scanned"] = stats.folders_scanned
        self.status["last_delta_items_seen"] = stats.items_seen
        self.status["last_delta_items_changed"] = stats.items_changed
        self.status["last_delta_files_changed"] = stats.files_changed
        self.status["last_delta_folders_changed"] = stats.folders_changed
        self.status["last_delta_deleted"] = stats.deleted
        self.status["delta_folder_cursor_type"] = stats.cursor_type
        self.status["delta_folder_cursor_id"] = stats.cursor_id
        self._save_state()

    def _set_next_times(
        self,
        metadata_at: datetime | None,
        content_at: datetime | None,
        delta_at: datetime | None,
    ) -> None:
        self.status["next_metadata_sync_at"] = _format_time(metadata_at) if metadata_at else None
        self.status["next_content_sync_at"] = _format_time(content_at) if content_at else None
        self.status["next_delta_sync_at"] = _format_time(delta_at) if delta_at else None

    def _periodic_enabled(self) -> bool:
        return self._settings.search_background_periodic_enabled

    def _metadata_enabled(self) -> bool:
        return self._settings.search_background_periodic_metadata_enabled

    def _content_enabled(self) -> bool:
        return self._settings.search_content_enabled and self._settings.search_background_periodic_content_enabled

    def _delta_enabled(self) -> bool:
        return self._settings.search_delta_indexer_enabled and self._settings.search_background_periodic_delta_enabled

    def _next_run_at(
        self,
        last_run_key: str,
        interval_seconds: int,
        *,
        initial_delay: timedelta,
        now: datetime,
    ) -> datetime:
        last_run = _parse_time(self.status.get(last_run_key))
        if not last_run:
            return now + initial_delay
        due_at = last_run + timedelta(seconds=interval_seconds)
        if due_at > now:
            return due_at
        return now + timedelta(seconds=1)

    def _record_success(self) -> None:
        self.status["last_success_at"] = _format_time(_now())
        self.status["consecutive_errors"] = 0
        self.status["last_error"] = None
        self.status["last_error_at"] = None

    def _record_error(self, exc: Exception) -> None:
        self.status["last_error"] = f"{type(exc).__name__}: {exc}"
        self.status["last_error_at"] = _format_time(_now())
        self.status["consecutive_errors"] = int(self.status.get("consecutive_errors") or 0) + 1
        self._save_state()

    def _ensure_lock_for_operation(self) -> bool:
        if self._owns_lock:
            self._heartbeat_lock()
            return False
        if self._acquire_lock():
            return True
        owner = self.status.get("lock_owner")
        raise RuntimeError(f"Portal search indexer is locked by another process: {owner}")

    def _acquire_lock(self) -> bool:
        path = self._settings.search_background_lock_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._lock_payload()
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(path), flags)
        except FileExistsError:
            existing = self._read_lock()
            if self._lock_is_stale(existing):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    self._set_lock_status(acquired=False, owner=existing)
                    return False
                return self._acquire_lock()
            self._set_lock_status(acquired=False, owner=existing)
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        self._owns_lock = True
        self._set_lock_status(acquired=True, owner=payload)
        return True

    def _heartbeat_lock(self) -> None:
        if not self._owns_lock:
            return
        payload = self._lock_payload()
        try:
            self._settings.search_background_lock_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            logger.warning("Failed to update portal search indexer lock heartbeat", exc_info=True)
            return
        self._set_lock_status(acquired=True, owner=payload)

    def _release_lock(self) -> None:
        if not self._owns_lock:
            return
        path = self._settings.search_background_lock_path
        existing = self._read_lock()
        if existing and int(existing.get("pid") or -1) != os.getpid():
            self._owns_lock = False
            self._set_lock_status(acquired=False, owner=existing)
            return
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning("Failed to remove portal search indexer lock", exc_info=True)
        self._owns_lock = False
        self._set_lock_status(acquired=False, owner=None)

    def _read_lock(self) -> dict[str, Any] | None:
        path = self._settings.search_background_lock_path
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("search_indexer lock file corrupt, treating as stale: %s", exc)
            return {"pid": None, "heartbeat_at": None, "corrupt": True}
        return data if isinstance(data, dict) else {"pid": None, "heartbeat_at": None}

    def _lock_is_stale(self, payload: dict[str, Any] | None) -> bool:
        if not payload:
            return False
        heartbeat = str(payload.get("heartbeat_at") or payload.get("started_at") or "")
        try:
            heartbeat_at = datetime.fromisoformat(heartbeat)
        except ValueError:
            return True
        age = (_now() - heartbeat_at.astimezone(MOSCOW_TZ)).total_seconds()
        return age > self._settings.search_background_lock_stale_seconds

    def _lock_payload(self) -> dict[str, Any]:
        now = _format_time(_now())
        existing = self._read_lock() if self._owns_lock else {}
        return {
            "pid": os.getpid(),
            "started_at": existing.get("started_at") if existing else now,
            "heartbeat_at": now,
            "state_path": str(self._settings.search_background_state_path),
        }

    def _set_lock_status(self, *, acquired: bool, owner: dict[str, Any] | None) -> None:
        self.status["lock_acquired"] = acquired
        self.status["lock_owner"] = owner
        self.status["lock_last_heartbeat_at"] = str(owner.get("heartbeat_at") or "") if owner else None
        self.status["lock_path"] = str(self._settings.search_background_lock_path)
        if acquired:
            self.status["lock_retry_at"] = None

    def _load_state(self) -> None:
        path = self._settings.search_background_state_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load portal search indexer state", exc_info=True)
            return
        if not isinstance(data, dict):
            return
        for key in PERSISTED_STATUS_KEYS:
            if key in data:
                self.status[key] = data[key]

    def _save_state(self) -> None:
        path = self._settings.search_background_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {key: self.status.get(key) for key in PERSISTED_STATUS_KEYS}
        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            logger.warning("Failed to save portal search indexer state", exc_info=True)


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)


def _format_time(value: datetime) -> str:
    return value.astimezone(MOSCOW_TZ).isoformat()


def _parse_time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed.astimezone(MOSCOW_TZ)
