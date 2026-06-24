from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Collection
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ai_server.agent_store import AgentStore
from ai_server.agents.ports import AgentQueuePort
from ai_server.attachments import AttachmentService
from ai_server.integrations.bitrix.chat_parser import (
    build_agent_task_from_bitrix_chat,
    build_agent_task_from_task_event,
)
from ai_server.integrations.bitrix.events import MESSAGE_EVENTS
from ai_server.integrations.webhook_utils import (
    sanitize_webhook_payload,
    webhook_event_key,
    webhook_event_partition_key,
)
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ
from ai_server.workers.bitrix.search_webhook_indexer import DISK_FILE_EVENT_MARKERS
from ai_server.workers.ports import WebhookConsumePort

logger = logging.getLogger(__name__)

_TASK_EVENT_PREFIXES = ("ONTASKUPDATE", "ONTASKCOMPLETE", "ONTASKADD")


def _is_task_event(event_type: str) -> bool:
    return any(event_type.startswith(p) for p in _TASK_EVENT_PREFIXES)


def _is_disk_event(event_type: str) -> bool:
    return all(marker in event_type for marker in DISK_FILE_EVENT_MARKERS)


class WebhookEventQueue(AgentStore):
    def __init__(self, path: Path, *, settings: Settings) -> None:
        super().__init__("webhook_event_queue", path=path)
        self._settings = settings

    def ensure_schema(self) -> None:
        super().ensure_schema()
        with self._connection() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS webhook_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_key TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    received_at TEXT NOT NULL,
                    started_at TEXT,
                    processed_at TEXT,
                    next_attempt_at TEXT,
                    last_error TEXT,
                    last_result_json TEXT
                )
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_webhook_events_status_next_attempt
                    ON webhook_events(status, next_attempt_at, id)
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_webhook_events_received_at
                    ON webhook_events(received_at)
                """
            )

    async def enqueue(
        self,
        payload: dict[str, Any],
        *,
        event_type: str,
        dedupe_key: str | None = None,
    ) -> tuple[int, bool]:
        return await asyncio.to_thread(self._sync_enqueue, payload, event_type=event_type, dedupe_key=dedupe_key)

    def _sync_enqueue(
        self,
        payload: dict[str, Any],
        *,
        event_type: str,
        dedupe_key: str | None = None,
    ) -> tuple[int, bool]:
        safe_payload = sanitize_webhook_payload(payload)
        received_at = _now().isoformat()
        event_key = dedupe_key or webhook_event_key(
            safe_payload,
            event_type=event_type,
            received_at=received_at,
        )
        payload_json = json.dumps(safe_payload, ensure_ascii=False, sort_keys=True, default=str)
        with self._connection() as db:
            cursor = db.execute(
                """
                INSERT OR IGNORE INTO webhook_events
                    (event_key, event_type, payload_json, status, received_at)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (event_key, event_type, payload_json, received_at),
            )
            inserted = bool(cursor.lastrowid)
            if inserted:
                return int(cursor.lastrowid), True
            row = db.execute(
                "SELECT id FROM webhook_events WHERE event_key = ?",
                (event_key,),
            ).fetchone()
        return (int(row["id"]) if row else 0), False

    async def claim_next(
        self,
        *,
        blocked_partition_keys: Collection[str] | None = None,
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._sync_claim_next, blocked_partition_keys=blocked_partition_keys)

    def _sync_claim_next(
        self,
        *,
        blocked_partition_keys: Collection[str] | None = None,
    ) -> dict[str, Any] | None:
        settings = self._settings
        now = _now()
        now_iso = now.isoformat()
        stale_before = (now - timedelta(seconds=settings.webhook_event_queue_stale_processing_seconds)).isoformat()
        blocked = set(blocked_partition_keys or ())
        selected_payload: dict[str, Any] | None = None
        selected_partition_key = ""
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """
                UPDATE webhook_events
                SET status = 'pending',
                    next_attempt_at = ?,
                    last_error = COALESCE(last_error, 'processing_stale')
                WHERE status = 'processing'
                  AND started_at IS NOT NULL
                  AND started_at < ?
                """,
                (now_iso, stale_before),
            )
            rows = db.execute(
                """
                SELECT *
                FROM webhook_events
                WHERE status = 'pending'
                  AND attempts < ?
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                ORDER BY id ASC
                LIMIT ?
                """,
                (
                    settings.webhook_event_queue_max_attempts,
                    now_iso,
                    settings.webhook_event_queue_claim_scan_limit,
                ),
            ).fetchall()
            row = None
            for candidate in rows:
                payload = _decode_payload_json(str(candidate["payload_json"] or "{}"))
                partition_key = webhook_event_partition_key(
                    payload,
                    event_type=str(candidate["event_type"] or ""),
                )
                if partition_key in blocked:
                    continue
                row = candidate
                selected_payload = payload
                selected_partition_key = partition_key
                break
            if not row:
                return None
            attempts = int(row["attempts"] or 0) + 1
            db.execute(
                """
                UPDATE webhook_events
                SET status = 'processing',
                    attempts = ?,
                    started_at = ?,
                    last_error = NULL
                WHERE id = ?
                """,
                (attempts, now_iso, int(row["id"])),
            )
        event = dict(row)
        event["attempts"] = attempts
        event["payload"] = selected_payload or {}
        event["partition_key"] = selected_partition_key
        return event

    async def mark_done(self, event_id: int, result: dict[str, Any]) -> None:
        await asyncio.to_thread(self._sync_mark_done, event_id, result)

    def _sync_mark_done(self, event_id: int, result: dict[str, Any]) -> None:
        with self._connection() as db:
            db.execute(
                """
                UPDATE webhook_events
                SET status = 'done',
                    processed_at = ?,
                    next_attempt_at = NULL,
                    last_error = NULL,
                    last_result_json = ?
                WHERE id = ?
                """,
                (
                    _now().isoformat(),
                    json.dumps(result, ensure_ascii=False, sort_keys=True, default=str),
                    event_id,
                ),
            )

    async def mark_failed(self, event_id: int, error: str) -> None:
        await asyncio.to_thread(self._sync_mark_failed, event_id, error)

    def _sync_mark_failed(self, event_id: int, error: str) -> None:
        settings = self._settings
        now = _now()
        with self._connection() as db:
            row = db.execute("SELECT attempts FROM webhook_events WHERE id = ?", (event_id,)).fetchone()
            attempts = int(row["attempts"] or 0) if row else 0
            if attempts >= settings.webhook_event_queue_max_attempts:
                status = "failed"
                next_attempt_at = None
            else:
                status = "pending"
                delay = min(
                    settings.webhook_event_queue_retry_max_seconds,
                    settings.webhook_event_queue_retry_base_seconds * max(1, attempts),
                )
                next_attempt_at = (now + timedelta(seconds=delay)).isoformat()
            db.execute(
                """
                UPDATE webhook_events
                SET status = ?,
                    next_attempt_at = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (status, next_attempt_at, error[:1000], event_id),
            )

    async def stats(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._sync_stats)

    def _sync_stats(self) -> dict[str, Any]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM webhook_events
                GROUP BY status
                """
            ).fetchall()
            latest = db.execute(
                """
                SELECT id, event_type, status, attempts, received_at, processed_at, last_error
                FROM webhook_events
                ORDER BY id DESC
                LIMIT 1
                """
            ).fetchone()
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return {
            "path": str(self.path),
            "counts": counts,
            "pending": counts.get("pending", 0),
            "processing": counts.get("processing", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
            "latest": dict(latest) if latest else None,
        }

    async def latest(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._sync_latest, limit=limit)

    def _sync_latest(self, *, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT id, event_type, status, attempts, received_at, started_at,
                       processed_at, next_attempt_at, last_error
                FROM webhook_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


async def run_webhook_event_worker(
    queue: WebhookConsumePort,
    *,
    agent_queue: AgentQueuePort,
    attachment_service: AttachmentService,
    transcriber: Any,
    status: dict[str, Any],
    settings: Settings,
) -> None:
    worker_count = settings.webhook_event_queue_worker_count
    active_partition_keys: set[str] = set()
    active_lock = asyncio.Lock()
    status.update(
        {
            "enabled": settings.webhook_event_queue_enabled,
            "running": True,
            "path": str(getattr(queue, "path", "redis")),
            "worker_count": worker_count,
            "active_workers": 0,
            "active_partition_keys": [],
            "last_check_at": None,
            "last_event_id": None,
            "last_event": None,
            "last_error": None,
            "processed": 0,
            "errors": 0,
        }
    )
    tasks = [
        asyncio.create_task(
            _run_webhook_event_worker_loop(
                worker_id=index + 1,
                queue=queue,
                agent_queue=agent_queue,
                attachment_service=attachment_service,
                transcriber=transcriber,
                status=status,
                active_partition_keys=active_partition_keys,
                active_lock=active_lock,
                settings=settings,
            )
        )
        for index in range(worker_count)
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        status["running"] = False
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def _run_webhook_event_worker_loop(
    *,
    worker_id: int,
    queue: WebhookConsumePort,
    agent_queue: AgentQueuePort,
    attachment_service: AttachmentService,
    transcriber: Any,
    status: dict[str, Any],
    active_partition_keys: set[str],
    active_lock: asyncio.Lock,
    settings: Settings,
) -> None:
    while True:
        event_id: int | None = None
        partition_key = ""
        try:
            status["last_check_at"] = _now().isoformat()
            async with active_lock:
                event = await queue.claim_next(blocked_partition_keys=active_partition_keys)
                if event:
                    partition_key = str(event.get("partition_key") or "event:unknown")
                    active_partition_keys.add(partition_key)
                    _update_active_status(status, active_partition_keys)
            if not event:
                await asyncio.sleep(settings.webhook_event_queue_interval_seconds)
                continue
            event_id = int(event["id"])
            event_type = str(event.get("event_type") or "").upper()
            status["last_event_id"] = event_id
            status["last_event"] = event_type
            status["last_worker_id"] = worker_id
            result = await _route_event(
                event_type=event_type,
                payload=dict(event.get("payload") or {}),
                agent_queue=agent_queue,
                attachment_service=attachment_service,
                transcriber=transcriber,
                settings=settings,
            )
            await queue.mark_done(event_id, result)
            status["last_error"] = None
            status["processed"] = int(status.get("processed") or 0) + 1
        except asyncio.CancelledError:
            status["running"] = False
            raise
        except Exception as exc:
            logger.exception("Webhook event worker %s failed", worker_id)
            if event_id is not None:
                await queue.mark_failed(event_id, f"{type(exc).__name__}: {exc}")
            status["last_error"] = f"{type(exc).__name__}: {exc}"
            status["errors"] = int(status.get("errors") or 0) + 1
            await asyncio.sleep(settings.webhook_event_queue_interval_seconds)
        finally:
            if partition_key:
                async with active_lock:
                    active_partition_keys.discard(partition_key)
                    _update_active_status(status, active_partition_keys)


async def _route_event(
    *,
    event_type: str,
    payload: dict[str, Any],
    agent_queue: AgentQueuePort,
    attachment_service: AttachmentService,
    transcriber: Any,
    settings: Settings,
) -> dict[str, Any]:
    """Route a Bitrix webhook event to the appropriate agent queue."""
    if event_type in MESSAGE_EVENTS:
        task = await build_agent_task_from_bitrix_chat(
            payload,
            attachment_service=attachment_service,
            transcriber=transcriber,
            settings=settings,
        )
        await agent_queue.publish(
            {
                "to": "orchestrator",
                "from": "webhook_worker",
                "type": "bitrix_chat",
                "payload": task.model_dump(),
            }
        )
        return {"handled": True, "routed_to": "orchestrator", "event": event_type}

    if _is_task_event(event_type):
        task = build_agent_task_from_task_event(payload)
        await agent_queue.publish(
            {
                "to": "bitrix24",
                "from": "webhook_worker",
                "type": "bitrix_event",
                "payload": task.model_dump(),
            }
        )
        return {"handled": True, "routed_to": "bitrix24", "event": event_type}

    if _is_disk_event(event_type):
        await agent_queue.publish(
            {
                "to": "index_refresher",
                "from": "webhook_worker",
                "type": "bitrix_event",
                "payload": payload,
            }
        )
        return {"handled": True, "routed_to": "index_refresher", "event": event_type}

    return {"handled": False, "reason": "unsupported_event", "event": event_type}


def _update_active_status(status: dict[str, Any], active_partition_keys: set[str]) -> None:
    status["active_workers"] = len(active_partition_keys)
    status["active_partition_keys"] = sorted(active_partition_keys)[:20]


def _decode_payload_json(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)
