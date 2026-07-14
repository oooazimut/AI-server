from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from ai_server.agents.ports import AgentQueuePort
from ai_server.attachments import AttachmentService
from ai_server.integrations.bitrix.chat_parser import (
    build_agent_task_from_bitrix_chat,
    build_agent_task_from_task_event,
    make_line_dialog_key,
)
from ai_server.integrations.bitrix.events import MESSAGE_EVENTS
from ai_server.models import AgentTask
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ
from ai_server.workers.bitrix.dialog_lines import (
    DEFAULT_AUTO_LINE_MAX,
    choose_auto_line_id,
    dialog_line_label,
    is_auto_line_candidate,
)
from ai_server.workers.bitrix.search_webhook_indexer import DISK_FILE_EVENT_MARKERS
from ai_server.workers.ports import WebhookConsumePort

logger = logging.getLogger(__name__)

_TASK_EVENT_PREFIXES = ("ONTASKUPDATE", "ONTASKCOMPLETE", "ONTASKADD")


def _is_task_event(event_type: str) -> bool:
    return any(event_type.startswith(p) for p in _TASK_EVENT_PREFIXES)


def _is_disk_event(event_type: str) -> bool:
    return all(marker in event_type for marker in DISK_FILE_EVENT_MARKERS)


async def run_webhook_event_worker(
    queue: WebhookConsumePort,
    *,
    agent_queue: AgentQueuePort,
    attachment_service: AttachmentService,
    transcriber: Any,
    status: dict[str, Any],
    settings: Settings,
    feedback_receiver: Any = None,
    conversation_trace: Any = None,
) -> None:
    worker_count = settings.webhook_event_queue_worker_count
    active_partition_keys: set[str] = set()
    active_lock = asyncio.Lock()
    status.update(
        {
            "enabled": settings.webhook_event_queue_enabled,
            "running": True,
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
                feedback_receiver=feedback_receiver,
                conversation_trace=conversation_trace,
            )
        )
        for index in range(worker_count)
    ]
    heartbeat_task = _create_worker_heartbeat_task(queue, status)
    if heartbeat_task is not None:
        tasks.append(heartbeat_task)
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        status["running"] = False
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _create_worker_heartbeat_task(queue: WebhookConsumePort, status: dict[str, Any]) -> asyncio.Task | None:
    heartbeat = getattr(queue, "heartbeat_worker", None)
    if heartbeat is None:
        return None

    async def _run() -> None:
        while True:
            try:
                await heartbeat(status)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Failed to write webhook event worker heartbeat", exc_info=True)
            await asyncio.sleep(5)

    return asyncio.create_task(_run())


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
    feedback_receiver: Any = None,
    conversation_trace: Any = None,
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
                event_id=event_id,
                event_type=event_type,
                payload=dict(event.get("payload") or {}),
                partition_key=partition_key,
                agent_queue=agent_queue,
                attachment_service=attachment_service,
                transcriber=transcriber,
                settings=settings,
                feedback_receiver=feedback_receiver,
                conversation_trace=conversation_trace,
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
    event_id: int | None = None,
    event_type: str,
    payload: dict[str, Any],
    partition_key: str = "",
    agent_queue: AgentQueuePort,
    attachment_service: AttachmentService,
    transcriber: Any,
    settings: Settings,
    feedback_receiver: Any = None,
    conversation_trace: Any = None,
) -> dict[str, Any]:
    """Route a Bitrix webhook event to the appropriate agent queue."""
    if event_type in MESSAGE_EVENTS:
        task = await build_agent_task_from_bitrix_chat(
            payload,
            attachment_service=attachment_service,
            transcriber=transcriber,
            settings=settings,
        )
        task = await _maybe_assign_auto_dialog_line(task, agent_queue=agent_queue, settings=settings)
        if feedback_receiver is not None:
            try:
                intercepted = await feedback_receiver.handle(task)
            except Exception:
                logger.exception("_route_event: feedback_receiver.handle failed")
                intercepted = False
            if intercepted:
                await _trace_route(
                    conversation_trace,
                    event_id=event_id,
                    event_type=event_type,
                    routed_to="feedback_receiver",
                    task=task,
                    partition_key=partition_key,
                    result={"handled": True, "routed_to": "feedback_receiver", "event": event_type},
                )
                return {"handled": True, "routed_to": "feedback_receiver", "event": event_type}
        await agent_queue.publish(
            {
                "to": "orchestrator",
                "from": "webhook_worker",
                "type": "bitrix_chat",
                "payload": task.model_dump(),
            }
        )
        await _trace_route(
            conversation_trace,
            event_id=event_id,
            event_type=event_type,
            routed_to="orchestrator",
            task=task,
            partition_key=partition_key,
            result={"handled": True, "routed_to": "orchestrator", "event": event_type},
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
        await _trace_route(
            conversation_trace,
            event_id=event_id,
            event_type=event_type,
            routed_to="bitrix24",
            task=task,
            partition_key=partition_key,
            result={"handled": True, "routed_to": "bitrix24", "event": event_type},
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
        await _trace_route(
            conversation_trace,
            event_id=event_id,
            event_type=event_type,
            routed_to="index_refresher",
            partition_key=partition_key,
            result={"handled": True, "routed_to": "index_refresher", "event": event_type},
        )
        return {"handled": True, "routed_to": "index_refresher", "event": event_type}

    await _trace_route(
        conversation_trace,
        event_id=event_id,
        event_type=event_type,
        routed_to="unsupported",
        partition_key=partition_key,
        result={"handled": False, "reason": "unsupported_event", "event": event_type},
    )
    return {"handled": False, "reason": "unsupported_event", "event": event_type}


async def _maybe_assign_auto_dialog_line(
    task: AgentTask,
    *,
    agent_queue: AgentQueuePort,
    settings: Settings,
) -> AgentTask:
    if not getattr(settings, "bitrix_auto_lines_enabled", False):
        return task
    context = dict(task.context or {})
    if context.get("dialog_line_id"):
        return task
    if not is_auto_line_candidate(task.request):
        return task
    base_dialog_key = str(context.get("base_dialog_key") or context.get("dialog_key") or "").strip()
    if not base_dialog_key:
        return task
    active_partitions_fn = getattr(agent_queue, "active_partition_keys", None)
    if active_partitions_fn is None:
        return task
    try:
        active_partitions = await active_partitions_fn("orchestrator")
    except Exception:
        logger.exception("_maybe_assign_auto_dialog_line: failed to inspect active orchestrator partitions")
        return task
    line_id = choose_auto_line_id(
        set(active_partitions),
        base_dialog_key,
        max_lines=getattr(settings, "bitrix_auto_line_max", DEFAULT_AUTO_LINE_MAX),
    )
    if line_id is None:
        return task
    line_id_text = str(line_id)
    return task.model_copy(
        update={
            "context": {
                **context,
                "dialog_key": make_line_dialog_key(base_dialog_key, line_id_text),
                "base_dialog_key": base_dialog_key,
                "dialog_line_id": line_id_text,
                "dialog_line_label": dialog_line_label(line_id),
                "dialog_auto_line": True,
            }
        }
    )


async def _trace_route(
    conversation_trace: Any,
    *,
    event_id: int | None,
    event_type: str,
    routed_to: str,
    task: Any = None,
    partition_key: str = "",
    result: dict[str, Any] | None = None,
) -> None:
    if conversation_trace is None:
        return
    await conversation_trace.record_route(
        event_id=event_id,
        event_type=event_type,
        routed_to=routed_to,
        task=task,
        partition_key=partition_key,
        result=result,
    )


def _update_active_status(status: dict[str, Any], active_partition_keys: set[str]) -> None:
    status["active_workers"] = len(active_partition_keys)
    status["active_partition_keys"] = sorted(active_partition_keys)[:20]


def _now() -> datetime:
    return datetime.now(MOSCOW_TZ)
