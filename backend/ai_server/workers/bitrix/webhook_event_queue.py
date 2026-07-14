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
)
from ai_server.integrations.bitrix.events import MESSAGE_EVENTS
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
    dialog_guard: Any = None,
    bitrix_sender: Any = None,
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
                dialog_guard=dialog_guard,
                bitrix_sender=bitrix_sender,
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
    dialog_guard: Any = None,
    bitrix_sender: Any = None,
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
                dialog_guard=dialog_guard,
                bitrix_sender=bitrix_sender,
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
    dialog_guard: Any = None,
    bitrix_sender: Any = None,
) -> dict[str, Any]:
    """Route a Bitrix webhook event to the appropriate agent queue."""
    if event_type in MESSAGE_EVENTS:
        task = await build_agent_task_from_bitrix_chat(
            payload,
            attachment_service=attachment_service,
            transcriber=transcriber,
            settings=settings,
        )
        guard_result = await _handle_dialog_guard(
            task,
            agent_queue=agent_queue,
            settings=settings,
            dialog_guard=dialog_guard,
            bitrix_sender=bitrix_sender,
            conversation_trace=conversation_trace,
            event_id=event_id,
            event_type=event_type,
            partition_key=partition_key,
        )
        if guard_result is not None:
            return guard_result
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


async def _handle_dialog_guard(
    task: Any,
    *,
    agent_queue: AgentQueuePort,
    settings: Settings,
    dialog_guard: Any,
    bitrix_sender: Any,
    conversation_trace: Any,
    event_id: int | None,
    event_type: str,
    partition_key: str,
) -> dict[str, Any] | None:
    if dialog_guard is None or not getattr(settings, "bitrix_dialog_guard_enabled", True):
        return None
    context = task.context or {}
    dialog_key = str(context.get("dialog_key") or "").strip()
    recipient_id = str(context.get("recipient_id") or "").strip()
    if not dialog_key or not recipient_id:
        return None

    decision = _parse_stuck_dialog_decision(task.request)
    pending = await dialog_guard.get_pending(dialog_key)
    if pending is not None:
        if decision == "reset":
            pending_task = await dialog_guard.pop_pending(dialog_key)
            generation = await dialog_guard.increment_generation(dialog_key)
            deleted = 0
            remove_pending = getattr(agent_queue, "remove_pending_by_partition", None)
            if remove_pending is not None:
                deleted += int(await remove_pending("orchestrator", f"dialog:{dialog_key}") or 0)
                deleted += int(await remove_pending("bitrix24", f"dialog:{dialog_key}") or 0)
            if pending_task is not None:
                _set_task_generation(pending_task, generation)
                await _publish_orchestrator_task(agent_queue, pending_task)
            await _send_guard_message(
                bitrix_sender,
                recipient_id,
                "Сбросил предыдущий запрос и выполняю новый.",
                settings=settings,
            )
            await _trace_route(
                conversation_trace,
                event_id=event_id,
                event_type=event_type,
                routed_to="dialog_guard",
                task=task,
                partition_key=partition_key,
                result={
                    "handled": True,
                    "routed_to": "dialog_guard",
                    "event": event_type,
                    "action": "reset_previous",
                    "deleted_pending": deleted,
                },
            )
            return {"handled": True, "routed_to": "dialog_guard", "event": event_type, "action": "reset_previous"}
        if decision == "wait":
            pending_task = await dialog_guard.pop_pending(dialog_key)
            if pending_task is not None:
                _set_task_generation(pending_task, await dialog_guard.current_generation(dialog_key))
                await _publish_orchestrator_task(agent_queue, pending_task)
            await _send_guard_message(
                bitrix_sender,
                recipient_id,
                "Хорошо, выполню новый запрос после предыдущего.",
                settings=settings,
            )
            await _trace_route(
                conversation_trace,
                event_id=event_id,
                event_type=event_type,
                routed_to="dialog_guard",
                task=task,
                partition_key=partition_key,
                result={"handled": True, "routed_to": "dialog_guard", "event": event_type, "action": "wait_previous"},
            )
            return {"handled": True, "routed_to": "dialog_guard", "event": event_type, "action": "wait_previous"}
        await _send_guard_message(bitrix_sender, recipient_id, _guard_choice_message(), settings=settings)
        await _trace_route(
            conversation_trace,
            event_id=event_id,
            event_type=event_type,
            routed_to="dialog_guard",
            task=task,
            partition_key=partition_key,
            result={"handled": True, "routed_to": "dialog_guard", "event": event_type, "action": "clarify_choice"},
        )
        return {"handled": True, "routed_to": "dialog_guard", "event": event_type, "action": "clarify_choice"}

    active = await dialog_guard.get_active(dialog_key)
    if active is None:
        _set_task_generation(task, await dialog_guard.current_generation(dialog_key))
        return None
    active_age_seconds = float(active.get("age_seconds") or 0)
    if active_age_seconds < float(settings.bitrix_dialog_stuck_seconds):
        _set_task_generation(task, await dialog_guard.current_generation(dialog_key))
        return None

    try:
        await dialog_guard.save_pending(task, ttl_seconds=settings.bitrix_dialog_pending_ttl_seconds)
    except Exception:
        logger.exception("_handle_dialog_guard: failed to save pending task")
        _set_task_generation(task, await dialog_guard.current_generation(dialog_key))
        return None
    await _send_guard_message(bitrix_sender, recipient_id, _stuck_dialog_message(), settings=settings)
    await _trace_route(
        conversation_trace,
        event_id=event_id,
        event_type=event_type,
        routed_to="dialog_guard",
        task=task,
        partition_key=partition_key,
        result={
            "handled": True,
            "routed_to": "dialog_guard",
            "event": event_type,
            "action": "stuck_prompt",
            "active_age_seconds": round(active_age_seconds, 1),
        },
    )
    return {"handled": True, "routed_to": "dialog_guard", "event": event_type, "action": "stuck_prompt"}


async def _publish_orchestrator_task(agent_queue: AgentQueuePort, task: Any) -> None:
    await agent_queue.publish(
        {
            "to": "orchestrator",
            "from": "webhook_worker",
            "type": "bitrix_chat",
            "payload": task.model_dump(),
        }
    )


async def _send_guard_message(bitrix_sender: Any, recipient_id: str, message: str, *, settings: Settings) -> None:
    if bitrix_sender is None or settings.agent_dry_run:
        return
    sender = getattr(bitrix_sender, "send_bot_message", None)
    if sender is None:
        return
    await sender(recipient_id, message, bot_id=settings.bitrix_bot_id)


def _set_task_generation(task: Any, generation: int) -> None:
    task.context["dialog_cancel_generation"] = int(generation)


def _parse_stuck_dialog_decision(text: str) -> str:
    normalized = " ".join(str(text or "").casefold().strip().split())
    if normalized in {"сбросить предыдущий запрос", "отменить предыдущий запрос", "сбросить предыдущий"}:
        return "reset"
    if normalized in {"выполнить после предыдущего", "дождаться предыдущего ответа", "дождаться ответа"}:
        return "wait"
    return ""


def _stuck_dialog_message() -> str:
    return (
        "Предыдущий запрос в этом диалоге обрабатывается дольше обычного.\n\n"
        "Ответьте одной из фраз:\n"
        '"сбросить предыдущий запрос" — остановлю старую обработку и выполню новый запрос;\n'
        '"выполнить после предыдущего" — дождусь старого ответа и выполню новый запрос после него.'
    )


def _guard_choice_message() -> str:
    return 'Ответьте одной из фраз: "сбросить предыдущий запрос" или "выполнить после предыдущего".'


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
