from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from ai_server.models import AgentResult, AgentTask

logger = logging.getLogger(__name__)


async def deliver_outbound_once(
    queue: Any,
    *,
    channels: dict[str, Any],
    conversation_trace: Any = None,
    incident_queue: Any = None,
) -> dict[str, Any] | None:
    recover = getattr(queue, "recover_stale", None)
    if callable(recover):
        for recovered_id in await recover():
            recovered = await queue.get(recovered_id)
            if not recovered:
                continue
            recovered_task = _validated(AgentTask, recovered.get("task") or _decoded(recovered.get("task_json")))
            recovered_result = _validated(
                AgentResult,
                recovered.get("result") or _decoded(recovered.get("result_json")),
            )
            await _trace(
                conversation_trace,
                task=recovered_task,
                result=recovered_result,
                recipient_id=str(recovered.get("recipient_id") or ""),
                body=str(recovered.get("body") or ""),
                status="unknown",
                error="worker_lost_during_delivery",
                delivery_id=recovered_id,
            )
            await _publish_incident(incident_queue, task=recovered_task, result=recovered_result)
    delivery = await queue.claim_next()
    if delivery is None:
        return None
    delivery_id = str(delivery.get("delivery_id") or "")
    claim_token = str(delivery.get("claim_token") or "")
    channel_id = str(delivery.get("channel_id") or "")
    recipient_id = str(delivery.get("recipient_id") or "")
    body = str(delivery.get("body") or "")
    channel = channels.get(channel_id)
    task = _validated(AgentTask, delivery.get("task"))
    result = _validated(AgentResult, delivery.get("result"))

    if channel is None:
        marked = await queue.mark_retryable_failed(
            delivery_id,
            claim_token=claim_token,
            error=f"channel_not_registered:{channel_id}",
        )
        return {"delivery_id": delivery_id, "status": "retry", "marked": bool(marked)}

    if not await queue.renew_claim(delivery_id, claim_token=claim_token):
        return {"delivery_id": delivery_id, "status": "claim_lost", "marked": False}
    if not await queue.begin_delivery(delivery_id, claim_token=claim_token):
        return {"delivery_id": delivery_id, "status": "claim_lost", "marked": False}

    try:
        await channel.send(recipient_id, body)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Once transport was invoked its remote outcome is ambiguous. Keep the
        # durable record terminal-unknown; a blind retry could duplicate a message.
        marked = await queue.mark_unknown(delivery_id, claim_token=claim_token, error=error)
        await _trace(
            conversation_trace,
            task=task,
            result=result,
            recipient_id=recipient_id,
            body=body,
            status="unknown",
            error="channel_send_outcome_unknown",
            delivery_id=delivery_id,
        )
        await _publish_incident(incident_queue, task=task, result=result)
        logger.exception("Outbound delivery outcome unknown id=%s channel=%s", delivery_id, channel_id)
        return {"delivery_id": delivery_id, "status": "unknown", "marked": bool(marked)}

    marked = await queue.mark_sent(delivery_id, claim_token=claim_token)
    await _trace(
        conversation_trace,
        task=task,
        result=result,
        recipient_id=recipient_id,
        body=body,
        status="sent",
        error="" if marked else "claim_lost_after_send",
        delivery_id=delivery_id,
    )
    if not marked:
        logger.error("Outbound claim lost after send id=%s", delivery_id)
        return {"delivery_id": delivery_id, "status": "unknown", "marked": False}
    return {"delivery_id": delivery_id, "status": "sent", "marked": True}


async def run_outbound_delivery_worker(
    queue: Any,
    *,
    channels: dict[str, Any],
    conversation_trace: Any = None,
    incident_queue: Any = None,
    interval_seconds: float = 0.2,
) -> None:
    while True:
        try:
            delivered = await deliver_outbound_once(
                queue,
                channels=channels,
                conversation_trace=conversation_trace,
                incident_queue=incident_queue,
            )
            if delivered is None:
                await asyncio.sleep(max(0.05, interval_seconds))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Outbound delivery worker iteration failed")
            await asyncio.sleep(max(0.05, interval_seconds))


def _validated(model: Any, payload: object) -> Any | None:
    if not isinstance(payload, dict):
        return None
    try:
        return model.model_validate(payload)
    except Exception:
        return None


def _decoded(raw: object) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


async def _publish_incident(incident_queue: Any, *, task: AgentTask | None, result: AgentResult | None) -> None:
    if incident_queue is None or task is None or result is None:
        return
    await incident_queue.publish(
        {
            "event_type": "agent_result",
            "source": "outbound_delivery",
            "task": task.model_dump(),
            "result": result.model_dump(),
            "created_at": datetime.now(UTC).isoformat(),
        }
    )


async def _trace(
    conversation_trace: Any,
    *,
    task: AgentTask | None,
    result: AgentResult | None,
    recipient_id: str,
    body: str,
    status: str,
    error: str,
    delivery_id: str,
) -> None:
    if conversation_trace is None or task is None or result is None:
        return
    await conversation_trace.record_outbound(
        task=task,
        result=result,
        recipient_id=recipient_id,
        body=body,
        status=status,
        error=error,
        delivery_id=delivery_id,
    )
