from __future__ import annotations

import asyncio
import logging
from typing import Any

from ai_server.integrations.postgres.diagnost_agent import PostgresDiagnostStore
from ai_server.integrations.redis.diagnost_queue import RedisDiagnostQueue
from ai_server.models import AgentResult, AgentTask

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 1.0
_AUTO_INCIDENT_CONFIDENCE_THRESHOLD = 0.5


async def run_diagnost_event_worker(
    queue: RedisDiagnostQueue,
    store: PostgresDiagnostStore,
    *,
    confidence_threshold: float = _AUTO_INCIDENT_CONFIDENCE_THRESHOLD,
    conversation_trace: Any = None,
    trace_snapshot_enabled: bool = True,
    trace_settle_seconds: float = 1.0,
    high_latency_ms: float = 120000.0,
) -> None:
    """Reads agent_result events from RedisDiagnostQueue and writes to PostgresDiagnostStore.

    Automatically creates incidents for failed or low-confidence results.
    """
    logger.info("DiagnostWorker: started (confidence_threshold=%.2f)", confidence_threshold)
    while True:
        try:
            msg = await queue.claim_next()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("DiagnostWorker: claim_next failed")
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        if msg is None:
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        msg_id = str(msg.get("_id") or "")
        try:
            raw_task = msg.get("task")
            raw_result = msg.get("result")
            if not isinstance(raw_task, dict) or not isinstance(raw_result, dict):
                logger.warning("DiagnostWorker: malformed message id=%s", msg_id)
                await queue.ack(msg_id)
                continue

            task = AgentTask.model_validate(raw_task)
            result = AgentResult.model_validate(raw_result)

            source = str(msg.get("source") or "orchestrator")
            await store.save_event(task, result, source=source)

            trace_snapshot: list[dict[str, Any]] = []
            if trace_snapshot_enabled and conversation_trace is not None:
                if trace_settle_seconds > 0:
                    await asyncio.sleep(trace_settle_seconds)
                trace_snapshot = await conversation_trace.by_task(task.task_id, limit=500, hours=48)
                await store.save_trace_snapshot(task.task_id, trace_snapshot)

            confidence = result.confidence if result.confidence is not None else 1.0
            if result.status == "failed":
                await store.save_incident(task.task_id, reason="failed")
            elif confidence < confidence_threshold:
                await store.save_incident(task.task_id, reason="low_confidence")

            if source in {"orchestrator", "outbound_delivery"} and trace_snapshot:
                for reason in _trace_incident_reasons(task, result, trace_snapshot, high_latency_ms=high_latency_ms):
                    await store.save_incident(task.task_id, reason=reason)

            await queue.ack(msg_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("DiagnostWorker: failed processing message id=%s", msg_id)
            if msg_id:
                await queue.nack(msg_id, error=f"{type(exc).__name__}: {exc}")


def _trace_incident_reasons(
    task: AgentTask,
    result: AgentResult,
    trace: list[dict[str, Any]],
    *,
    high_latency_ms: float,
) -> list[str]:
    reasons: list[str] = []
    outbound = [event for event in trace if event.get("trace_type") == "outbound_message"]
    if any(str(event.get("send_status") or "").lower() == "error" for event in outbound):
        reasons.append("outbound_failed")
    if any(str(event.get("send_status") or "").lower() == "unknown" for event in outbound):
        reasons.append("outbound_unknown")
    if any(str(event.get("send_status") or "").lower() == "suppressed" for event in outbound):
        reasons.append("outbound_suppressed")
    expects_outbound = bool(
        result.status == "completed"
        and result.answer
        and task.context.get("channel_id")
        and task.context.get("recipient_id")
    )
    if expects_outbound and not outbound:
        reasons.append("missing_outbound")
    if any(
        event.get("trace_type") == "timing_step"
        and event.get("stage") == "handle_total"
        and float(event.get("elapsed_ms") or 0) >= high_latency_ms
        for event in trace
    ):
        reasons.append("high_latency")
    return reasons
