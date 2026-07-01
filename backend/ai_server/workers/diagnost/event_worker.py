from __future__ import annotations

import asyncio
import logging

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

            await store.save_event(task, result)

            confidence = result.confidence if result.confidence is not None else 1.0
            if result.status == "failed":
                await store.save_incident(task.task_id, reason="failed")
            elif confidence < confidence_threshold:
                await store.save_incident(task.task_id, reason="low_confidence")

            await queue.ack(msg_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("DiagnostWorker: failed processing message id=%s", msg_id)
            if msg_id:
                await queue.nack(msg_id, error=f"{type(exc).__name__}: {exc}")
