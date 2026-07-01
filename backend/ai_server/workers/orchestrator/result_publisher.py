from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ai_server.integrations.redis.diagnost_queue import RedisDiagnostQueue
from ai_server.models import AgentResult, AgentTask


class OrchestratorResultPublisher:
    """Implements ResultPublisherPort: publishes orchestrator results to RedisDiagnostQueue."""

    def __init__(self, queue: RedisDiagnostQueue) -> None:
        self._queue = queue

    async def publish(self, task: AgentTask, result: AgentResult) -> None:
        try:
            await self._queue.publish(
                {
                    "event_type": "agent_result",
                    "task": task.model_dump(),
                    "result": result.model_dump(),
                    "created_at": _now_iso(),
                }
            )
        except Exception:
            import logging

            logging.getLogger(__name__).exception("OrchestratorResultPublisher: failed to publish event")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _no_op_publisher() -> _NoOpPublisher:
    return _NoOpPublisher()


class _NoOpPublisher:
    async def publish(self, task: Any, result: Any) -> None:
        pass
