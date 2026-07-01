from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from ai_server.integrations.redis.diagnost_queue import RedisDiagnostQueue
from ai_server.models import AgentResult, AgentTask

logger = logging.getLogger(__name__)


class OrchestratorResultPublisher:
    """Implements ResultPublisherPort: publishes orchestrator results to RedisDiagnostQueue."""

    def __init__(self, queue: RedisDiagnostQueue) -> None:
        self._queue = queue

    async def publish(self, task: AgentTask, result: AgentResult) -> None:
        try:
            await self._queue.publish(
                {
                    "event_type": "agent_result",
                    "source": "orchestrator",
                    "task": task.model_dump(),
                    "result": result.model_dump(),
                    "created_at": _now_iso(),
                }
            )
        except Exception:
            logger.exception("OrchestratorResultPublisher: failed to publish event")


class SpecialistResultPublisher:
    """Implements ResultPublisherPort: publishes specialist results to RedisDiagnostQueue."""

    def __init__(self, queue: RedisDiagnostQueue) -> None:
        self._queue = queue

    async def publish(self, task: AgentTask, result: AgentResult) -> None:
        try:
            await self._queue.publish(
                {
                    "event_type": "agent_result",
                    "source": "specialist",
                    "task": task.model_dump(),
                    "result": result.model_dump(),
                    "created_at": _now_iso(),
                }
            )
        except Exception:
            logger.exception("SpecialistResultPublisher: failed to publish event")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _no_op_publisher() -> _NoOpPublisher:
    return _NoOpPublisher()


class _NoOpPublisher:
    async def publish(self, task: Any, result: Any) -> None:
        pass
