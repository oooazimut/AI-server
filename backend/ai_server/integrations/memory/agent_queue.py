from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4


class InMemoryAgentQueue:
    """In-process agent queue — asyncio.Queue per agent_id. Fallback when Redis is not configured."""

    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}

    def _queue(self, agent_id: str) -> asyncio.Queue[dict[str, Any]]:
        if agent_id not in self._queues:
            self._queues[agent_id] = asyncio.Queue()
        return self._queues[agent_id]

    async def publish(self, message: dict[str, Any]) -> None:
        agent_id = str(message.get("to") or "")
        if not agent_id:
            return
        msg = {**message, "id": message.get("id") or uuid4().hex}
        await self._queue(agent_id).put(msg)

    async def claim_next(self, agent_id: str) -> dict[str, Any] | None:
        q = self._queue(agent_id)
        try:
            return q.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def ack(self, message_id: str) -> None:
        pass  # in-memory: no persistence, nothing to ack

    async def nack(self, message_id: str, *, error: str) -> None:
        pass  # in-memory: no retry, message is dropped on error
