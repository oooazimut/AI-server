from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from ai_server.agent_queue_utils import agent_queue_partition_key


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

    async def claim_next(
        self,
        agent_id: str,
        *,
        blocked_partition_keys: set[str] | None = None,
    ) -> dict[str, Any] | None:
        q = self._queue(agent_id)
        blocked = set(blocked_partition_keys or ())
        deferred: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        while True:
            try:
                message = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            partition_key = agent_queue_partition_key(message)
            if partition_key and partition_key in blocked:
                deferred.append(message)
                continue
            if partition_key:
                message["_partition_key"] = partition_key
            selected = message
            break
        for message in deferred:
            await q.put(message)
        return selected

    async def ack(self, message_id: str) -> None:
        pass  # in-memory: no persistence, nothing to ack

    async def nack(self, message_id: str, *, error: str) -> None:
        pass  # in-memory: no retry, message is dropped on error

    async def active_partition_keys(self, agent_id: str) -> set[str]:
        q = self._queue(agent_id)
        partitions: set[str] = set()
        for message in list(q._queue):
            partition_key = agent_queue_partition_key(message)
            if partition_key:
                partitions.add(partition_key)
        return partitions

    async def remove_pending_by_partition(self, agent_id: str, partition_key: str) -> int:
        q = self._queue(agent_id)
        kept: list[dict[str, Any]] = []
        removed = 0
        while True:
            try:
                message = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if agent_queue_partition_key(message) == partition_key:
                removed += 1
            else:
                kept.append(message)
        for message in kept:
            await q.put(message)
        return removed
