from __future__ import annotations

import json
import logging
import time
from typing import Any
from uuid import uuid4

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_PREFIX = "ai_server:aq"
_PROCESSING_TTL = 120  # seconds before a claimed message is considered stale
_MAX_NACK_RETRIES = 3


def _pending_key(agent_id: str) -> str:
    return f"{_PREFIX}:{agent_id}:pending"


def _processing_key(agent_id: str) -> str:
    return f"{_PREFIX}:{agent_id}:processing"


def _data_key(msg_id: str) -> str:
    return f"{_PREFIX}:data:{msg_id}"


class RedisAgentQueue:
    """Redis-backed agent queue using sorted sets (score = timestamp)."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def publish(self, message: dict[str, Any]) -> None:
        agent_id = str(message.get("to") or "")
        if not agent_id:
            return
        msg_id = message.get("id") or uuid4().hex
        msg = {**message, "id": msg_id}
        r = await self._get_client()
        score = time.time()
        pipe = r.pipeline()
        pipe.set(_data_key(msg_id), json.dumps(msg, ensure_ascii=False, default=str), ex=3600 * 24)
        pipe.zadd(_pending_key(agent_id), {msg_id: score})
        await pipe.execute()

    async def claim_next(self, agent_id: str) -> dict[str, Any] | None:
        r = await self._get_client()
        # Reclaim stale processing messages first
        stale_before = time.time() - _PROCESSING_TTL
        stale_ids = await r.zrangebyscore(_processing_key(agent_id), 0, stale_before)
        if stale_ids:
            pipe = r.pipeline()
            pipe.zrem(_processing_key(agent_id), *stale_ids)
            for sid in stale_ids:
                pipe.zadd(_pending_key(agent_id), {sid: time.time()})
            await pipe.execute()

        # Pop next pending message
        results = await r.zpopmin(_pending_key(agent_id), 1)
        if not results:
            return None
        msg_id = str(results[0][0])
        await r.zadd(_processing_key(agent_id), {msg_id: time.time()})
        raw = await r.get(_data_key(msg_id))
        if raw is None:
            await r.zrem(_processing_key(agent_id), msg_id)
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("RedisAgentQueue: invalid JSON for message %s", msg_id)
            await r.zrem(_processing_key(agent_id), msg_id)
            return None

    async def ack(self, message_id: str) -> None:
        r = await self._get_client()
        raw = await r.get(_data_key(message_id))
        if raw is not None:
            try:
                agent_id = str(json.loads(raw).get("to") or "")
                if agent_id:
                    await r.zrem(_processing_key(agent_id), message_id)
            except json.JSONDecodeError:
                pass
        await r.delete(_data_key(message_id))

    async def nack(self, message_id: str, *, error: str) -> None:
        r = await self._get_client()
        raw = await r.get(_data_key(message_id))
        if raw is None:
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        retries = int(msg.get("_retries") or 0)
        agent_id = str(msg.get("to") or "")
        if agent_id:
            await r.zrem(_processing_key(agent_id), message_id)
        if retries < _MAX_NACK_RETRIES and agent_id:
            msg["_retries"] = retries + 1
            msg["_last_error"] = error
            pipe = r.pipeline()
            pipe.set(_data_key(message_id), json.dumps(msg, ensure_ascii=False, default=str), ex=3600 * 24)
            pipe.zadd(_pending_key(agent_id), {message_id: time.time()})
            await pipe.execute()
        else:
            logger.warning("RedisAgentQueue: message %s exceeded max retries, dropping. error=%s", message_id, error)
            await r.delete(_data_key(message_id))
