from __future__ import annotations

import json
import logging
import time
from typing import Any
from uuid import uuid4

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_PREFIX = "ai_server:dq"
_PROCESSING_TTL = 300
_MAX_NACK_RETRIES = 3
_DATA_TTL = 3600 * 48
_MISSING_DATA_SCAN_LIMIT = 100


def _pending_key() -> str:
    return f"{_PREFIX}:pending"


def _processing_key() -> str:
    return f"{_PREFIX}:processing"


def _data_key(msg_id: str) -> str:
    return f"{_PREFIX}:data:{msg_id}"


class RedisDiagnostQueue:
    """Redis-backed queue for diagnostic events (agent_results from orchestrator)."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        if self._client is None:
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def publish(self, event: dict[str, Any]) -> None:
        msg_id = str(uuid4().hex)
        msg = {**event, "_id": msg_id}
        r = await self._get_client()
        pipe = r.pipeline()
        pipe.set(_data_key(msg_id), json.dumps(msg, ensure_ascii=False, default=str), ex=_DATA_TTL)
        pipe.zadd(_pending_key(), {msg_id: time.time()})
        await pipe.execute()

    async def claim_next(self) -> dict[str, Any] | None:
        r = await self._get_client()
        stale_before = time.time() - _PROCESSING_TTL
        stale_ids = await r.zrangebyscore(_processing_key(), 0, stale_before)
        if stale_ids:
            pipe = r.pipeline()
            pipe.zrem(_processing_key(), *stale_ids)
            for sid in stale_ids:
                pipe.zadd(_pending_key(), {sid: time.time()})
            await pipe.execute()

        for _ in range(_MISSING_DATA_SCAN_LIMIT):
            results = await r.zpopmin(_pending_key(), 1)
            if not results:
                return None
            msg_id = str(results[0][0])
            await r.zadd(_processing_key(), {msg_id: time.time()})
            raw = await r.get(_data_key(msg_id))
            if raw is None:
                await r.zrem(_processing_key(), msg_id)
                continue
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("RedisDiagnostQueue: invalid JSON for message %s", msg_id)
                await r.zrem(_processing_key(), msg_id)
        return None

    async def ack(self, msg_id: str) -> None:
        r = await self._get_client()
        await r.zrem(_processing_key(), msg_id)
        await r.delete(_data_key(msg_id))

    async def nack(self, msg_id: str, *, error: str) -> None:
        r = await self._get_client()
        raw = await r.get(_data_key(msg_id))
        await r.zrem(_processing_key(), msg_id)
        if raw is None:
            return
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        retries = int(msg.get("_retries") or 0)
        if retries < _MAX_NACK_RETRIES:
            msg["_retries"] = retries + 1
            msg["_last_error"] = error
            backoff = min(30 * (2**retries), 600)
            pipe = r.pipeline()
            pipe.set(_data_key(msg_id), json.dumps(msg, ensure_ascii=False, default=str), ex=_DATA_TTL)
            pipe.zadd(_pending_key(), {msg_id: time.time() + backoff})
            await pipe.execute()
        else:
            logger.warning("RedisDiagnostQueue: message %s exceeded max retries, dropping. error=%s", msg_id, error)
            await r.delete(_data_key(msg_id))
