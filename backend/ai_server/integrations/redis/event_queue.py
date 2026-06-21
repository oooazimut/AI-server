from __future__ import annotations

import json
import logging
import time
from collections.abc import Collection
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

_PREFIX = "ai_server:events"
_COUNTER_KEY = f"{_PREFIX}:counter"
_PENDING_KEY = f"{_PREFIX}:pending"  # sorted set, score = next_attempt_at epoch ms
_PROCESSING_KEY = f"{_PREFIX}:processing"  # sorted set, score = started_at epoch ms
_FAILED_KEY = f"{_PREFIX}:failed"  # set of failed event_ids
_RECENT_KEY = f"{_PREFIX}:recent"  # list of recent event_ids (newest first)
_DEDUPE_TTL = 48 * 3600  # dedupe key expiry in seconds
_RECENT_MAX = 200  # keep last N entries in recent list
_STALE_SECONDS = 120
_MAX_ATTEMPTS = 5
_RETRY_BASE = 30
_RETRY_MAX = 600


def _data_key(event_id: int) -> str:
    return f"{_PREFIX}:data:{event_id}"


def _dedupe_key(event_key: str) -> str:
    return f"{_PREFIX}:dedupe:{event_key}"


def _epoch_ms() -> float:
    return time.time() * 1000


class RedisEventQueue:
    """Реализует WebhookEnqueuePort и WebhookConsumePort структурно (без явного наследования)."""

    def __init__(self, redis_url: str) -> None:
        self._client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)

    async def enqueue(
        self,
        payload: dict[str, Any],
        *,
        event_type: str,
        dedupe_key: str | None = None,
    ) -> tuple[int, bool]:
        from ai_server.integrations.webhook_utils import sanitize_webhook_payload, webhook_event_key

        safe_payload = sanitize_webhook_payload(payload)
        received_at = _iso_now()
        if dedupe_key is None:
            dedupe_key = webhook_event_key(safe_payload, event_type=event_type, received_at=received_at)

        dk = _dedupe_key(dedupe_key)

        existing = await self._client.get(dk)
        if existing is not None:
            return int(existing), False

        event_id: int = await self._client.incr(_COUNTER_KEY)

        ok = await self._client.set(dk, str(event_id), nx=True, ex=_DEDUPE_TTL)
        if not ok:
            existing = await self._client.get(dk)
            return int(existing or event_id), False

        payload_json = json.dumps(safe_payload, ensure_ascii=False, sort_keys=True, default=str)
        now_ms = _epoch_ms()
        data = {
            "id": str(event_id),
            "event_key": dedupe_key,
            "event_type": event_type,
            "payload_json": payload_json,
            "status": "pending",
            "attempts": "0",
            "received_at": received_at,
            "started_at": "",
            "processed_at": "",
            "next_attempt_at": "",
            "last_error": "",
            "last_result_json": "",
        }
        pipe = self._client.pipeline(transaction=True)
        pipe.hset(_data_key(event_id), mapping=data)
        pipe.zadd(_PENDING_KEY, {str(event_id): now_ms})
        pipe.lpush(_RECENT_KEY, str(event_id))
        pipe.ltrim(_RECENT_KEY, 0, _RECENT_MAX - 1)
        await pipe.execute()

        return event_id, True

    async def claim_next(
        self,
        *,
        blocked_partition_keys: Collection[str] | None = None,
    ) -> dict[str, Any] | None:
        from ai_server.integrations.webhook_utils import webhook_event_partition_key

        blocked = set(blocked_partition_keys or ())
        now_ms = _epoch_ms()
        stale_before_ms = now_ms - _STALE_SECONDS * 1000

        stale_ids = await self._client.zrangebyscore(_PROCESSING_KEY, "-inf", stale_before_ms)
        for sid in stale_ids:
            pipe = self._client.pipeline(transaction=True)
            pipe.zrem(_PROCESSING_KEY, sid)
            pipe.zadd(_PENDING_KEY, {sid: now_ms})
            pipe.hset(_data_key(int(sid)), mapping={"status": "pending", "last_error": "processing_stale"})
            await pipe.execute()

        candidates = await self._client.zrangebyscore(_PENDING_KEY, "-inf", now_ms, start=0, num=50)
        for sid in candidates:
            event_id = int(sid)
            raw = await self._client.hgetall(_data_key(event_id))
            if not raw:
                await self._client.zrem(_PENDING_KEY, sid)
                continue
            attempts = int(raw.get("attempts") or 0)
            if attempts >= _MAX_ATTEMPTS:
                pipe = self._client.pipeline(transaction=True)
                pipe.zrem(_PENDING_KEY, sid)
                pipe.sadd(_FAILED_KEY, sid)
                pipe.hset(_data_key(event_id), "status", "failed")
                await pipe.execute()
                continue
            payload = _decode_json(raw.get("payload_json") or "{}")
            event_type = raw.get("event_type") or ""
            partition_key = webhook_event_partition_key(payload, event_type=event_type)
            if partition_key in blocked:
                continue
            new_attempts = attempts + 1
            started_at = _iso_now()
            pipe = self._client.pipeline(transaction=True)
            pipe.zrem(_PENDING_KEY, sid)
            pipe.zadd(_PROCESSING_KEY, {sid: now_ms})
            pipe.hset(
                _data_key(event_id),
                mapping={
                    "status": "processing",
                    "attempts": str(new_attempts),
                    "started_at": started_at,
                    "last_error": "",
                },
            )
            await pipe.execute()
            event = dict(raw)
            event["id"] = event_id
            event["attempts"] = new_attempts
            event["payload"] = payload
            event["partition_key"] = partition_key
            return event

        return None

    async def mark_done(self, event_id: int, result: dict[str, Any]) -> None:
        processed_at = _iso_now()
        result_json = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
        pipe = self._client.pipeline(transaction=True)
        pipe.zrem(_PROCESSING_KEY, str(event_id))
        pipe.hset(
            _data_key(event_id),
            mapping={
                "status": "done",
                "processed_at": processed_at,
                "next_attempt_at": "",
                "last_error": "",
                "last_result_json": result_json,
            },
        )
        await pipe.execute()

    async def mark_failed(self, event_id: int, error: str) -> None:
        raw = await self._client.hgetall(_data_key(event_id))
        attempts = int(raw.get("attempts") or 0) if raw else 0
        now_ms = _epoch_ms()

        if attempts >= _MAX_ATTEMPTS:
            pipe = self._client.pipeline(transaction=True)
            pipe.zrem(_PROCESSING_KEY, str(event_id))
            pipe.sadd(_FAILED_KEY, str(event_id))
            pipe.hset(
                _data_key(event_id),
                mapping={"status": "failed", "last_error": error[:1000], "next_attempt_at": ""},
            )
            await pipe.execute()
        else:
            delay_s = min(_RETRY_MAX, _RETRY_BASE * max(1, attempts))
            next_ms = now_ms + delay_s * 1000
            next_attempt_at = _iso_from_ms(next_ms)
            pipe = self._client.pipeline(transaction=True)
            pipe.zrem(_PROCESSING_KEY, str(event_id))
            pipe.zadd(_PENDING_KEY, {str(event_id): next_ms})
            pipe.hset(
                _data_key(event_id),
                mapping={
                    "status": "pending",
                    "last_error": error[:1000],
                    "next_attempt_at": next_attempt_at,
                },
            )
            await pipe.execute()

    async def stats(self) -> dict[str, Any]:
        pending = await self._client.zcard(_PENDING_KEY)
        processing = await self._client.zcard(_PROCESSING_KEY)
        failed = await self._client.scard(_FAILED_KEY)
        recent_ids = await self._client.lrange(_RECENT_KEY, 0, 0)
        latest = None
        if recent_ids:
            raw = await self._client.hgetall(_data_key(int(recent_ids[0])))
            if raw:
                latest = {
                    "id": raw.get("id"),
                    "event_type": raw.get("event_type"),
                    "status": raw.get("status"),
                    "attempts": raw.get("attempts"),
                    "received_at": raw.get("received_at"),
                    "processed_at": raw.get("processed_at"),
                    "last_error": raw.get("last_error"),
                }
        return {
            "backend": "redis",
            "pending": pending,
            "processing": processing,
            "done": None,
            "failed": failed,
            "latest": latest,
        }

    async def latest(self, *, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        ids = await self._client.lrange(_RECENT_KEY, 0, limit - 1)
        result = []
        for sid in ids:
            raw = await self._client.hgetall(_data_key(int(sid)))
            if raw:
                result.append(
                    {
                        "id": raw.get("id"),
                        "event_type": raw.get("event_type"),
                        "status": raw.get("status"),
                        "attempts": raw.get("attempts"),
                        "received_at": raw.get("received_at"),
                        "started_at": raw.get("started_at"),
                        "processed_at": raw.get("processed_at"),
                        "next_attempt_at": raw.get("next_attempt_at"),
                        "last_error": raw.get("last_error"),
                    }
                )
        return result

    async def close(self) -> None:
        await self._client.aclose()


def _iso_now() -> str:
    from ai_server.utils import MOSCOW_TZ

    return datetime.now(MOSCOW_TZ).isoformat()


def _iso_from_ms(epoch_ms: float) -> str:
    from ai_server.utils import MOSCOW_TZ

    return datetime.fromtimestamp(epoch_ms / 1000, tz=MOSCOW_TZ).isoformat()


def _decode_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
