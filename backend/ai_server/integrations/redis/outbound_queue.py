from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

import redis.asyncio as aioredis

_DEFAULT_PREFIX = "ai_server:outbound"
_DATA_TTL_SECONDS = 7 * 24 * 3600
_DEDUPE_TTL_SECONDS = 7 * 24 * 3600
_LEASE_SECONDS = 300
_MAX_ATTEMPTS = 5
_RETRY_BASE_SECONDS = 30
_RETRY_MAX_SECONDS = 600

logger = logging.getLogger(__name__)


_ENQUEUE_LUA = r"""
local existing = redis.call('GET', KEYS[1])
if existing then
  local existing_body_hash = redis.call('HGET', KEYS[2], 'body_sha256')
  if existing_body_hash and existing_body_hash ~= ARGV[11] then
    return {existing, 0, 1}
  end
  return {existing, 0, 0}
end
redis.call('SET', KEYS[1], ARGV[1])
redis.call('HSET', KEYS[2],
  'delivery_id', ARGV[1],
  'dedupe_key', KEYS[1],
  'channel_id', ARGV[3],
  'recipient_id', ARGV[4],
  'body', ARGV[5],
  'body_sha256', ARGV[11],
  'task_json', ARGV[6],
  'result_json', ARGV[7],
  'status', 'pending',
  'attempts', '0',
  'created_at', ARGV[8],
  'updated_at', ARGV[8],
  'claim_token', '',
  'claimed_at', '',
  'last_error', '',
  'next_attempt_at', '')
redis.call('ZADD', KEYS[3], ARGV[10], ARGV[1])
redis.call('LPUSH', KEYS[4], ARGV[1])
redis.call('LTRIM', KEYS[4], 0, ARGV[12])
return {ARGV[1], 1, 0}
"""


_RECOVER_STALE_LUA = r"""
local recovered_unknowns = {}
local stale = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', ARGV[2], 'LIMIT', 0, 50)
for _, delivery_id in ipairs(stale) do
  local data_key = ARGV[4] .. delivery_id
  if redis.call('HGET', data_key, 'status') == 'processing' then
    redis.call('ZREM', KEYS[2], delivery_id)
    redis.call('ZADD', KEYS[1], ARGV[1], delivery_id)
    redis.call('HSET', data_key,
      'status', 'pending', 'claim_token', '', 'claimed_at', '',
      'updated_at', ARGV[3], 'last_error', 'processing_stale')
  elseif redis.call('HGET', data_key, 'status') == 'sending' then
    redis.call('ZREM', KEYS[2], delivery_id)
    redis.call('ZADD', KEYS[3], ARGV[1], delivery_id)
    redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', ARGV[5])
    redis.call('HSET', data_key,
      'status', 'unknown', 'claim_token', '', 'claimed_at', '',
      'updated_at', ARGV[3], 'last_error', 'worker_lost_during_delivery')
    local dedupe_key = redis.call('HGET', data_key, 'dedupe_key')
    redis.call('EXPIRE', data_key, ARGV[6])
    if dedupe_key then redis.call('EXPIRE', dedupe_key, ARGV[6]) end
    table.insert(recovered_unknowns, delivery_id)
  else
    redis.call('ZREM', KEYS[2], delivery_id)
  end
end
return recovered_unknowns
"""


_CLAIM_LUA = r"""
local candidates = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1], 'LIMIT', 0, 50)
for _, delivery_id in ipairs(candidates) do
  local data_key = ARGV[5] .. delivery_id
  local status = redis.call('HGET', data_key, 'status')
  if not status then
    redis.call('ZREM', KEYS[1], delivery_id)
  elseif status == 'pending' then
    local attempts = tonumber(redis.call('HGET', data_key, 'attempts') or '0')
    if attempts >= tonumber(ARGV[3]) then
      redis.call('ZREM', KEYS[1], delivery_id)
      redis.call('ZADD', KEYS[3], ARGV[1], delivery_id)
      redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', ARGV[6])
      redis.call('HSET', data_key, 'status', 'failed', 'updated_at', ARGV[4],
        'last_error', 'max_attempts_exhausted', 'next_attempt_at', '')
      local dedupe_key = redis.call('HGET', data_key, 'dedupe_key')
      redis.call('EXPIRE', data_key, ARGV[7])
      if dedupe_key then redis.call('EXPIRE', dedupe_key, ARGV[7]) end
    elseif redis.call('ZREM', KEYS[1], delivery_id) == 1 then
      redis.call('ZADD', KEYS[2], ARGV[1], delivery_id)
      redis.call('HSET', data_key,
        'status', 'processing', 'attempts', attempts + 1,
        'claim_token', ARGV[2], 'claimed_at', ARGV[4], 'updated_at', ARGV[4],
        'last_error', '', 'next_attempt_at', '')
      return {delivery_id, ARGV[2]}
    end
  else
    redis.call('ZREM', KEYS[1], delivery_id)
  end
end
return {}
"""


_TERMINAL_LUA = r"""
if redis.call('HGET', KEYS[1], 'status') ~= ARGV[8] then return 0 end
if redis.call('HGET', KEYS[1], 'claim_token') ~= ARGV[1] then return 0 end
redis.call('ZREM', KEYS[2], ARGV[2])
redis.call('ZADD', KEYS[3], ARGV[6], ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', ARGV[7])
redis.call('HSET', KEYS[1],
  'status', ARGV[3], 'updated_at', ARGV[4], 'claim_token', '', 'claimed_at', '',
  'last_error', ARGV[5], 'next_attempt_at', '')
local dedupe_key = redis.call('HGET', KEYS[1], 'dedupe_key')
redis.call('EXPIRE', KEYS[1], ARGV[9])
if dedupe_key then redis.call('EXPIRE', dedupe_key, ARGV[9]) end
return 1
"""


_RECONCILE_UNKNOWN_LUA = r"""
if redis.call('HGET', KEYS[1], 'status') ~= 'unknown' then return 0 end
redis.call('ZREM', KEYS[2], ARGV[1])
if ARGV[2] == 'confirmed_sent' then
  redis.call('ZADD', KEYS[4], ARGV[3], ARGV[1])
  redis.call('HSET', KEYS[1], 'status', 'sent', 'updated_at', ARGV[4],
    'last_error', '', 'reconciliation', ARGV[2])
  local dedupe_key = redis.call('HGET', KEYS[1], 'dedupe_key')
  redis.call('EXPIRE', KEYS[1], ARGV[5])
  if dedupe_key then redis.call('EXPIRE', dedupe_key, ARGV[5]) end
else
  redis.call('ZADD', KEYS[3], ARGV[3], ARGV[1])
  redis.call('HSET', KEYS[1], 'status', 'pending', 'updated_at', ARGV[4],
    'last_error', '', 'next_attempt_at', '', 'reconciliation', ARGV[2])
  local dedupe_key = redis.call('HGET', KEYS[1], 'dedupe_key')
  redis.call('PERSIST', KEYS[1])
  if dedupe_key then redis.call('PERSIST', dedupe_key) end
end
return 1
"""


_BEGIN_DELIVERY_LUA = r"""
if redis.call('HGET', KEYS[1], 'status') ~= 'processing' then return 0 end
if redis.call('HGET', KEYS[1], 'claim_token') ~= ARGV[1] then return 0 end
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[2])
redis.call('HSET', KEYS[1], 'status', 'sending', 'claimed_at', ARGV[4], 'updated_at', ARGV[4])
return 1
"""


_RENEW_LUA = r"""
if redis.call('HGET', KEYS[1], 'status') ~= 'processing' then return 0 end
if redis.call('HGET', KEYS[1], 'claim_token') ~= ARGV[1] then return 0 end
redis.call('ZADD', KEYS[2], ARGV[3], ARGV[2])
redis.call('HSET', KEYS[1], 'claimed_at', ARGV[4], 'updated_at', ARGV[4])
return 1
"""


_RETRY_LUA = r"""
if redis.call('HGET', KEYS[1], 'status') ~= 'processing' then return 0 end
if redis.call('HGET', KEYS[1], 'claim_token') ~= ARGV[1] then return 0 end
redis.call('ZREM', KEYS[2], ARGV[2])
redis.call('ZADD', KEYS[3], ARGV[3], ARGV[2])
redis.call('HSET', KEYS[1],
  'status', 'pending', 'updated_at', ARGV[4], 'claim_token', '', 'claimed_at', '',
  'last_error', ARGV[5], 'next_attempt_at', ARGV[6])
return 1
"""


class RedisOutboundQueue:
    """Durable, idempotent outbound delivery queue with lease-token fencing."""

    def __init__(self, redis_url: str, *, prefix: str = _DEFAULT_PREFIX) -> None:
        self.prefix = prefix.rstrip(":")
        self._client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)

    @property
    def _pending_key(self) -> str:
        return f"{self.prefix}:pending"

    @property
    def _processing_key(self) -> str:
        return f"{self.prefix}:processing"

    @property
    def _sent_key(self) -> str:
        return f"{self.prefix}:sent"

    @property
    def _failed_key(self) -> str:
        return f"{self.prefix}:failed"

    @property
    def _unknown_key(self) -> str:
        return f"{self.prefix}:unknown"

    @property
    def _recent_key(self) -> str:
        return f"{self.prefix}:recent"

    def _data_key(self, delivery_id: str) -> str:
        return f"{self.prefix}:data:{delivery_id}"

    def _dedupe_key(self, delivery_key: str) -> str:
        digest = hashlib.sha256(delivery_key.encode("utf-8")).hexdigest()
        return f"{self.prefix}:dedupe:{digest}"

    async def enqueue(
        self,
        *,
        delivery_key: str,
        channel_id: str,
        recipient_id: str,
        body: str,
        task: dict[str, Any],
        result: dict[str, Any],
    ) -> tuple[str, bool]:
        if not delivery_key or not channel_id or not recipient_id or not body:
            raise ValueError("outbound delivery requires key, channel, recipient and body")
        delivery_id = hashlib.sha256(delivery_key.encode("utf-8")).hexdigest()
        body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()
        now_iso = _now_iso()
        response = await self._client.eval(
            _ENQUEUE_LUA,
            4,
            self._dedupe_key(delivery_key),
            self._data_key(delivery_id),
            self._pending_key,
            self._recent_key,
            delivery_id,
            str(_DEDUPE_TTL_SECONDS),
            channel_id,
            recipient_id,
            body,
            json.dumps(task, ensure_ascii=False, sort_keys=True, default=str),
            json.dumps(result, ensure_ascii=False, sort_keys=True, default=str),
            now_iso,
            str(_DATA_TTL_SECONDS),
            str(_epoch_ms()),
            body_sha256,
            "199",
        )
        if len(response) > 2 and bool(int(response[2])):
            raise ValueError(f"outbound logical delivery body conflict: {delivery_id}")
        return str(response[0]), bool(int(response[1]))

    async def recover_stale(self, *, lease_seconds: int = _LEASE_SECONDS) -> list[str]:
        now_ms = _epoch_ms()
        response = await self._client.eval(
            _RECOVER_STALE_LUA,
            3,
            self._pending_key,
            self._processing_key,
            self._unknown_key,
            str(now_ms),
            str(now_ms - max(60, int(lease_seconds)) * 1000),
            _now_iso(),
            f"{self.prefix}:data:",
            str(now_ms - _DATA_TTL_SECONDS * 1000),
            str(_DATA_TTL_SECONDS),
        )
        recovered = [str(delivery_id) for delivery_id in response or []]
        if recovered:
            logger.error(
                "Recovered stale outbound delivery attempts as terminal unknown ids=%s; manual audit required",
                ",".join(recovered),
            )
        return recovered

    async def claim_next(self, *, lease_seconds: int = _LEASE_SECONDS) -> dict[str, Any] | None:
        now_ms = _epoch_ms()
        token = uuid4().hex
        response = await self._client.eval(
            _CLAIM_LUA,
            3,
            self._pending_key,
            self._processing_key,
            self._failed_key,
            str(now_ms),
            token,
            str(_MAX_ATTEMPTS),
            _now_iso(),
            f"{self.prefix}:data:",
            str(now_ms - _DATA_TTL_SECONDS * 1000),
            str(_DATA_TTL_SECONDS),
        )
        if not response:
            return None
        delivery_id = str(response[0])
        claim_token = str(response[1])
        raw = await self._client.hgetall(self._data_key(delivery_id))
        if not raw or raw.get("claim_token") != claim_token or raw.get("status") != "processing":
            return None
        return {
            **raw,
            "delivery_id": delivery_id,
            "attempts": int(raw.get("attempts") or 0),
            "task": _decode_object(raw.get("task_json") or "{}"),
            "result": _decode_object(raw.get("result_json") or "{}"),
        }

    async def mark_sent(self, delivery_id: str, *, claim_token: str) -> bool:
        return bool(
            int(
                await self._client.eval(
                    _TERMINAL_LUA,
                    3,
                    self._data_key(delivery_id),
                    self._processing_key,
                    self._sent_key,
                    claim_token,
                    delivery_id,
                    "sent",
                    _now_iso(),
                    "",
                    str(_epoch_ms()),
                    str(_epoch_ms() - _DATA_TTL_SECONDS * 1000),
                    "sending",
                    str(_DATA_TTL_SECONDS),
                )
            )
        )

    async def renew_claim(self, delivery_id: str, *, claim_token: str) -> bool:
        now_ms = _epoch_ms()
        return bool(
            int(
                await self._client.eval(
                    _RENEW_LUA,
                    2,
                    self._data_key(delivery_id),
                    self._processing_key,
                    claim_token,
                    delivery_id,
                    str(now_ms),
                    _now_iso(),
                )
            )
        )

    async def begin_delivery(self, delivery_id: str, *, claim_token: str) -> bool:
        now_ms = _epoch_ms()
        return bool(
            int(
                await self._client.eval(
                    _BEGIN_DELIVERY_LUA,
                    2,
                    self._data_key(delivery_id),
                    self._processing_key,
                    claim_token,
                    delivery_id,
                    str(now_ms),
                    _now_iso(),
                )
            )
        )

    async def mark_unknown(self, delivery_id: str, *, claim_token: str, error: str) -> bool:
        return bool(
            int(
                await self._client.eval(
                    _TERMINAL_LUA,
                    3,
                    self._data_key(delivery_id),
                    self._processing_key,
                    self._unknown_key,
                    claim_token,
                    delivery_id,
                    "unknown",
                    _now_iso(),
                    error[:1000],
                    str(_epoch_ms()),
                    str(_epoch_ms() - _DATA_TTL_SECONDS * 1000),
                    "sending",
                    str(_DATA_TTL_SECONDS),
                )
            )
        )

    async def mark_retryable_failed(self, delivery_id: str, *, claim_token: str, error: str) -> bool:
        raw = await self._client.hgetall(self._data_key(delivery_id))
        attempts = int(raw.get("attempts") or 0) if raw else 0
        if attempts >= _MAX_ATTEMPTS:
            return bool(
                int(
                    await self._client.eval(
                        _TERMINAL_LUA,
                        3,
                        self._data_key(delivery_id),
                        self._processing_key,
                        self._failed_key,
                        claim_token,
                        delivery_id,
                        "failed",
                        _now_iso(),
                        error[:1000],
                        str(_epoch_ms()),
                        str(_epoch_ms() - _DATA_TTL_SECONDS * 1000),
                        "processing",
                        str(_DATA_TTL_SECONDS),
                    )
                )
            )
        delay = min(_RETRY_MAX_SECONDS, _RETRY_BASE_SECONDS * max(1, attempts))
        next_ms = _epoch_ms() + delay * 1000
        return bool(
            int(
                await self._client.eval(
                    _RETRY_LUA,
                    3,
                    self._data_key(delivery_id),
                    self._processing_key,
                    self._pending_key,
                    claim_token,
                    delivery_id,
                    str(next_ms),
                    _now_iso(),
                    error[:1000],
                    datetime.fromtimestamp(next_ms / 1000, tz=UTC).isoformat(),
                )
            )
        )

    async def get(self, delivery_id: str) -> dict[str, Any] | None:
        raw = await self._client.hgetall(self._data_key(delivery_id))
        if not raw:
            return None
        return {
            **raw,
            "attempts": int(raw.get("attempts") or 0),
            "task": _decode_object(raw.get("task_json") or "{}"),
            "result": _decode_object(raw.get("result_json") or "{}"),
        }

    async def stats(self) -> dict[str, int]:
        pending, processing, sent, failed, unknown = (
            await self._client.pipeline(transaction=False)
            .zcard(self._pending_key)
            .zcard(self._processing_key)
            .zcard(self._sent_key)
            .zcard(self._failed_key)
            .zcard(self._unknown_key)
            .execute()
        )
        return {
            "pending": int(pending),
            "processing": int(processing),
            "sent": int(sent),
            "failed": int(failed),
            "unknown": int(unknown),
        }

    async def list_unknown(self, *, limit: int = 20) -> list[dict[str, Any]]:
        delivery_ids = await self._client.zrevrange(self._unknown_key, 0, max(0, min(limit, 100) - 1))
        if not delivery_ids:
            return []
        pipe = self._client.pipeline(transaction=False)
        for delivery_id in delivery_ids:
            pipe.hmget(
                self._data_key(str(delivery_id)),
                "channel_id",
                "created_at",
                "updated_at",
                "last_error",
                "attempts",
            )
        rows = await pipe.execute()
        return [
            {
                "delivery_id": str(delivery_id),
                "channel_id": str(row[0] or ""),
                "created_at": str(row[1] or ""),
                "updated_at": str(row[2] or ""),
                "last_error": str(row[3] or ""),
                "attempts": int(row[4] or 0),
            }
            for delivery_id, row in zip(delivery_ids, rows, strict=True)
        ]

    async def public_status(self) -> dict[str, Any]:
        return {**await self.stats(), "unknown_items": await self.list_unknown(limit=20)}

    async def reconcile_unknown(
        self,
        delivery_id: str,
        *,
        outcome: Literal["confirmed_sent", "confirmed_not_sent"],
    ) -> bool:
        """Resolve one ambiguous delivery after an operator verifies the remote outcome."""
        if outcome not in {"confirmed_sent", "confirmed_not_sent"}:
            raise ValueError("unsupported outbound reconciliation outcome")
        now_ms = _epoch_ms()
        return bool(
            int(
                await self._client.eval(
                    _RECONCILE_UNKNOWN_LUA,
                    4,
                    self._data_key(delivery_id),
                    self._unknown_key,
                    self._pending_key,
                    self._sent_key,
                    delivery_id,
                    outcome,
                    str(now_ms),
                    _now_iso(),
                    str(_DATA_TTL_SECONDS),
                )
            )
        )

    async def purge_exact(self, delivery_id: str, *, delivery_key: str) -> None:
        pipe = self._client.pipeline(transaction=True)
        pipe.zrem(self._pending_key, delivery_id)
        pipe.zrem(self._processing_key, delivery_id)
        pipe.zrem(self._sent_key, delivery_id)
        pipe.zrem(self._failed_key, delivery_id)
        pipe.zrem(self._unknown_key, delivery_id)
        pipe.lrem(self._recent_key, 0, delivery_id)
        pipe.delete(self._data_key(delivery_id))
        pipe.delete(self._dedupe_key(delivery_key))
        await pipe.execute()

    async def close(self) -> None:
        await self._client.aclose()


def outbound_delivery_key(*, channel_id: str, recipient_id: str, task_id: str, body: str = "") -> str:
    del body
    return f"{channel_id}:{recipient_id}:{task_id}:result"


def _decode_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _epoch_ms() -> int:
    return int(time.time() * 1000)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
