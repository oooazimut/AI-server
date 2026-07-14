from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import redis.asyncio as aioredis

from ai_server.models import AgentTask
from ai_server.settings import Settings

_PREFIX = "ai_server:dialog_guard"


class RedisDialogGuard:
    """Short-lived Redis state for stuck Bitrix dialog handling."""

    def __init__(self, redis_url: str, *, settings: Settings) -> None:
        self.enabled = settings.bitrix_dialog_guard_enabled
        self._client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)

    async def current_generation(self, dialog_key: str) -> int:
        if not self.enabled:
            return 0
        raw = await self._client.get(_generation_key(dialog_key))
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    async def increment_generation(self, dialog_key: str) -> int:
        if not self.enabled:
            return 0
        value = await self._client.incr(_generation_key(dialog_key))
        await self._client.expire(_generation_key(dialog_key), 24 * 3600)
        return int(value)

    async def mark_active(self, task: AgentTask, *, ttl_seconds: int) -> int:
        dialog_key = _task_dialog_key(task)
        if not self.enabled or not dialog_key:
            return 0
        generation = await self.current_generation(dialog_key)
        payload = {
            "dialog_key": dialog_key,
            "task_id": task.task_id,
            "request": task.request,
            "started_at": time.time(),
            "generation": generation,
        }
        await self._client.set(_active_key(dialog_key), json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)
        return generation

    async def clear_active(self, task: AgentTask) -> None:
        dialog_key = _task_dialog_key(task)
        if not self.enabled or not dialog_key:
            return
        raw = await self._client.get(_active_key(dialog_key))
        if raw is None:
            return
        active = _decode(raw)
        if str(active.get("task_id") or "") == task.task_id:
            await self._client.delete(_active_key(dialog_key))

    async def get_active(self, dialog_key: str) -> dict[str, Any] | None:
        if not self.enabled or not dialog_key:
            return None
        raw = await self._client.get(_active_key(dialog_key))
        if raw is None:
            return None
        active = _decode(raw)
        if not active:
            return None
        active["age_seconds"] = max(0.0, time.time() - float(active.get("started_at") or 0))
        return active

    async def save_pending(self, task: AgentTask, *, ttl_seconds: int) -> None:
        dialog_key = _task_dialog_key(task)
        if not self.enabled or not dialog_key:
            return
        payload = {
            "dialog_key": dialog_key,
            "created_at": time.time(),
            "task": task.model_dump(),
        }
        await self._client.set(_pending_key(dialog_key), json.dumps(payload, ensure_ascii=False), ex=ttl_seconds)

    async def get_pending(self, dialog_key: str) -> AgentTask | None:
        if not self.enabled or not dialog_key:
            return None
        raw = await self._client.get(_pending_key(dialog_key))
        if raw is None:
            return None
        payload = _decode(raw)
        task_payload = payload.get("task")
        if not isinstance(task_payload, dict):
            return None
        return AgentTask.model_validate(task_payload)

    async def pop_pending(self, dialog_key: str) -> AgentTask | None:
        task = await self.get_pending(dialog_key)
        if self.enabled and dialog_key:
            await self._client.delete(_pending_key(dialog_key))
        return task

    async def task_is_stale(self, task: AgentTask) -> bool:
        if not self.enabled:
            return False
        dialog_key = _task_dialog_key(task)
        if not dialog_key:
            return False
        current = await self.current_generation(dialog_key)
        try:
            task_generation = int(task.context.get("dialog_cancel_generation") or 0)
        except (TypeError, ValueError):
            task_generation = 0
        return task_generation < current


def _task_dialog_key(task: AgentTask) -> str:
    return str((task.context or {}).get("dialog_key") or "").strip()


def _active_key(dialog_key: str) -> str:
    return f"{_PREFIX}:active:{_hash_key(dialog_key)}"


def _pending_key(dialog_key: str) -> str:
    return f"{_PREFIX}:pending:{_hash_key(dialog_key)}"


def _generation_key(dialog_key: str) -> str:
    return f"{_PREFIX}:generation:{_hash_key(dialog_key)}"


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _decode(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
