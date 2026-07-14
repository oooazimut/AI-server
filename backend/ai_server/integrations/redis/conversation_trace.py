from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis

from ai_server.integrations.bitrix.events import MESSAGE_EVENTS, parse_incoming_message
from ai_server.integrations.webhook_utils import sanitize_webhook_payload
from ai_server.models import AgentResult, AgentTask
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)

_PREFIX = "ai_server:conversation_trace"
_COUNTER_KEY = f"{_PREFIX}:counter"
_RECENT_KEY = f"{_PREFIX}:recent"


class RedisConversationTrace:
    """Short-lived, read-only diagnostic trace for Bitrix conversations."""

    def __init__(self, redis_url: str, *, settings: Settings) -> None:
        self.enabled = settings.conversation_trace_enabled
        self.ttl_seconds = max(3600, int(settings.conversation_trace_ttl_hours or 48) * 3600)
        self.max_text_chars = max(1000, int(settings.conversation_trace_max_text_chars or 8000))
        self._client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)

    async def record_inbound(
        self,
        *,
        event_id: int,
        event_type: str,
        payload: dict[str, Any],
        inserted: bool,
    ) -> None:
        if not self.enabled:
            return
        event: dict[str, Any] = {
            "trace_type": "inbound_message",
            "event_id": event_id,
            "event_type": event_type,
            "duplicate": not inserted,
        }
        if str(event_type or "").upper() in MESSAGE_EVENTS:
            try:
                incoming = parse_incoming_message(payload)
                event.update(
                    {
                        "message_id": incoming.message_id,
                        "user_id": incoming.user_id,
                        "user_name": _payload_user_name(payload),
                        "dialog_id": incoming.dialog_id,
                        "chat_id": incoming.chat_id,
                        "text": incoming.text,
                    }
                )
            except Exception:
                logger.debug("ConversationTrace: failed to parse inbound message", exc_info=True)
        await self.record(event)

    async def record_route(
        self,
        *,
        event_id: int | None,
        event_type: str,
        routed_to: str,
        task: AgentTask | None = None,
        partition_key: str = "",
        result: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        event: dict[str, Any] = {
            "trace_type": "route_event",
            "event_id": event_id,
            "event_type": event_type,
            "routed_to": routed_to,
            "partition_key": partition_key,
            "result": result or {},
        }
        if task is not None:
            event.update(_task_trace_fields(task))
        await self.record(event)

    async def record_agent_result(self, *, task: AgentTask, result: AgentResult, source: str) -> None:
        if not self.enabled:
            return
        event = {
            "trace_type": "agent_result",
            "source": source,
            **_task_trace_fields(task),
            "agent_id": result.agent_id,
            "status": result.status,
            "answer": result.answer,
            "handoff_to": result.handoff_to,
            "confidence": result.confidence,
            "actions": [action.model_dump() for action in result.actions_taken],
            "actions_requiring_approval": [action.model_dump() for action in result.actions_requiring_approval],
            "model_usage": [usage.model_dump() for usage in result.model_usage],
            "metadata": result.metadata,
        }
        await self.record(event)

    async def record_outbound(
        self,
        *,
        task: AgentTask,
        result: AgentResult,
        recipient_id: str,
        body: str,
        status: str,
        error: str = "",
    ) -> None:
        if not self.enabled:
            return
        await self.record(
            {
                "trace_type": "outbound_message",
                **_task_trace_fields(task),
                "agent_id": result.agent_id,
                "result_status": result.status,
                "recipient_id": recipient_id,
                "body": body,
                "send_status": status,
                "send_error": error,
            }
        )

    async def record_timing(
        self,
        *,
        task: AgentTask,
        component: str,
        stage: str,
        elapsed_ms: float,
        started_at: str = "",
        status: str = "",
        step: int | None = None,
        tool: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        event: dict[str, Any] = {
            "trace_type": "timing_step",
            **_task_trace_fields(task),
            "component": component,
            "stage": stage,
            "elapsed_ms": round(float(elapsed_ms), 1),
            "started_at": started_at,
            "status": status,
            "tool": tool,
            "details": details or {},
        }
        if step is not None:
            event["step"] = step
        await self.record(event)

    async def record(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            await self._record(event)
        except Exception:
            logger.exception("ConversationTrace: failed to record event")

    async def _record(self, event: dict[str, Any]) -> None:
        now_ms = _epoch_ms()
        trace_id = int(await self._client.incr(_COUNTER_KEY))
        compact = _compact_event(
            {
                **event,
                "id": trace_id,
                "created_at": _iso_now(),
            },
            max_text_chars=self.max_text_chars,
        )
        payload_json = json.dumps(sanitize_webhook_payload(compact), ensure_ascii=False, sort_keys=True, default=str)
        data_key = _data_key(trace_id)
        pipe = self._client.pipeline(transaction=True)
        pipe.set(data_key, payload_json, ex=self.ttl_seconds)
        pipe.zadd(_RECENT_KEY, {str(trace_id): now_ms})
        pipe.expire(_RECENT_KEY, self.ttl_seconds)
        for index_key in _index_keys(compact):
            pipe.zadd(index_key, {str(trace_id): now_ms})
            pipe.expire(index_key, self.ttl_seconds)
        await pipe.execute()
        await self._trim_older_than(now_ms - self.ttl_seconds * 1000)

    async def recent(self, *, limit: int = 100, hours: int = 24) -> list[dict[str, Any]]:
        return await self._events_from_zset(_RECENT_KEY, limit=limit, hours=hours)

    async def by_user(self, user_id: str | int, *, limit: int = 100, hours: int = 24) -> list[dict[str, Any]]:
        return await self._events_from_zset(_user_key(user_id), limit=limit, hours=hours)

    async def by_dialog(self, dialog_key: str, *, limit: int = 100, hours: int = 24) -> list[dict[str, Any]]:
        return await self._events_from_zset(_dialog_key(dialog_key), limit=limit, hours=hours)

    async def by_message(self, message_id: str | int, *, limit: int = 50, hours: int = 48) -> list[dict[str, Any]]:
        return await self._events_from_zset(_message_key(message_id), limit=limit, hours=hours)

    async def by_task(self, task_id: str, *, limit: int = 100, hours: int = 48) -> list[dict[str, Any]]:
        return await self._events_from_zset(_task_key(task_id), limit=limit, hours=hours)

    async def _events_from_zset(self, key: str, *, limit: int, hours: int) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        limit = max(1, min(int(limit or 100), 500))
        hours = max(1, min(int(hours or 24), max(1, self.ttl_seconds // 3600)))
        min_score = _epoch_ms() - hours * 3600 * 1000
        ids = await self._client.zrevrangebyscore(key, "+inf", min_score, start=0, num=limit)
        result: list[dict[str, Any]] = []
        for trace_id in ids:
            raw = await self._client.get(_data_key(int(trace_id)))
            if raw is None:
                continue
            parsed = _decode_json(raw)
            if parsed:
                result.append(parsed)
        return result

    async def _trim_older_than(self, min_score: float) -> None:
        await self._client.zremrangebyscore(_RECENT_KEY, "-inf", min_score)


def _task_trace_fields(task: AgentTask) -> dict[str, Any]:
    context = task.context or {}
    user = task.user
    return {
        "task_id": task.task_id,
        "request": task.request,
        "user_id": user.id if user else None,
        "channel": user.channel if user else "",
        "dialog_key": context.get("dialog_key") or "",
        "base_dialog_key": context.get("base_dialog_key") or "",
        "dialog_line_id": context.get("dialog_line_id") or "",
        "dialog_line_label": context.get("dialog_line_label") or "",
        "dialog_auto_line": bool(context.get("dialog_auto_line")),
        "dialog_id": context.get("dialog_id") or "",
        "recipient_id": context.get("recipient_id") or "",
        "message_id": (user.raw or {}).get("message_id") if user else None,
        "chat_id": (user.raw or {}).get("chat_id") if user else None,
    }


def _payload_user_name(payload: dict[str, Any]) -> str:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    user = data.get("user") if isinstance(data.get("user"), dict) else data.get("USER")
    if not isinstance(user, dict):
        return ""
    return str(
        user.get("name")
        or user.get("NAME")
        or " ".join(str(part).strip() for part in (user.get("firstName"), user.get("lastName")) if part)
        or " ".join(str(part).strip() for part in (user.get("FIRST_NAME"), user.get("LAST_NAME")) if part)
    ).strip()


def _compact_event(value: Any, *, max_text_chars: int) -> Any:
    if isinstance(value, dict):
        return {str(key): _compact_event(item, max_text_chars=max_text_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_event(item, max_text_chars=max(1000, max_text_chars // 2)) for item in value]
    if isinstance(value, str):
        return _truncate(value, max_text_chars)
    return value


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 24)] + "\n...[truncated]"


def _index_keys(event: dict[str, Any]) -> list[str]:
    keys = []
    if event.get("user_id") not in (None, ""):
        keys.append(_user_key(event["user_id"]))
    if event.get("dialog_key"):
        keys.append(_dialog_key(str(event["dialog_key"])))
    if event.get("dialog_id"):
        keys.append(_dialog_key(str(event["dialog_id"])))
    if event.get("message_id") not in (None, ""):
        keys.append(_message_key(event["message_id"]))
    if event.get("task_id"):
        keys.append(_task_key(str(event["task_id"])))
    return keys


def _data_key(trace_id: int) -> str:
    return f"{_PREFIX}:data:{trace_id}"


def _user_key(user_id: str | int) -> str:
    return f"{_PREFIX}:user:{user_id}"


def _message_key(message_id: str | int) -> str:
    return f"{_PREFIX}:message:{message_id}"


def _task_key(task_id: str) -> str:
    return f"{_PREFIX}:task:{_hash_key(task_id)}"


def _dialog_key(dialog_key: str) -> str:
    return f"{_PREFIX}:dialog:{_hash_key(dialog_key)}"


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _epoch_ms() -> float:
    return time.time() * 1000


def _iso_now() -> str:
    return datetime.now(MOSCOW_TZ).isoformat()


def _decode_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}
