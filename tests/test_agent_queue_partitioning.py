from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import anyio

from ai_server.agent_queue_utils import agent_queue_partition_key
from ai_server.integrations.memory.agent_queue import InMemoryAgentQueue
from ai_server.integrations.redis.agent_queue import RedisAgentQueue


def _run(coro):
    async def _runner():
        return await coro

    return anyio.run(_runner)


def _message(message_id: str, dialog_key: str, *, to: str = "orchestrator") -> dict:
    return {
        "id": message_id,
        "to": to,
        "type": "task",
        "payload": {
            "task_id": f"task-{message_id}",
            "request": "Битрикс покажи склад Борисов",
            "context": {
                "dialog_key": dialog_key,
                "dialog_id": dialog_key.split(":")[1],
                "channel_id": "bitrix24",
            },
        },
    }


def test_agent_queue_partition_key_prefers_dialog_key():
    message = {
        "payload": {
            "task_id": "task-1",
            "user": {"id": "27"},
            "context": {"dialog_key": "chat:3669:user:27", "dialog_id": "3669"},
        }
    }

    assert agent_queue_partition_key(message) == "dialog:chat:3669:user:27"


def test_agent_queue_partition_key_falls_back_to_dialog_id_and_user():
    message = {
        "payload": {
            "task_id": "task-1",
            "user": {"id": "27"},
            "context": {"dialog_id": "3669"},
        }
    }

    assert agent_queue_partition_key(message) == "dialog_id:3669:user:27"


def test_redis_agent_queue_skips_blocked_dialog_partition():
    queue = RedisAgentQueue("redis://localhost/0")
    client = MagicMock()
    client.zrangebyscore = AsyncMock(return_value=[])
    client.zrange = AsyncMock(return_value=["m1", "m2"])
    client.get = AsyncMock(
        side_effect=[
            json.dumps(_message("m1", "chat:1:user:1"), ensure_ascii=False),
            json.dumps(_message("m2", "chat:2:user:2"), ensure_ascii=False),
        ]
    )
    client.zrem = AsyncMock(return_value=1)
    client.zadd = AsyncMock()
    queue._client = client

    result = _run(queue.claim_next("orchestrator", blocked_partition_keys={"dialog:chat:1:user:1"}))

    assert result is not None
    assert result["id"] == "m2"
    assert result["_partition_key"] == "dialog:chat:2:user:2"
    client.zrem.assert_awaited_once_with("ai_server:aq:orchestrator:pending", "m2")
    client.zadd.assert_awaited_once()


def test_redis_agent_queue_returns_none_when_all_candidates_blocked():
    queue = RedisAgentQueue("redis://localhost/0")
    client = MagicMock()
    client.zrangebyscore = AsyncMock(return_value=[])
    client.zrange = AsyncMock(return_value=["m1"])
    client.get = AsyncMock(return_value=json.dumps(_message("m1", "chat:1:user:1"), ensure_ascii=False))
    client.zrem = AsyncMock()
    client.zadd = AsyncMock()
    queue._client = client

    result = _run(queue.claim_next("orchestrator", blocked_partition_keys={"dialog:chat:1:user:1"}))

    assert result is None
    client.zrem.assert_not_awaited()
    client.zadd.assert_not_awaited()


def test_redis_agent_queue_reports_active_partition_keys():
    queue = RedisAgentQueue("redis://localhost/0")
    client = MagicMock()
    client.zrange = AsyncMock(side_effect=[["m1"], ["m2"]])
    client.get = AsyncMock(
        side_effect=[
            json.dumps(_message("m1", "chat:1:user:1"), ensure_ascii=False),
            json.dumps(_message("m2", "chat:2:user:2"), ensure_ascii=False),
        ]
    )
    queue._client = client

    result = _run(queue.active_partition_keys("orchestrator"))

    assert result == {"dialog:chat:1:user:1", "dialog:chat:2:user:2"}


def test_memory_agent_queue_skips_blocked_dialog_partition():
    async def _impl() -> dict | None:
        queue = InMemoryAgentQueue()
        await queue.publish(_message("m1", "chat:1:user:1"))
        await queue.publish(_message("m2", "chat:2:user:2"))
        return await queue.claim_next("orchestrator", blocked_partition_keys={"dialog:chat:1:user:1"})

    result = anyio.run(_impl)

    assert result is not None
    assert result["id"] == "m2"
    assert result["_partition_key"] == "dialog:chat:2:user:2"


def test_memory_agent_queue_reports_active_partition_keys():
    async def _impl() -> set[str]:
        queue = InMemoryAgentQueue()
        await queue.publish(_message("m1", "chat:1:user:1"))
        await queue.publish(_message("m2", "chat:2:user:2"))
        return await queue.active_partition_keys("orchestrator")

    result = anyio.run(_impl)

    assert result == {"dialog:chat:1:user:1", "dialog:chat:2:user:2"}
