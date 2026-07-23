from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio

from ai_server.agent_queue_utils import agent_queue_partition_key
from ai_server.integrations.memory.agent_queue import InMemoryAgentQueue
from ai_server.integrations.redis.agent_queue import RedisAgentQueue
from ai_server.orchestrators.internal import OrchestratorTransportRuntime


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


def test_redis_agent_queue_removes_pending_by_partition():
    queue = RedisAgentQueue("redis://localhost/0")
    client = MagicMock()
    client.zrange = AsyncMock(return_value=["m1", "m2"])
    client.get = AsyncMock(
        side_effect=[
            json.dumps(_message("m1", "chat:1:user:1"), ensure_ascii=False),
            json.dumps(_message("m2", "chat:2:user:2"), ensure_ascii=False),
        ]
    )
    client.zrem = AsyncMock(return_value=1)
    client.delete = AsyncMock()
    queue._client = client

    result = _run(queue.remove_pending_by_partition("orchestrator", "dialog:chat:1:user:1"))

    assert result == 1
    client.zrem.assert_awaited_once_with("ai_server:aq:orchestrator:pending", "m1")
    client.delete.assert_awaited_once_with("ai_server:aq:data:m1")


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


def test_memory_agent_queue_removes_pending_by_partition():
    async def _impl() -> tuple[int, dict | None]:
        queue = InMemoryAgentQueue()
        await queue.publish(_message("m1", "chat:1:user:1"))
        await queue.publish(_message("m2", "chat:2:user:2"))
        removed = await queue.remove_pending_by_partition("orchestrator", "dialog:chat:1:user:1")
        remaining = await queue.claim_next("orchestrator")
        return removed, remaining

    removed, remaining = anyio.run(_impl)

    assert removed == 1
    assert remaining is not None
    assert remaining["id"] == "m2"


def test_five_orchestrator_workers_can_claim_five_numbered_branches():
    async def _impl() -> list[str]:
        queue = InMemoryAgentQueue()
        orchestrator = OrchestratorTransportRuntime(
            SimpleNamespace(id="internal_orchestrator"),
            agent_tools=[],
            llm=None,
        )
        for number in range(101, 106):
            await queue.publish(_message(f"m{number}", f"chat:1:user:1:day:20260722:conversation:{number}"))
        claims = await asyncio.gather(*(orchestrator._claim_queue_message(queue, "orchestrator") for _ in range(5)))
        return [partition for message, partition in claims if message is not None]

    partitions = anyio.run(_impl)

    assert len(partitions) == 5
    assert len(set(partitions)) == 5


def test_same_numbered_branch_remains_serialized_while_active():
    async def _impl() -> tuple[dict | None, dict | None]:
        queue = InMemoryAgentQueue()
        orchestrator = OrchestratorTransportRuntime(
            SimpleNamespace(id="internal_orchestrator"),
            agent_tools=[],
            llm=None,
        )
        branch = "chat:1:user:1:day:20260722:conversation:101"
        await queue.publish(_message("m1", branch))
        await queue.publish(_message("m2", branch))
        first, _ = await orchestrator._claim_queue_message(queue, "orchestrator")
        second, _ = await orchestrator._claim_queue_message(queue, "orchestrator")
        return first, second

    first, second = anyio.run(_impl)

    assert first is not None
    assert second is None
