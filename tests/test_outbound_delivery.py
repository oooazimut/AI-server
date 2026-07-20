from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio

from ai_server.integrations.redis.outbound_queue import RedisOutboundQueue, outbound_delivery_key
from ai_server.models import AgentResult, AgentTask
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.workers.orchestrator.outbound_delivery import deliver_outbound_once


def run(coro):
    async def _runner():
        return await coro

    return anyio.run(_runner)


def test_outbound_delivery_key_is_task_scoped_and_body_stable() -> None:
    first = outbound_delivery_key(channel_id="bitrix24", recipient_id="13", task_id="t1", body="ok")
    same = outbound_delivery_key(channel_id="bitrix24", recipient_id="13", task_id="t1", body="ok")
    changed_body = outbound_delivery_key(channel_id="bitrix24", recipient_id="13", task_id="t1", body="changed")
    other = outbound_delivery_key(channel_id="bitrix24", recipient_id="13", task_id="t2", body="ok")
    assert first == same
    assert first == changed_body
    assert first != other


def test_redis_outbound_enqueue_is_one_atomic_eval() -> None:
    queue = RedisOutboundQueue("redis://localhost/0")
    queue._client = MagicMock()
    queue._client.eval = AsyncMock(return_value=["delivery-id", 1])

    delivery_id, created = run(
        queue.enqueue(
            delivery_key="key",
            channel_id="bitrix24",
            recipient_id="13",
            body="hello",
            task={"task_id": "t1"},
            result={"status": "completed"},
        )
    )

    assert (delivery_id, created) == ("delivery-id", True)
    queue._client.eval.assert_awaited_once()
    assert queue._client.eval.await_args.args[1] == 4


def test_redis_outbound_enqueue_rejects_changed_body_for_same_logical_id() -> None:
    queue = RedisOutboundQueue("redis://localhost/0")
    queue._client = MagicMock()
    queue._client.eval = AsyncMock(return_value=["delivery-id", 0, 1])

    try:
        run(
            queue.enqueue(
                delivery_key="stable-key",
                channel_id="bitrix24",
                recipient_id="13",
                body="changed",
                task={"task_id": "t1"},
                result={"status": "completed"},
            )
        )
    except ValueError as exc:
        assert "body conflict" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("body conflict must stop a duplicate logical delivery")


def test_redis_outbound_claim_returns_fencing_token() -> None:
    queue = RedisOutboundQueue("redis://localhost/0", prefix="test:outbound")
    queue._client = MagicMock()
    queue._client.eval = AsyncMock(return_value=["abc", "token-1"])
    queue._client.hgetall = AsyncMock(
        return_value={
            "delivery_id": "abc",
            "status": "processing",
            "claim_token": "token-1",
            "attempts": "1",
            "task_json": "{}",
            "result_json": "{}",
        }
    )

    claimed = run(queue.claim_next())

    assert claimed is not None
    assert claimed["delivery_id"] == "abc"
    assert claimed["claim_token"] == "token-1"
    assert claimed["attempts"] == 1


class FakeQueue:
    def __init__(self, delivery: dict | None, *, begin_ok: bool = True) -> None:
        self.delivery = delivery
        self.begin_ok = begin_ok
        self.sent: list[tuple[str, str]] = []
        self.unknown: list[tuple[str, str, str]] = []
        self.retry: list[tuple[str, str, str]] = []
        self.renewed: list[tuple[str, str]] = []
        self.begun: list[tuple[str, str]] = []

    async def claim_next(self):
        value, self.delivery = self.delivery, None
        return value

    async def mark_sent(self, delivery_id: str, *, claim_token: str):
        self.sent.append((delivery_id, claim_token))
        return True

    async def renew_claim(self, delivery_id: str, *, claim_token: str):
        self.renewed.append((delivery_id, claim_token))
        return True

    async def begin_delivery(self, delivery_id: str, *, claim_token: str):
        self.begun.append((delivery_id, claim_token))
        return self.begin_ok

    async def mark_unknown(self, delivery_id: str, *, claim_token: str, error: str):
        self.unknown.append((delivery_id, claim_token, error))
        return True

    async def mark_retryable_failed(self, delivery_id: str, *, claim_token: str, error: str):
        self.retry.append((delivery_id, claim_token, error))
        return True


def delivery() -> dict:
    task = AgentTask(task_id="t1", request="hello", context={"dialog_key": "d1"})
    result = AgentResult(status="completed", agent_id="orchestrator", answer="hello")
    return {
        "delivery_id": "d1",
        "claim_token": "c1",
        "channel_id": "bitrix24",
        "recipient_id": "13",
        "body": "hello",
        "task": task.model_dump(),
        "result": result.model_dump(),
    }


def test_delivery_worker_marks_sent_after_transport_success() -> None:
    queue = FakeQueue(delivery())
    channel = MagicMock()
    channel.send = AsyncMock(return_value=None)

    outcome = run(deliver_outbound_once(queue, channels={"bitrix24": channel}))

    assert outcome == {"delivery_id": "d1", "status": "sent", "marked": True}
    assert queue.sent == [("d1", "c1")]
    assert queue.renewed == [("d1", "c1")]
    assert queue.begun == [("d1", "c1")]
    assert not queue.unknown


def test_delivery_worker_fences_ambiguous_transport_as_unknown_without_retry() -> None:
    queue = FakeQueue(delivery())
    channel = MagicMock()
    channel.send = AsyncMock(side_effect=TimeoutError("remote outcome ambiguous"))

    outcome = run(deliver_outbound_once(queue, channels={"bitrix24": channel}))

    assert outcome == {"delivery_id": "d1", "status": "unknown", "marked": True}
    assert len(queue.unknown) == 1
    assert not queue.retry
    assert not queue.sent


def test_delivery_worker_retries_only_before_transport_when_channel_missing() -> None:
    queue = FakeQueue(delivery())

    outcome = run(deliver_outbound_once(queue, channels={}))

    assert outcome == {"delivery_id": "d1", "status": "retry", "marked": True}
    assert len(queue.retry) == 1
    assert not queue.unknown


def test_delivery_worker_does_not_contact_transport_after_claim_loss() -> None:
    queue = FakeQueue(delivery(), begin_ok=False)
    channel = MagicMock()
    channel.send = AsyncMock()

    outcome = run(deliver_outbound_once(queue, channels={"bitrix24": channel}))

    assert outcome == {"delivery_id": "d1", "status": "claim_lost", "marked": False}
    channel.send.assert_not_awaited()


def test_orchestrator_queues_outbound_without_contacting_channel() -> None:
    channel = MagicMock()
    channel.send = AsyncMock()
    outbox = MagicMock()
    outbox.enqueue = AsyncMock(return_value=("delivery-id", True))
    orchestrator = InternalOrchestrator(
        SimpleNamespace(id="internal_orchestrator"),
        agent_tools=[],
        llm=None,
        channels={"bitrix24": channel},
        outbound_queue=outbox,
    )
    task = AgentTask(
        task_id="task-1",
        request="hello",
        context={"channel_id": "bitrix24", "recipient_id": "13", "dialog_key": "d1"},
    )
    result = AgentResult(status="completed", agent_id="orchestrator", answer="hello")

    run(orchestrator._send_to_channel(task, result))

    outbox.enqueue.assert_awaited_once()
    channel.send.assert_not_awaited()
    assert outbox.enqueue.await_args.kwargs["delivery_key"].startswith("bitrix24:13:task-1:")


class RecoveringQueue:
    def __init__(self) -> None:
        self._item = delivery()

    async def recover_stale(self):
        return ["d1"]

    async def get(self, delivery_id: str):
        assert delivery_id == "d1"
        return self._item

    async def claim_next(self):
        return None


def test_delivery_worker_traces_and_incidents_crash_recovered_unknown() -> None:
    trace = MagicMock()
    trace.record_outbound = AsyncMock()
    incidents = MagicMock()
    incidents.publish = AsyncMock()

    outcome = run(
        deliver_outbound_once(
            RecoveringQueue(),
            channels={},
            conversation_trace=trace,
            incident_queue=incidents,
        )
    )

    assert outcome is None
    trace.record_outbound.assert_awaited_once()
    assert trace.record_outbound.await_args.kwargs["status"] == "unknown"
    assert trace.record_outbound.await_args.kwargs["delivery_id"] == "d1"
    incidents.publish.assert_awaited_once()
