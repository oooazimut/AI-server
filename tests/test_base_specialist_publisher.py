"""Tests for BaseSpecialist.run() — result_publisher integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import anyio

from ai_server.agents.base import BaseSpecialist
from ai_server.models import AgentManifest, AgentResult, AgentTask


def _manifest() -> AgentManifest:
    return AgentManifest(id="test-specialist", kind="specialist", name="Test", version="0.1.0", description="тест")


def _task() -> AgentTask:
    return AgentTask(task_id="t-1", request="hello")


def _result() -> AgentResult:
    return AgentResult(status="completed", agent_id="test-specialist", answer="ok", confidence=0.9)


class _FakeSpecialist(BaseSpecialist):
    """Minimal subclass — handle() always returns a fixed result."""

    action_prefix = "test"

    def __init__(self, manifest, *, result_publisher=None, fake_result=None):
        self._fake_result = fake_result or _result()
        super().__init__(manifest, result_publisher=result_publisher)

    async def handle(self, task: AgentTask) -> AgentResult:
        return self._fake_result

    def _logs(self):
        return []

    def _llm_failure_result(self, message):
        return _result()


def _make_queue(task: AgentTask) -> AsyncMock:
    msg = {
        "id": "m-1",
        "payload": task.model_dump(),
        "reply_to": "",
        "correlation_id": "",
    }
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[msg, None, None, None])
    queue.ack = AsyncMock()
    queue.nack = AsyncMock()
    queue.publish = AsyncMock()
    return queue


def _run_one(specialist: _FakeSpecialist, queue: AsyncMock) -> None:
    async def _impl() -> None:
        worker_task = asyncio.create_task(specialist.run(queue))
        await asyncio.sleep(0.05)
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    anyio.run(_impl)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_run_calls_result_publisher():
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    specialist = _FakeSpecialist(_manifest(), result_publisher=publisher)
    queue = _make_queue(_task())

    _run_one(specialist, queue)

    publisher.publish.assert_awaited_once()
    call_task, call_result = publisher.publish.call_args.args
    assert call_task.task_id == "t-1"
    assert call_result.answer == "ok"


def test_run_without_publisher_does_not_raise():
    specialist = _FakeSpecialist(_manifest(), result_publisher=None)
    queue = _make_queue(_task())

    _run_one(specialist, queue)

    queue.ack.assert_awaited_once_with("m-1")


def test_run_publisher_failure_does_not_prevent_ack():
    """Even if result_publisher raises, the worker must ack the message."""
    publisher = AsyncMock()
    publisher.publish = AsyncMock(side_effect=RuntimeError("redis down"))
    specialist = _FakeSpecialist(_manifest(), result_publisher=publisher)
    queue = _make_queue(_task())

    _run_one(specialist, queue)

    queue.ack.assert_awaited_once_with("m-1")


def test_run_does_not_publish_to_queue_when_no_reply_to():
    publisher = AsyncMock()
    specialist = _FakeSpecialist(_manifest(), result_publisher=publisher)
    queue = _make_queue(_task())

    _run_one(specialist, queue)

    queue.publish.assert_not_called()
    publisher.publish.assert_awaited_once()
