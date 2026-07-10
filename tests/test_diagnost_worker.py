"""Tests for run_diagnost_event_worker — auto-incident logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import anyio

from ai_server.models import AgentResult, AgentTask, UserContext
from ai_server.workers.diagnost.event_worker import run_diagnost_event_worker


def _make_task(task_id: str = "task-1") -> AgentTask:
    return AgentTask(task_id=task_id, request="тест")


def _make_result(status: str = "completed", confidence: float = 0.9) -> AgentResult:
    return AgentResult(status=status, agent_id="internal_orchestrator", answer="тест", confidence=confidence)


def _msg(task: AgentTask, result: AgentResult, msg_id: str = "msg-1") -> dict:
    return {
        "_id": msg_id,
        "task": task.model_dump(),
        "result": result.model_dump(),
    }


def _run_one(queue: AsyncMock, store: AsyncMock, **kwargs) -> None:
    """Run the worker loop for exactly one message cycle then cancel."""

    async def _impl() -> None:
        worker_task = asyncio.create_task(run_diagnost_event_worker(queue, store, **kwargs))
        await asyncio.sleep(0.05)
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    anyio.run(_impl)


# ── happy path ────────────────────────────────────────────────────────────────


def test_worker_saves_event_on_completed():
    task = _make_task()
    result = _make_result(status="completed", confidence=0.9)
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[_msg(task, result), None, None, None])
    queue.ack = AsyncMock()
    store = AsyncMock()
    store.save_event = AsyncMock()
    store.save_incident = AsyncMock()

    _run_one(queue, store)

    store.save_event.assert_awaited_once()
    store.save_incident.assert_not_called()
    queue.ack.assert_awaited_once_with("msg-1")


def test_worker_creates_incident_on_failed():
    task = _make_task("task-failed")
    result = _make_result(status="failed", confidence=0.9)
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[_msg(task, result), None, None, None])
    queue.ack = AsyncMock()
    store = AsyncMock()
    store.save_event = AsyncMock()
    store.save_incident = AsyncMock()

    _run_one(queue, store)

    store.save_incident.assert_awaited_once_with("task-failed", reason="failed")
    queue.ack.assert_awaited_once_with("msg-1")


def test_worker_creates_incident_on_low_confidence():
    task = _make_task("task-low")
    result = _make_result(status="completed", confidence=0.3)
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[_msg(task, result), None, None, None])
    queue.ack = AsyncMock()
    store = AsyncMock()
    store.save_event = AsyncMock()
    store.save_incident = AsyncMock()

    _run_one(queue, store, confidence_threshold=0.5)

    store.save_incident.assert_awaited_once_with("task-low", reason="low_confidence")


def test_worker_no_incident_on_confidence_above_threshold():
    task = _make_task()
    result = _make_result(status="completed", confidence=0.7)
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[_msg(task, result), None, None, None])
    queue.ack = AsyncMock()
    store = AsyncMock()
    store.save_event = AsyncMock()
    store.save_incident = AsyncMock()

    _run_one(queue, store, confidence_threshold=0.5)

    store.save_incident.assert_not_called()


def test_worker_no_incident_on_high_confidence():
    """confidence=1.0 → above any threshold, no incident."""
    task = _make_task()
    result = _make_result(status="completed", confidence=1.0)
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[_msg(task, result), None, None, None])
    queue.ack = AsyncMock()
    store = AsyncMock()
    store.save_event = AsyncMock()
    store.save_incident = AsyncMock()

    _run_one(queue, store, confidence_threshold=0.5)

    store.save_incident.assert_not_called()


def test_worker_uses_recipient_id_for_feedback_prompt():
    task = AgentTask(
        task_id="task-feedback",
        request="тест",
        user=UserContext(id="1", channel="bitrix24_chat"),
        context={
            "dialog_key": "chat:4321:user:1",
            "dialog_id": "chat4321",
            "recipient_id": "chat4321",
        },
    )
    result = _make_result(status="completed", confidence=0.9)
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[_msg(task, result), None, None, None])
    queue.ack = AsyncMock()
    store = AsyncMock()
    store.save_event = AsyncMock()
    store.save_incident = AsyncMock()
    store.create_pending_feedback = AsyncMock()

    _run_one(queue, store)

    store.create_pending_feedback.assert_awaited_once_with(
        "task-feedback",
        "1",
        "chat4321",
        channel="bitrix24_chat",
    )


def test_worker_skips_malformed_message():
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[{"_id": "bad", "task": "not-a-dict"}, None, None])
    queue.ack = AsyncMock()
    store = AsyncMock()
    store.save_event = AsyncMock()

    _run_one(queue, store)

    store.save_event.assert_not_called()
    queue.ack.assert_awaited_once_with("bad")


def test_worker_nacks_on_exception():
    task = _make_task()
    result = _make_result()
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[_msg(task, result), None, None])
    queue.ack = AsyncMock()
    queue.nack = AsyncMock()
    store = AsyncMock()
    store.save_event = AsyncMock(side_effect=RuntimeError("db down"))
    store.save_incident = AsyncMock()

    _run_one(queue, store)

    queue.nack.assert_awaited_once()
    nack_call = queue.nack.call_args
    assert nack_call.args[0] == "msg-1"
    assert "RuntimeError" in nack_call.kwargs["error"]
