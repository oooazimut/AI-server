"""Tests for run_diagnost_event_worker — auto-incident logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import anyio

from ai_server.models import AgentResult, AgentTask
from ai_server.workers.diagnost.event_worker import _trace_incident_reasons, run_diagnost_event_worker


def _make_task(task_id: str = "task-1") -> AgentTask:
    return AgentTask(task_id=task_id, request="тест")


def _make_result(status: str = "completed", confidence: float = 0.9) -> AgentResult:
    return AgentResult(status=status, agent_id="internal_orchestrator", answer="тест", confidence=confidence)


def _msg(task: AgentTask, result: AgentResult, msg_id: str = "msg-1", *, source: str = "orchestrator") -> dict:
    return {
        "_id": msg_id,
        "source": source,
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


def test_trace_incidents_cover_delivery_and_latency():
    task = AgentTask(
        task_id="task-trace",
        request="test",
        context={"channel_id": "bitrix24", "recipient_id": "chat1"},
    )
    result = _make_result()
    trace = [
        {"trace_type": "outbound_message", "send_status": "error"},
        {"trace_type": "timing_step", "stage": "handle_total", "elapsed_ms": 120001},
    ]

    assert _trace_incident_reasons(task, result, trace, high_latency_ms=120000) == [
        "outbound_failed",
        "high_latency",
    ]


def test_trace_incident_detects_missing_outbound():
    task = AgentTask(
        task_id="task-missing",
        request="test",
        context={"channel_id": "bitrix24", "recipient_id": "chat1"},
    )

    assert _trace_incident_reasons(task, _make_result(), [], high_latency_ms=120000) == ["missing_outbound"]


def test_trace_incident_detects_unknown_outbound_outcome():
    task = _make_task()
    trace = [{"trace_type": "outbound_message", "send_status": "unknown"}]

    assert _trace_incident_reasons(task, _make_result(), trace, high_latency_ms=120000) == ["outbound_unknown"]


def test_outbound_delivery_event_creates_unknown_incident_from_terminal_trace():
    task = _make_task("task-outbound-unknown")
    result = _make_result()
    queue = AsyncMock()
    queue.claim_next = AsyncMock(side_effect=[_msg(task, result, source="outbound_delivery"), None, None, None])
    queue.ack = AsyncMock()
    store = AsyncMock()
    store.save_event = AsyncMock()
    store.save_trace_snapshot = AsyncMock()
    store.save_incident = AsyncMock()
    trace = AsyncMock()
    trace.by_task = AsyncMock(return_value=[{"trace_type": "outbound_message", "send_status": "unknown"}])

    _run_one(
        queue,
        store,
        conversation_trace=trace,
        trace_snapshot_enabled=True,
        trace_settle_seconds=0,
    )

    store.save_incident.assert_awaited_once_with("task-outbound-unknown", reason="outbound_unknown")


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
