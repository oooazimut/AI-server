from ai_server.models import AgentTask
from ai_server.tracing import TraceRecorder


def test_trace_recorder_writes_and_reads_trace_events(tmp_path):
    recorder = TraceRecorder(path=tmp_path / "traces.jsonl", enabled=True)
    task = AgentTask(task_id="t1", request="Проверка trace")

    trace_id, span_id = recorder.ensure_task_context(task)
    result = recorder.record(
        event_name="user_message_received",
        trace_id=trace_id,
        span_id=span_id,
        agent_id="internal_orchestrator",
        task_id=task.task_id,
        status="received",
        payload={"request": task.request},
    )

    assert result["recorded"] is True
    assert task.context["trace_id"] == trace_id
    assert task.context["span_id"] == span_id

    events = recorder.for_trace(trace_id)
    assert len(events) == 1
    assert events[0]["event_name"] == "user_message_received"
    assert events[0]["payload"]["request"] == "Проверка trace"

    stats = recorder.stats()
    assert stats["total_events"] == 1
    assert stats["by_event_name"] == {"user_message_received": 1}
