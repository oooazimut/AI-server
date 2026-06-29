import anyio

from ai_server.agents.diagnostic_agent import DiagnosticLLMService
from ai_server.feedback_loop import FeedbackLoopService, FeedbackPromptStore, parse_feedback_text
from ai_server.learning import LearningEventRecorder
from ai_server.models import AgentResult, AgentTask, UserContext
from ai_server.registry import load_agent_manifests
from ai_server.tracing import TraceRecorder
from tests.fakes import RecordingLLMClient


def test_parse_feedback_text_accepts_rating_and_comment():
    assert parse_feedback_text("7").rating == 7
    parsed = parse_feedback_text("3/10 не нашел товар")

    assert parsed.rating == 3
    assert parsed.comment == "не нашел товар"
    assert parsed.outcome == "not_completed"
    assert parse_feedback_text("найди 7 задач") is None


def test_feedback_loop_records_feedback_incident_diagnostic_report_and_trace(tmp_path):
    recorder = LearningEventRecorder(path=tmp_path / "learning_events.jsonl", enabled=True)
    trace_recorder = TraceRecorder(path=tmp_path / "traces.jsonl", enabled=True)
    store = FeedbackPromptStore(path=tmp_path / "dialog_state.sqlite")
    channel = FakeChannel()
    llm = DiagnosticLLMService(
        RecordingLLMClient(
            '{"status":"completed","answer":"Вероятный сбой в поиске склада.","confidence":0.8,'
            '"tool_calls":[{"name":"none","args":{},"summary":""}]}'
        )
    )
    service = FeedbackLoopService(
        store=store,
        learning_recorder=recorder,
        trace_recorder=trace_recorder,
        manifests=load_agent_manifests(),
        channels={"bitrix24": channel},
        diagnostic_llm=llm,
    )
    task = AgentTask(
        task_id="task-1",
        source="bitrix24_chat",
        user=UserContext(id="9", channel="bitrix24_chat"),
        request="найди датчик",
        context={
            "dialog_key": "chat:77:user:9",
            "channel_id": "bitrix24",
            "recipient_id": "chat77",
        },
    )
    trace_id, span_id = trace_recorder.ensure_task_context(task)
    trace_recorder.record(
        event_name="orchestrator_decision",
        trace_id=trace_id,
        span_id=span_id,
        agent_id="internal_orchestrator",
        task_id=task.task_id,
        status="completed",
        payload={"tool_calls": [{"name": "call_bitrix24"}]},
    )
    result = AgentResult(
        status="completed",
        agent_id="internal_orchestrator",
        answer="Не найдено",
        handoff_to=["bitrix24"],
    )
    learning_record = recorder.record_agent_result(task, result)
    service.remember_answer(task, result, learning_record)

    feedback_task = AgentTask(
        task_id="feedback-1",
        source="bitrix24_chat",
        user=UserContext(id="9", channel="bitrix24_chat"),
        request="3 не нашел товар",
        context={
            "dialog_key": "chat:77:user:9",
            "channel_id": "bitrix24",
            "recipient_id": "chat77",
        },
    )
    handled = anyio.run(service.try_handle_feedback, feedback_task)

    assert handled["routed_to"] == "feedback_loop"
    assert handled["feedback_event_id"]
    assert handled["incident_event_id"]
    assert handled["diagnostic_report_id"]
    assert channel.messages[0][0] == "chat77"
    assert "Спасибо, оценка 3/10 сохранена." in channel.messages[0][1]
    assert "Diagnostic Agent" in channel.messages[0][1]
    assert "Вероятный сбой" not in channel.messages[0][1]
    assert store.get_pending("chat:77:user:9") is None

    incident = recorder.get_event(handled["incident_event_id"])
    assert incident["event_type"] == "incident"
    assert incident["metadata"]["trace_events"][0]["event_name"] == "orchestrator_decision"

    diagnostic_report = recorder.get_event(handled["diagnostic_report_id"])
    assert diagnostic_report["event_type"] == "diagnostic_report"
    assert diagnostic_report["metadata"]["target_event_id"] == learning_record["event_id"]

    trace_events = trace_recorder.for_trace(trace_id)
    assert trace_events[-1]["event_name"] == "human_feedback_received"
    assert trace_events[-1]["payload"]["diagnostic_report_id"] == handled["diagnostic_report_id"]


class FakeChannel:
    def __init__(self) -> None:
        self.messages = []

    async def send(self, recipient_id: str, body: str) -> None:
        self.messages.append((recipient_id, body))
