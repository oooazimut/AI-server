import anyio
from fastapi.testclient import TestClient

from ai_server.learning import EventStream, LearningEventRecorder
from ai_server.main import app
from ai_server.models import ActionRecord, AgentResult, AgentTask, ModelUsageRecord, UserContext
from ai_server.orchestrators.internal import InternalOrchestrator


def test_learning_recorder_records_agent_result_and_feedback(tmp_path):
    recorder = LearningEventRecorder(path=tmp_path / "learning_events.jsonl", enabled=True)
    task = AgentTask(
        task_id="task-1",
        source="local_test",
        user=UserContext(id="9", channel="bitrix24_chat"),
        request="Создай задачу проверить камеру",
        context={"dialog_id": "chat99"},
    )
    result = AgentResult(
        status="needs_human",
        agent_id="internal_orchestrator",
        answer="Нужно подтверждение.",
        handoff_to=["bitrix24"],
        actions_requiring_approval=[
            ActionRecord(
                name="bitrix_api",
                status="approval_required",
                details={"method": "tasks.task.add", "summary": "создать задачу"},
            )
        ],
        model_usage=[
            ModelUsageRecord(
                agent_id="bitrix24",
                provider="deepseek",
                model="deepseek-v4-flash",
                input_tokens=10,
                output_tokens=20,
            )
        ],
        confidence=0.8,
    )

    write_result = recorder.record_agent_result(task, result)
    feedback_result = recorder.record_feedback(
        event_id=write_result["event_id"],
        rating=1,
        corrected_answer="Ок",
        comment="Подтверждение сформулировано нормально",
        tags=["task_create"],
        user_id="1",
    )

    latest = recorder.latest(limit=5)
    stats = recorder.stats()

    assert write_result["recorded"] is True
    assert feedback_result["recorded"] is True
    assert stats["total_events"] == 2
    assert stats["by_event_type"] == {"agent_result": 1, "human_feedback": 1}
    assert latest[0]["request"] == "Создай задачу проверить камеру"
    assert latest[0]["actions"][0]["kind"] == "approval_required"
    assert latest[0]["model_usage"][0]["model"] == "deepseek-v4-flash"
    assert latest[1]["event_type"] == "human_feedback"
    assert latest[1]["metadata"]["target_event_id"] == write_result["event_id"]


def test_learning_recorder_can_disable_text_capture(tmp_path):
    recorder = LearningEventRecorder(
        path=tmp_path / "learning_events.jsonl",
        enabled=True,
        capture_text=False,
    )

    recorder.record_event(
        event_type="agent_result",
        source="local_test",
        request="секретный запрос",
        response="секретный ответ",
        status="completed",
    )

    event = recorder.latest(limit=1)[0]

    assert event["request"] == ""
    assert event["response"] == ""
    assert event["privacy"]["text_captured"] is False


def test_orchestrator_calls_result_publisher(monkeypatch):
    """Оркестратор вызывает result_publisher.publish() после обработки задачи."""
    from unittest.mock import AsyncMock

    monkeypatch.setenv("AGENT_DRY_RUN", "true")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ENABLED", "false")

    class FakeOrchestratorLLM:
        async def decide(self, **kwargs):
            from ai_server.models import ModelUsageRecord
            from ai_server.orchestrators.orchestrator_llm import OrchestratorDecision, OrchestratorDecisionResult

            return OrchestratorDecisionResult(
                decision=OrchestratorDecision(status="completed", answer="", tool_calls=[], confidence=0.9),
                model_usage=ModelUsageRecord(agent_id="internal_orchestrator", provider="test", model="test"),
            )

        async def compose(self, **kwargs):
            from ai_server.models import ModelUsageRecord
            from ai_server.orchestrators.orchestrator_llm import OrchestratorFinalResult

            return OrchestratorFinalResult(
                answer="Готово",
                status="completed",
                model_usage=ModelUsageRecord(agent_id="internal_orchestrator", provider="test", model="test"),
            )

    from ai_server.models import AgentManifest
    from ai_server.orchestrators.tools import CallSpecialistTool

    publisher = AsyncMock()
    orch_manifest = AgentManifest(
        id="internal_orchestrator", name="Переговорщик", kind="orchestrator", description="test"
    )
    orchestrator = InternalOrchestrator(
        orch_manifest,
        agent_tools=[CallSpecialistTool({}, [])],
        llm=FakeOrchestratorLLM(),
        result_publisher=publisher,
    )

    task = AgentTask(
        task_id="test-task",
        source="bitrix24_chat",
        request="Покажи задачи в Битриксе",
        user=UserContext(id="9", channel="bitrix24_chat", raw={"dialog_id": "chat99"}),
        context={
            "dialog_key": "chat:77:user:9",
            "dialog_id": "chat99",
            "channel_id": "bitrix24",
            "recipient_id": "chat99",
        },
    )
    anyio.run(orchestrator.handle, task)

    publisher.publish.assert_called_once()
    call_task, call_result = publisher.publish.call_args.args
    assert call_task.task_id == "test-task"
    assert call_result.answer == "Готово"


def test_learning_status_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "")

    with TestClient(app) as client:
        status = client.get("/learning/status")

    assert status.status_code == 200
    assert "diagnost" in status.json()["status"].lower()


def test_event_stream_subscriber_called(tmp_path):
    stream = EventStream(path=tmp_path / "events.jsonl", enabled=True)
    received: list[dict] = []
    stream.subscribe(received.append)

    stream.record_event(
        event_type="agent_result",
        source="test",
        agent_id="internal_orchestrator",
        request="тест",
        response="ответ",
        status="completed",
    )

    assert len(received) == 1
    assert received[0]["event_type"] == "agent_result"
    assert received[0]["agent_id"] == "internal_orchestrator"


def test_event_stream_unsubscribe(tmp_path):
    stream = EventStream(path=tmp_path / "events.jsonl", enabled=True)
    received: list[dict] = []
    stream.subscribe(received.append)
    stream.unsubscribe(received.append)

    stream.record_event(event_type="agent_result", source="test", status="completed")

    assert received == []


def test_event_stream_elapsed_ms_in_metadata(tmp_path):
    stream = EventStream(path=tmp_path / "events.jsonl", enabled=True, capture_text=True)
    task = AgentTask(task_id="t1", request="запрос")
    result = AgentResult(status="completed", agent_id="test_agent", answer="ответ")

    stream.record_agent_result(task, result, elapsed_ms={"total_ms": 123.4})

    events = stream.latest(limit=1)
    assert events[0]["metadata"]["elapsed_ms"] == {"total_ms": 123.4}


def test_learning_event_recorder_alias(tmp_path):
    assert LearningEventRecorder is EventStream


def _bitrix_v2_message_payload() -> dict:
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "auth": {"application_token": "secret-token"},
        "data": {
            "bot": {"id": 42},
            "chat": {"id": 77, "dialogId": "chat99"},
            "message": {"id": 123, "authorId": 9, "text": "Покажи задачи в Битриксе"},
            "user": {"id": 9},
        },
    }
