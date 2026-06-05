import anyio
from fastapi.testclient import TestClient

from ai_server.channels.bitrix import BitrixWebhookProcessor
from ai_server.learning import LearningEventRecorder
from ai_server.main import app
from ai_server.models import ActionRecord, AgentResult, AgentTask, ModelUsageRecord, UserContext


def test_learning_recorder_records_agent_result_and_feedback(tmp_path):
    recorder = LearningEventRecorder(tmp_path / "learning_events.jsonl", enabled=True)
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
        tmp_path / "learning_events.jsonl",
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


def test_bitrix_webhook_processor_records_learning_event(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DRY_RUN", "true")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ENABLED", "false")
    recorder = LearningEventRecorder(tmp_path / "learning_events.jsonl", enabled=True)

    class FakeOrchestrator:
        async def handle(self, task):
            return AgentResult(
                status="completed",
                agent_id="internal_orchestrator",
                answer="Готово",
                handoff_to=["bitrix24"],
                confidence=0.9,
            )

    processor = BitrixWebhookProcessor(
        orchestrator=FakeOrchestrator(),
        learning_recorder=recorder,
    )

    result = anyio.run(processor.process, _bitrix_v2_message_payload())
    event = recorder.latest(limit=1)[0]

    assert result["learning_event"]["recorded"] is True
    assert event["source"] == "bitrix24_chat"
    assert event["agent_id"] == "internal_orchestrator"
    assert event["request"] == "Покажи задачи в Битриксе"
    assert event["response"] == "Готово"
    assert event["metadata"]["dialog_key"] == "chat:77:user:9"


def test_learning_feedback_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setenv("LEARNING_EVENTS_ENABLED", "true")
    monkeypatch.setenv("LEARNING_EVENTS_CAPTURE_TEXT", "true")

    with TestClient(app) as client:
        status = client.get("/learning/status")
        feedback = client.post(
            "/learning/feedback",
            json={
                "event_id": "event-1",
                "rating": -1,
                "comment": "Нужно поправить тон",
                "corrected_answer": "Более короткий ответ",
                "tags": ["tone"],
                "user_id": "9",
            },
        )
        events = client.get("/learning/events")

    assert status.status_code == 200
    assert feedback.status_code == 200
    assert feedback.json()["recorded"] is True
    assert events.json()["events"][0]["event_type"] == "human_feedback"
    assert events.json()["events"][0]["metadata"]["rating"] == -1


def test_learning_events_endpoint_requires_secret_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "learning-secret")

    with TestClient(app) as client:
        forbidden = client.get("/learning/events")
        allowed = client.get("/learning/events?secret=learning-secret")

    assert forbidden.status_code == 403
    assert allowed.status_code == 200


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
