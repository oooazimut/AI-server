import json

import anyio
from fastapi.testclient import TestClient

from ai_server.agents.diagnostic_agent import DiagnosticLLMService
from ai_server.learning import EventStream, LearningEventRecorder
from ai_server.main import app
from ai_server.models import ActionRecord, AgentResult, AgentTask, ModelUsageRecord, UserContext
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import load_agent_manifests
from tests.fakes import RecordingLLMClient


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


def test_learning_feedback_low_rating_creates_incident(tmp_path):
    recorder = LearningEventRecorder(path=tmp_path / "learning_events.jsonl", enabled=True)
    target = recorder.record_event(
        event_type="agent_result",
        source="local_test",
        agent_id="internal_orchestrator",
        task_id="task-incident",
        request="найди товар",
        response="товара нет",
        status="completed",
        handoff_to=["bitrix24"],
        actions=[
            {
                "name": "delegate_to_specialist",
                "status": "completed",
                "details": {"specialist": "bitrix24"},
            }
        ],
        model_usage=[{"agent_id": "bitrix24", "provider": "fake", "model": "fake"}],
        metadata={"trace_id": "trace-1"},
    )

    feedback = recorder.record_feedback(
        event_id=target["event_id"],
        rating=3,
        rating_scale=10,
        outcome="not_completed",
        comment="товар есть, но агент сказал что нет",
        tags=["catalog"],
    )

    incidents = recorder.incidents_for(target["event_id"])
    latest = recorder.latest(limit=5)

    assert feedback["recorded"] is True
    assert feedback["incident"]["recorded"] is True
    assert len(incidents) == 1
    assert incidents[0]["event_type"] == "incident"
    assert incidents[0]["status"] == "open"
    assert incidents[0]["metadata"]["target_event_id"] == target["event_id"]
    assert incidents[0]["metadata"]["feedback_event_id"] == feedback["event_id"]
    assert incidents[0]["metadata"]["trace_id"] == "trace-1"
    assert incidents[0]["metadata"]["reason"] == "feedback_outcome:not_completed"
    assert incidents[0]["actions"][0]["name"] == "delegate_to_specialist"
    assert latest[-1]["event_type"] == "incident"


def test_learning_feedback_positive_rating_does_not_create_incident(tmp_path):
    recorder = LearningEventRecorder(path=tmp_path / "learning_events.jsonl", enabled=True)
    target = recorder.record_event(
        event_type="agent_result",
        source="local_test",
        agent_id="internal_orchestrator",
        request="создай задачу",
        response="готово",
        status="completed",
    )

    feedback = recorder.record_feedback(
        event_id=target["event_id"],
        rating=1,
        comment="все хорошо",
    )

    assert feedback["recorded"] is True
    assert "incident" not in feedback
    assert recorder.incidents_for(target["event_id"]) == []


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


def test_orchestrator_records_learning_event(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DRY_RUN", "true")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ENABLED", "false")
    recorder = LearningEventRecorder(path=tmp_path / "learning_events.jsonl", enabled=True)

    class FakeOrchestratorLLM:
        async def decide(self, **kwargs):
            from ai_server.models import ModelUsageRecord
            from ai_server.orchestrators.orchestrator_llm import OrchestratorDecision, OrchestratorDecisionResult

            decision = OrchestratorDecision(
                status="completed",
                answer="",
                tool_calls=[],
                scheduled_tasks=[],
                confidence=0.9,
            )
            return OrchestratorDecisionResult(
                decision=decision,
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

    manifests = load_agent_manifests()
    orchestrator = InternalOrchestrator(
        manifests,
        specialists={},
        orchestrator_llm=FakeOrchestratorLLM(),
        learning_recorder=recorder,
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
    event = recorder.latest(limit=1)[0]

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


def test_learning_diagnose_endpoint_runs_diagnostic_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setenv("LEARNING_EVENTS_ENABLED", "true")
    monkeypatch.setenv("LEARNING_EVENTS_CAPTURE_TEXT", "true")
    client_llm = RecordingLLMClient(
        '{"status":"completed","answer":"Сбой вероятно в skill catalog.","confidence":0.8,'
        '"tool_calls":[{"name":"none","args":{},"summary":""}]}'
    )

    with TestClient(app) as client:
        client.app.state.diagnostic_llm = DiagnosticLLMService(client_llm)
        recorder = client.app.state.learning_recorder
        write_result = recorder.record_event(
            event_type="agent_result",
            source="local_test",
            agent_id="internal_orchestrator",
            request="найди датчик коленвала",
            response="Такого товара нет",
            status="completed",
            handoff_to=["bitrix24"],
            actions=[
                {
                    "name": "load_bitrix24_specialist_context",
                    "details": {"loaded_skills": [{"id": "catalog"}]},
                }
            ],
        )
        feedback_result = recorder.record_feedback(
            event_id=write_result["event_id"],
            rating=-1,
            comment="товар есть, но агент сказал что нет",
            tags=["catalog"],
        )
        response = client.post(
            "/learning/diagnose",
            json={"event_id": write_result["event_id"], "comment": "Разбери ошибку поиска товара"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["feedback_events"] == [feedback_result["event_id"]]
    assert payload["diagnostic_event"]["recorded"] is True
    assert "Сбой вероятно" in payload["answer"]

    diagnostic_payload = json.loads(client_llm.calls[0]["messages"][1]["content"])
    assert diagnostic_payload["context"]["target_event"]["id"] == write_result["event_id"]
    assert diagnostic_payload["context"]["feedback_events"][0]["id"] == feedback_result["event_id"]

    events = recorder.latest(limit=5)
    diagnostic_event = events[-1]
    assert diagnostic_event["event_type"] == "diagnostic_report"
    assert diagnostic_event["metadata"]["target_event_id"] == write_result["event_id"]
    assert diagnostic_event["metadata"]["feedback_event_ids"] == [feedback_result["event_id"]]


def test_learning_diagnostic_groups_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setenv("LEARNING_EVENTS_ENABLED", "true")
    monkeypatch.setenv("LEARNING_EVENTS_CAPTURE_TEXT", "true")

    with TestClient(app) as client:
        recorder = client.app.state.learning_recorder
        for index in range(2):
            target = recorder.record_event(
                event_type="agent_result",
                source="local_test",
                agent_id="internal_orchestrator",
                request=f"найди датчик {index}",
                response="Не найдено",
                status="completed",
                metadata={
                    "diagnostic_trace": {
                        "called_agents": ["bitrix24"],
                        "loaded_rules": [],
                        "loaded_skills": [{"id": "catalog", "file": "skills/catalog.md"}],
                        "tool_calls": [{"name": "bitrix_api"}],
                        "errors": [],
                    }
                },
            )
            feedback = recorder.record_feedback(
                event_id=target["event_id"],
                rating=-1,
                comment="товар есть",
                tags=["catalog"],
            )
            recorder.record_event(
                event_type="diagnostic_report",
                source="learning_diagnose",
                agent_id="diagnostic_agent",
                response=(
                    "**What went wrong:** catalog search missed an existing item.\n"
                    "**Where to fix:** skills/catalog.md\n"
                    "**Fix proposal:** add synonym and partial-name fallback.\n"
                    "**Regression test:** query existing sensor by partial name."
                ),
                status="completed",
                metadata={
                    "target_event_id": target["event_id"],
                    "feedback_event_ids": [feedback["event_id"]],
                    "task_context": {
                        "target_event": recorder.get_event(target["event_id"]),
                        "feedback_events": [recorder.get_event(feedback["event_id"])],
                    },
                    "diagnostic_trace": {
                        "called_agents": [],
                        "loaded_rules": [{"id": "feedback_triage"}],
                        "loaded_skills": [],
                        "tool_calls": [],
                        "errors": [],
                    },
                },
            )

        response = client.get("/learning/diagnostics/groups")
        detailed_response = client.get("/learning/diagnostics/groups?detailed=true")

    assert response.status_code == 200
    groups = {group["key"]: group for group in response.json()["groups"]}
    assert response.json()["mode"] == "brief"
    assert groups["loaded_skill:catalog"]["count"] == 2
    assert "diagnosis" not in groups["loaded_skill:catalog"]
    assert groups["tag:catalog"]["count"] == 2
    assert groups["target_agent:bitrix24"]["count"] == 2

    assert detailed_response.status_code == 200
    detailed_groups = {group["key"]: group for group in detailed_response.json()["groups"]}
    catalog_diagnosis = detailed_groups["loaded_skill:catalog"]["diagnosis"]
    assert detailed_response.json()["mode"] == "detailed"
    assert "skill `catalog`" in catalog_diagnosis["problem"]
    assert "fix_proposal" in catalog_diagnosis
    catalog_suggestion = detailed_groups["loaded_skill:catalog"]["suggestions"][0]
    assert catalog_suggestion["where_to_fix"] == "skills/catalog.md"
    assert catalog_suggestion["fix_proposal"] == "add synonym and partial-name fallback."
    assert catalog_suggestion["regression_test"] == "query existing sensor by partial name."


def test_learning_error_report_endpoint_runs_through_diagnostic_orchestrator(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setenv("LEARNING_EVENTS_ENABLED", "true")
    monkeypatch.setenv("LEARNING_EVENTS_CAPTURE_TEXT", "true")

    with TestClient(app) as client:
        recorder = client.app.state.learning_recorder
        target = recorder.record_event(
            event_type="agent_result",
            source="bitrix24_chat",
            agent_id="internal_orchestrator",
            request="Покажи мои задачи",
            response="Показал статистику вместо списка",
            status="completed",
            handoff_to=["bitrix24"],
            metadata={"diagnostic_trace": {"called_agents": ["bitrix24"], "loaded_skills": [{"id": "tasks"}]}},
        )
        feedback = recorder.record_feedback(
            event_id=target["event_id"],
            rating=4,
            rating_scale=10,
            outcome="not_completed",
            comment="нужен список задач",
            tags=["tasks"],
        )
        recorder.record_event(
            event_type="diagnostic_report",
            source="learning_diagnose",
            agent_id="internal_orchestrator",
            response="**Fix proposal:** return flat task list.",
            status="completed",
            metadata={
                "target_event_id": target["event_id"],
                "feedback_event_ids": [feedback["event_id"]],
                "incident_event_ids": [feedback["incident"]["event_id"]],
            },
        )
        response = client.get("/learning/reports/errors?format=json")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["handoff_to"] == ["diagnostic_agent"]
    assert payload["report"]["total_incidents"] == 1
    assert payload["report"]["groups"][0]["fix_proposal"] == "return flat task list."


def test_learning_incidents_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setenv("LEARNING_EVENTS_ENABLED", "true")
    monkeypatch.setenv("LEARNING_EVENTS_CAPTURE_TEXT", "true")

    with TestClient(app) as client:
        recorder = client.app.state.learning_recorder
        client.app.state.trace_recorder.record(
            event_name="orchestrator_decision",
            trace_id="trace-endpoint",
            span_id="span-endpoint",
            agent_id="internal_orchestrator",
            task_id="task-endpoint",
            status="completed",
            payload={"tool_calls": [{"name": "bitrix_api"}]},
        )
        target = recorder.record_event(
            event_type="agent_result",
            source="local_test",
            agent_id="internal_orchestrator",
            task_id="task-endpoint",
            request="найди датчик",
            response="не найдено",
            status="completed",
            metadata={"trace_id": "trace-endpoint"},
        )
        feedback = client.post(
            "/learning/feedback",
            json={
                "event_id": target["event_id"],
                "rating": 3,
                "rating_scale": 10,
                "outcome": "not_completed",
                "comment": "датчик есть",
                "tags": ["catalog"],
            },
        )
        incidents = client.get("/learning/incidents")
        target_incidents = client.get(f"/learning/incidents?event_id={target['event_id']}")

    assert feedback.status_code == 200
    assert feedback.json()["incident"]["recorded"] is True
    assert incidents.status_code == 200
    assert incidents.json()["incidents"][0]["event_type"] == "incident"
    assert target_incidents.status_code == 200
    assert target_incidents.json()["incidents"][0]["metadata"]["target_event_id"] == target["event_id"]
    assert target_incidents.json()["incidents"][0]["metadata"]["trace_events"][0]["event_name"] == "orchestrator_decision"


def test_learning_incident_groups(tmp_path):
    recorder = LearningEventRecorder(path=tmp_path / "learning_events.jsonl", enabled=True)
    for index in range(2):
        target = recorder.record_event(
            event_type="agent_result",
            source="local_test",
            agent_id="internal_orchestrator",
            request=f"найди датчик {index}",
            response="не найдено",
            status="completed",
            handoff_to=["bitrix24"],
            metadata={
                "trace_id": f"trace-{index}",
                "diagnostic_trace": {
                    "called_agents": ["bitrix24"],
                    "loaded_rules": [],
                    "loaded_skills": [{"id": "catalog", "file": "skills/catalog.md"}],
                    "tool_calls": [{"name": "bitrix_api"}],
                    "errors": [],
                },
            },
        )
        feedback = recorder.record_feedback(
            event_id=target["event_id"],
            rating=3,
            rating_scale=10,
            outcome="not_completed",
            comment="товар есть",
            tags=["catalog"],
        )
        recorder.record_event(
            event_type="diagnostic_report",
            source="learning_diagnose",
            agent_id="internal_orchestrator",
            response=(
                "**Where to fix:** skills/catalog.md\n"
                "**Fix proposal:** add catalog fallback search.\n"
                "**Regression test:** missing item feedback creates grouped report."
            ),
            status="completed",
            metadata={
                "target_event_id": target["event_id"],
                "feedback_event_ids": [feedback["event_id"]],
                "incident_event_ids": [feedback["incident"]["event_id"]],
            },
        )

    brief = recorder.incident_groups()
    detailed = recorder.incident_groups(detailed=True)

    brief_groups = {group["key"]: group for group in brief["groups"]}
    detailed_groups = {group["key"]: group for group in detailed["groups"]}

    assert brief["mode"] == "brief"
    assert brief["total_incidents"] == 2
    assert brief_groups["loaded_skill:catalog"]["count"] == 2
    assert brief_groups["tool_call:bitrix_api"]["count"] == 2
    assert brief_groups["target_agent:bitrix24"]["count"] == 2
    assert brief_groups["incident_reason:feedback_outcome:not_completed"]["count"] == 2
    assert "diagnosis" not in brief_groups["loaded_skill:catalog"]

    assert detailed["mode"] == "detailed"
    assert "fix_proposal" in detailed_groups["loaded_skill:catalog"]["diagnosis"]
    linked_report = detailed_groups["loaded_skill:catalog"]["diagnostic_reports"][0]
    assert linked_report["fix_proposal"] == "add catalog fallback search."
    assert linked_report["regression_test"] == "missing item feedback creates grouped report."
    assert "Повторяется причина" in detailed_groups["incident_reason:feedback_outcome:not_completed"]["diagnosis"][
        "problem"
    ]


def test_learning_incident_groups_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "")
    monkeypatch.setenv("LEARNING_EVENTS_ENABLED", "true")
    monkeypatch.setenv("LEARNING_EVENTS_CAPTURE_TEXT", "true")

    with TestClient(app) as client:
        recorder = client.app.state.learning_recorder
        target = recorder.record_event(
            event_type="agent_result",
            source="local_test",
            agent_id="internal_orchestrator",
            request="найди датчик",
            response="не найдено",
            status="completed",
            metadata={
                "diagnostic_trace": {
                    "called_agents": ["bitrix24"],
                    "loaded_skills": [{"id": "catalog"}],
                    "tool_calls": [{"name": "bitrix_api"}],
                }
            },
        )
        recorder.record_feedback(
            event_id=target["event_id"],
            rating=-1,
            comment="датчик есть",
            tags=["catalog"],
        )
        response = client.get("/learning/incidents/groups?detailed=true")

    assert response.status_code == 200
    payload = response.json()
    groups = {group["key"]: group for group in payload["groups"]}
    assert payload["mode"] == "detailed"
    assert groups["loaded_skill:catalog"]["count"] == 1
    assert "fix_proposal" in groups["loaded_skill:catalog"]["diagnosis"]


def test_learning_events_endpoint_requires_secret_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "learning-secret")

    with TestClient(app) as client:
        forbidden = client.get("/learning/events")
        allowed = client.get("/learning/events?secret=learning-secret")

    assert forbidden.status_code == 403
    assert allowed.status_code == 200


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


def test_learning_event_includes_diagnostic_trace_summary(tmp_path):
    stream = EventStream(path=tmp_path / "events.jsonl", enabled=True, capture_text=True)
    task = AgentTask(task_id="t1", request="найди датчик коленвала")
    result = AgentResult(
        status="completed",
        agent_id="internal_orchestrator",
        answer="ответ",
        handoff_to=["bitrix24"],
        actions_taken=[
            ActionRecord(
                name="orchestrator_llm_decision",
                status="completed",
                details={
                    "loaded_rules": [{"id": "routing_guidelines", "file": "knowledge/routing_guidelines.md"}],
                    "tool_calls": [{"name": "call_bitrix24", "summary": "поиск товара"}],
                },
            ),
            ActionRecord(
                name="bitrix24_llm_decision",
                status="completed",
                details={"loaded_skills": [{"id": "catalog", "file": "skills/catalog.md"}]},
            ),
        ],
    )

    stream.record_agent_result(task, result)

    trace = stream.latest(limit=1)[0]["metadata"]["diagnostic_trace"]
    assert trace["called_agents"] == ["bitrix24"]
    assert trace["loaded_rules"][0]["id"] == "routing_guidelines"
    assert trace["loaded_skills"][0]["id"] == "catalog"
    assert trace["tool_calls"][0]["name"] == "call_bitrix24"


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
