import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from ai_server.agents.diagnostic_agent import DiagnosticAgent
from ai_server.agents.diagnostic_agent.error_report import ErrorReportService
from ai_server.agents.diagnostic_agent import DiagnosticLLMService
from ai_server.learning import LearningEventRecorder
from ai_server.models import AgentTask, UserContext
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import get_agent_manifest, load_agent_manifests
from ai_server.settings import get_settings
from ai_server.specialists import build_specialist_registry
from tests.fakes import RecordingLLMClient


def test_diagnostic_agent_manifest_exists_and_is_internal_audience():
    manifest = get_agent_manifest("diagnostic_agent")

    assert manifest is not None
    assert manifest.kind == "specialist"
    assert manifest.audience == "diagnostics"
    assert manifest.instructions_file == "agents/diagnostic_agent/instructions.md"


def test_employee_specialist_registry_excludes_diagnostic_agent():
    manifests = load_agent_manifests()

    specialists = build_specialist_registry(manifests, audience="employee")

    assert "diagnostic_agent" not in specialists


def test_diagnostic_llm_payload_includes_loaded_feedback_rules():
    client = RecordingLLMClient(
        '{"status":"completed","answer":"Нужен разбор feedback.","confidence":0.7,'
        '"tool_calls":[{"name":"none","args":{},"summary":""}]}'
    )
    manifest = get_agent_manifest("diagnostic_agent")

    result = asyncio.run(
        DiagnosticLLMService(client).decide(
            manifest=manifest,
            task=AgentTask(
                task_id="diag-1",
                request="Разбери feedback: пользователь отметил задачу как не выполнено",
                context={
                    "feedback": {"rating": 3, "comment": "товар есть, но агент сказал что нет"},
                    "rating": 3,
                },
            ),
            retrieval_hits=[],
            tool_definitions=[],
        )
    )

    system_prompt = client.calls[0]["messages"][0]["content"]
    payload = json.loads(client.calls[0]["messages"][1]["content"])

    assert "Агент диагностики" in system_prompt
    assert "Глава: разбор feedback" in system_prompt
    assert payload["loaded_rules"][0]["id"] == "feedback_triage"
    assert payload["loaded_rules"][0]["file"] == "knowledge/feedback_triage.md"
    assert result.raw["loaded_rules"][0]["id"] == "feedback_triage"


def test_diagnostic_agent_builds_error_report(tmp_path):
    recorder = _recorder_with_report(tmp_path)
    manifest = get_agent_manifest("diagnostic_agent")
    agent = DiagnosticAgent(
        manifest,
        llm=object(),
        error_report_service=ErrorReportService(recorder),
    )

    result = asyncio.run(
        agent.handle(
            AgentTask(
                task_id="report-1",
                request="отчет ошибок",
                context={"error_report_request": {"since_hours": 24, "limit": 50, "max_groups": 3}},
            )
        )
    )

    assert result.status == "completed"
    assert "Отчет Диагноста" in result.answer
    assert result.artifacts[0].type == "diagnostic_error_report"
    report = result.artifacts[0].metadata["report"]
    assert report["total_incidents"] == 1
    assert report["groups"][0]["fix_proposal"] == "return flat task list."


def test_orchestrator_routes_admin_error_report_to_diagnostic_agent(tmp_path):
    recorder = _recorder_with_report(tmp_path)
    manifests = load_agent_manifests()
    diagnostic = DiagnosticAgent(
        get_agent_manifest("diagnostic_agent"),
        llm=object(),
        error_report_service=ErrorReportService(recorder),
    )
    orchestrator = InternalOrchestrator(
        manifests,
        specialists={"diagnostic_agent": diagnostic},
        settings=SimpleNamespace(
            resolved_supervisor_admin_user_ids=[9],
            resolved_vehicle_usage_admin_notify_user_ids=[],
            vehicle_usage_manager_user_id=None,
        ),
    )

    result = asyncio.run(
        orchestrator.handle(
            AgentTask(
                task_id="report-admin",
                source="bitrix24_chat",
                user=UserContext(id="9", channel="bitrix24_chat"),
                request="дай отчет ошибок за сегодня",
            )
        )
    )

    assert result.status == "completed"
    assert result.handoff_to == ["diagnostic_agent"]
    assert "Отчет Диагноста" in result.answer


def test_orchestrator_build_wires_diagnostic_error_report_service(tmp_path):
    recorder = _recorder_with_report(tmp_path)
    manifests = load_agent_manifests()
    orchestrator = InternalOrchestrator.build(
        get_agent_manifest("internal_orchestrator"),
        manifests=manifests,
        learning_recorder=recorder,
        settings=replace(get_settings(), diagnostic_report_admin_user_ids="9"),
    )

    result = asyncio.run(
        orchestrator.handle(
            AgentTask(
                task_id="report-build",
                source="bitrix24_chat",
                user=UserContext(id="9", channel="bitrix24_chat"),
                request="выведи отчет по ошибкам",
            )
        )
    )

    assert result.status == "completed"
    assert result.handoff_to == ["diagnostic_agent"]
    assert result.artifacts[0].type == "diagnostic_error_report"
    assert result.artifacts[0].metadata["report"]["total_incidents"] == 1


def test_orchestrator_denies_bitrix_error_report_for_non_admin(tmp_path):
    manifests = load_agent_manifests()
    orchestrator = InternalOrchestrator(
        manifests,
        specialists={},
        settings=SimpleNamespace(
            resolved_supervisor_admin_user_ids=[9],
            resolved_vehicle_usage_admin_notify_user_ids=[],
            vehicle_usage_manager_user_id=None,
        ),
    )

    result = asyncio.run(
        orchestrator.handle(
            AgentTask(
                task_id="report-denied",
                source="bitrix24_chat",
                user=UserContext(id="10", channel="bitrix24_chat"),
                request="дай отчет ошибок за сегодня",
            )
        )
    )

    assert result.status == "needs_human"
    assert result.handoff_to == []
    assert "dev/admin" in result.answer


def _recorder_with_report(tmp_path):
    recorder = LearningEventRecorder(path=tmp_path / "learning_events.jsonl", enabled=True)
    target = recorder.record_event(
        event_type="agent_result",
        source="bitrix24_chat",
        agent_id="internal_orchestrator",
        request="Покажи мои задачи",
        response="Показал статистику вместо списка",
        status="completed",
        handoff_to=["bitrix24"],
        metadata={
            "diagnostic_trace": {
                "called_agents": ["bitrix24"],
                "loaded_skills": [{"id": "tasks"}],
                "tool_calls": [{"name": "bitrix_api"}],
            }
        },
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
        response=(
            "**Where to fix:** agents/bitrix24/task_formatting.md\n\n"
            "**Fix proposal:** return flat task list.\n\n"
            "**Regression test:** task 8071 is present in all-tasks response."
        ),
        status="completed",
        metadata={
            "target_event_id": target["event_id"],
            "feedback_event_ids": [feedback["event_id"]],
            "incident_event_ids": [feedback["incident"]["event_id"]],
        },
    )
    return recorder
