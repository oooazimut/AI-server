import asyncio
import json

from ai_server.agents.diagnostic_agent import DiagnosticLLMService
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest, load_agent_manifests
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
