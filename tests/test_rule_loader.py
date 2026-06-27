import asyncio
import json

from ai_server.models import AgentManifest, AgentTask
from ai_server.orchestrators.orchestrator_llm import OrchestratorLLMService
from ai_server.rule_loader import load_rules_for_task
from tests.fakes import RecordingLLMClient


def _manifest(instructions_file: str) -> AgentManifest:
    return AgentManifest(
        id="internal_orchestrator",
        name="Переговорщик",
        kind="orchestrator",
        description="Test orchestrator",
        instructions_file=instructions_file,
    )


def test_load_rules_for_task_selects_default_keyword_and_context_rules(tmp_path):
    agent_dir = tmp_path / "internal_orchestrator"
    knowledge_dir = agent_dir / "knowledge"
    knowledge_dir.mkdir(parents=True)
    instructions = agent_dir / "instructions.md"
    instructions.write_text("core", encoding="utf-8")
    (knowledge_dir / "routing.md").write_text("routing chapter", encoding="utf-8")
    (knowledge_dir / "diagnostics.md").write_text("diagnostics chapter", encoding="utf-8")
    (agent_dir / "rule_index.yaml").write_text(
        """
rules:
  - id: routing
    title: Routing
    file: knowledge/routing.md
    priority: 100
    use_when:
      default_for_orchestrator: true
      request_topics:
        - Битрикс
    load_reason: route
  - id: diagnostics
    title: Diagnostics
    file: knowledge/diagnostics.md
    priority: 80
    use_when:
      context_keys:
        - trace_id
      request_topics:
        - ошибка
    load_reason: debug
""",
        encoding="utf-8",
    )

    rules = load_rules_for_task(
        _manifest(str(instructions)),
        request="Есть ошибка в Битрикс",
        context={"trace_id": "abc"},
    )

    assert [rule.id for rule in rules] == ["routing", "diagnostics"]
    assert rules[0].content == "routing chapter"
    assert rules[1].matched_context_keys == ["trace_id"]
    assert "ошибка" in rules[1].matched_keywords


def test_orchestrator_llm_payload_includes_loaded_rules(tmp_path):
    agent_dir = tmp_path / "internal_orchestrator"
    knowledge_dir = agent_dir / "knowledge"
    knowledge_dir.mkdir(parents=True)
    instructions = agent_dir / "instructions.md"
    instructions.write_text("core instructions", encoding="utf-8")
    (knowledge_dir / "routing.md").write_text("routing chapter for current task", encoding="utf-8")
    (agent_dir / "rule_index.yaml").write_text(
        """
rules:
  - id: routing
    title: Routing
    file: knowledge/routing.md
    priority: 100
    use_when:
      default_for_orchestrator: true
    load_reason: route
""",
        encoding="utf-8",
    )
    client = RecordingLLMClient(
        '{"status":"completed","answer":"ok","confidence":0.7,"tool_calls":[{"name":"none","args":{},"summary":""}]}'
    )

    result = asyncio.run(
        OrchestratorLLMService(client).decide(
            manifest=_manifest(str(instructions)),
            task=AgentTask(task_id="t1", request="найди датчик коленвала"),
            dialog_history=[],
            retrieval_hits=[],
            tool_definitions=[],
        )
    )

    system_prompt = client.calls[0]["messages"][0]["content"]
    payload = json.loads(client.calls[0]["messages"][1]["content"])

    assert "core instructions" in system_prompt
    assert "routing chapter for current task" in system_prompt
    assert payload["loaded_rules"][0]["id"] == "routing"
    assert payload["loaded_rules"][0]["file"] == "knowledge/routing.md"
    assert result.raw["loaded_rules"][0]["id"] == "routing"
