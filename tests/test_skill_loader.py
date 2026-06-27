import asyncio
import json

from ai_server.agents.bitrix24.llm import BitrixLLMService
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.settings import get_settings
from ai_server.skill_loader import load_skills_for_task
from tests.fakes import RecordingLLMClient


def test_bitrix_skill_loader_selects_catalog_for_product_request():
    manifest = get_agent_manifest("bitrix24")

    skills = load_skills_for_task(
        manifest,
        request="найди датчик коленвала",
        context={},
    )

    assert [skill.id for skill in skills] == ["catalog"]
    assert "датчик" in skills[0].matched_keywords
    assert "Складской учёт" in skills[0].content


def test_bitrix_llm_payload_includes_loaded_skills(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    client = RecordingLLMClient(
        '{"status":"completed","answer":"ok","confidence":0.7,"tool_calls":[{"name":"none","args":{},"summary":""}]}'
    )
    manifest = get_agent_manifest("bitrix24")

    result = asyncio.run(
        BitrixLLMService(client, settings=get_settings()).decide(
            manifest=manifest,
            task=AgentTask(task_id="t1", request="найди датчик коленвала"),
            retrieval_hits=[],
            tool_definitions=[],
        )
    )

    system_prompt = client.calls[0]["messages"][0]["content"]
    payload = json.loads(client.calls[0]["messages"][1]["content"])

    assert "Подгруженные скилы" in system_prompt
    assert "Складской учёт и каталог товаров" in system_prompt
    assert payload["loaded_skills"][0]["id"] == "catalog"
    assert payload["loaded_skills"][0]["file"] == "skills/catalog.md"
    assert result.raw["loaded_skills"][0]["id"] == "catalog"
