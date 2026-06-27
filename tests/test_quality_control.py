from __future__ import annotations

import asyncio

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.bitrix24.quality_control import handle_quality_control_task
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.result_templates import active_result_templates_context
from ai_server.settings import get_settings
from tests.fakes import FakeBitrixLLM


class FakeQualityBitrix:
    def __init__(self, *, status: str = "4") -> None:
        self.calls: list = []
        self._status = status

    async def get_task(self, task_id, *, select=None):
        self.calls.append(("get_task", task_id))
        return {
            "task": {
                "id": str(task_id),
                "title": "Проверить IP-камеру",
                "description": "Проверить доступность камеры.",
                "status": self._status,
                "responsibleId": "9",
                "createdBy": "1",
                "groupId": "44",
                "taskControl": "Y",
                "changedDate": "2026-06-04T10:00:00+03:00",
            }
        }

    async def list_task_results(self, task_id):
        self.calls.append(("list_task_results", task_id))
        return [
            {
                "id": "501",
                "text": "Готово.",
                "createdBy": "9",
                "createdAt": "2026-06-04T10:01:00+03:00",
            }
        ]


def _qc_task(task_id: int = 101) -> AgentTask:
    return AgentTask(
        task_id=f"qc_input_{task_id}",
        request="quality_control",
        context={"bitrix_event_type": "ONTASKUPDATE", "task_id": task_id},
    )


def _specialist() -> Bitrix24Specialist:
    return Bitrix24Specialist(get_agent_manifest("bitrix24"), llm=FakeBitrixLLM())


def test_quality_control_calls_llm_for_status4_task(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")

    bitrix = FakeQualityBitrix(status="4")
    specialist = _specialist()

    result = asyncio.run(handle_quality_control_task(specialist, _qc_task(101), bitrix=bitrix, settings=get_settings()))

    assert result.status in ("completed", "needs_human", "failed")
    assert ("get_task", 101) in bitrix.calls
    assert ("list_task_results", 101) in bitrix.calls


def test_quality_control_skips_non_status4_task(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")

    bitrix = FakeQualityBitrix(status="5")
    specialist = _specialist()

    result = asyncio.run(handle_quality_control_task(specialist, _qc_task(202), bitrix=bitrix, settings=get_settings()))

    assert result.status == "completed"
    assert "status_5" in result.answer
    assert ("list_task_results", 202) not in bitrix.calls


def test_quality_control_disabled_skips(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "false")

    bitrix = FakeQualityBitrix(status="4")
    specialist = _specialist()

    result = asyncio.run(handle_quality_control_task(specialist, _qc_task(303), bitrix=bitrix, settings=get_settings()))

    assert result.status == "completed"
    assert "quality_control_disabled" in result.answer
    assert bitrix.calls == []


def test_quality_control_deduplication(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")

    bitrix = FakeQualityBitrix(status="4")
    specialist = _specialist()
    settings = get_settings()

    first = asyncio.run(handle_quality_control_task(specialist, _qc_task(404), bitrix=bitrix, settings=settings))
    assert first.status in ("completed", "needs_human")

    bitrix2 = FakeQualityBitrix(status="4")
    duplicate = asyncio.run(handle_quality_control_task(specialist, _qc_task(404), bitrix=bitrix2, settings=settings))

    assert "already_processed" in duplicate.answer


def test_result_templates_catalog_contains_default_template():
    context = active_result_templates_context()

    assert context["templates"][0]["id"] == "default_result_v1"
    assert "Первая строка" in context["templates"][0]["rules"][0]
