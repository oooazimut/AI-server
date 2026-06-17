from __future__ import annotations

import asyncio

from ai_server.models import ActionRecord, AgentResult, AgentTask
from ai_server.result_templates import active_result_templates_context
from ai_server.workers.bitrix.quality_control import handle_quality_control_webhook_event


def _task_update_payload(task_id: int = 101) -> dict:
    return {
        "event": "onTaskUpdate",
        "data": {
            "FIELDS_AFTER": {
                "ID": str(task_id),
            }
        },
    }


class FakeQualityBitrix:
    def __init__(self, *, status: str = "4", group_id: str = "44") -> None:
        self.calls: list = []
        self._status = status
        self._group_id = group_id

    async def get_task(self, task_id, *, select=None):
        self.calls.append(("get_task", task_id))
        return {
            "task": {
                "id": str(task_id),
                "title": "Проверить IP-камеру",
                "description": "Проверить доступность камеры и время на устройстве.",
                "status": self._status,
                "responsibleId": "9",
                "createdBy": "1",
                "groupId": self._group_id,
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


class FakeSpecialist:
    def __init__(self, *, actions: list[str] | None = None, answer: str = "Проверено.") -> None:
        self.calls: list[AgentTask] = []
        self._actions = actions or ["bitrix_api"]
        self._answer = answer

    async def handle(self, task: AgentTask) -> AgentResult:
        self.calls.append(task)
        return AgentResult(
            status="completed",
            agent_id="bitrix24",
            answer=self._answer,
            actions_taken=[ActionRecord(name=a, status="ok", details={}) for a in self._actions],
        )


def test_quality_control_calls_specialist_for_status4_task(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")

    bitrix = FakeQualityBitrix(status="4")
    specialist = FakeSpecialist(actions=["bitrix_api", "bitrix_api"])

    result = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(101),
            status={},
            specialist=specialist,
        )
    )

    assert result["handled"] is True
    assert result["task_id"] == 101
    assert len(specialist.calls) == 1
    called_task = specialist.calls[0]
    assert called_task.task_id == "qc_101"
    assert called_task.source == "quality_control_webhook"
    assert called_task.context["task_detail"]["status"] == "4"
    assert result["actions"] == ["bitrix_api", "bitrix_api"]


def test_quality_control_skips_non_status4_task(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")

    bitrix = FakeQualityBitrix(status="5")
    specialist = FakeSpecialist()

    result = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(202),
            status={},
            specialist=specialist,
        )
    )

    assert result["handled"] is False
    assert "status_5" in result["reason"]
    assert len(specialist.calls) == 0


def test_quality_control_disabled_skips(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "false")

    bitrix = FakeQualityBitrix(status="4")
    specialist = FakeSpecialist()

    result = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(303),
            status={},
            specialist=specialist,
        )
    )

    assert result["handled"] is False
    assert result["reason"] == "disabled"
    assert len(specialist.calls) == 0


def test_quality_control_deduplication(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")

    bitrix = FakeQualityBitrix(status="4")
    specialist = FakeSpecialist()

    first = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(404),
            status={},
            specialist=specialist,
        )
    )
    assert first["handled"] is True

    duplicate = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(404),
            status={},
            specialist=specialist,
        )
    )
    assert duplicate["handled"] is False
    assert duplicate.get("duplicate") is True
    assert len(specialist.calls) == 1


def test_quality_control_no_specialist_marks_done_with_empty_actions(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")

    bitrix = FakeQualityBitrix(status="4")

    result = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(505),
            status={},
            specialist=None,
        )
    )

    assert result["handled"] is True
    assert result["actions"] == []


def test_result_templates_catalog_contains_default_template():
    context = active_result_templates_context()

    assert context["templates"][0]["id"] == "default_result_v1"
    assert "Первая строка" in context["templates"][0]["rules"][0]
