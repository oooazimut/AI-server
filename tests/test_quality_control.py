import asyncio

from ai_server.result_templates import active_result_templates_context
from ai_server.workers.bitrix.quality_control import (
    QualityControlDecision,
    QualityControlToolCall,
    handle_quality_control_webhook_event,
)


def _task_update_payload(task_id: int = 101) -> dict:
    return {
        "event": "onTaskUpdate",
        "data": {
            "FIELDS_AFTER": {
                "ID": str(task_id),
            }
        },
    }


def test_quality_control_dry_run_does_not_write(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "true")
    monkeypatch.setenv("QUALITY_CONTROL_AUTO_MANAGE_PROJECT_ID", "44")
    monkeypatch.setenv("QUALITY_CONTROL_NOTIFY_DIRECTOR", "false")
    monkeypatch.setenv("QUALITY_CONTROL_NOTIFY_RESPONSIBLE", "false")

    bitrix = FakeQualityBitrix()
    result = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(),
            quality_llm=FakeQualityControlLLM(valid=False),
            status={},
        )
    )

    assert result["handled"] is True
    assert result["valid"] is False
    assert result["actions"] == ["disapprove"]
    assert bitrix.calls == [
        ("get_task", 101),
        ("list_task_results", 101),
    ]


def test_quality_control_disapproves_invalid_result(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")
    monkeypatch.setenv("QUALITY_CONTROL_AUTO_MANAGE_PROJECT_ID", "44")
    monkeypatch.setenv("QUALITY_CONTROL_NOTIFY_DIRECTOR", "false")
    monkeypatch.setenv("QUALITY_CONTROL_NOTIFY_RESPONSIBLE", "false")

    bitrix = FakeQualityBitrix()
    result = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(),
            quality_llm=FakeQualityControlLLM(valid=False),
            status={},
        )
    )

    assert result["handled"] is True
    assert result["actions"] == ["disapprove"]
    assert ("disapprove_task", 101) in bitrix.calls
    assert ("add_task_comment", 101) in [call[:2] for call in bitrix.calls]

    duplicate = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(),
            quality_llm=FakeQualityControlLLM(valid=False),
            status={},
        )
    )

    assert duplicate["duplicate"] is True
    assert [call[0] for call in bitrix.calls].count("disapprove_task") == 1


def test_quality_control_approves_valid_waiting_control_result(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path))
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")
    monkeypatch.setenv("QUALITY_CONTROL_AUTO_MANAGE_PROJECT_ID", "44")
    monkeypatch.setenv("QUALITY_CONTROL_NOTIFY_DIRECTOR", "false")
    monkeypatch.setenv("QUALITY_CONTROL_NOTIFY_RESPONSIBLE", "false")

    bitrix = FakeQualityBitrix()
    result = asyncio.run(
        handle_quality_control_webhook_event(
            bitrix,
            payload=_task_update_payload(),
            quality_llm=FakeQualityControlLLM(valid=True),
            status={},
        )
    )

    assert result["handled"] is True
    assert result["valid"] is True
    assert result["actions"] == ["approve"]
    assert ("approve_task", 101) in bitrix.calls
    assert ("add_task_comment", 101) in [call[:2] for call in bitrix.calls]


def test_result_templates_catalog_contains_default_template():
    context = active_result_templates_context()

    assert context["templates"][0]["id"] == "default_result_v1"
    assert "Первая строка" in context["templates"][0]["rules"][0]


class FakeQualityControlLLM:
    def __init__(self, *, valid: bool) -> None:
        self.valid = valid
        self.calls = []

    async def decide(self, **kwargs):
        self.calls.append(kwargs)
        tool_results = kwargs["tool_results"]
        if not any(result["tool"] == "bitrix_task_get" for result in tool_results):
            return QualityControlDecision(
                status="continue",
                answer="Нужно прочитать задачу и результат.",
                tool_calls=[
                    QualityControlToolCall(name="bitrix_task_get", args={"task_id": kwargs["task_id"]}),
                    QualityControlToolCall(name="bitrix_task_results_list", args={"task_id": kwargs["task_id"]}),
                ],
            )
        return QualityControlDecision(
            status="completed",
            answer="Решение принято моделью.",
            tool_calls=[
                QualityControlToolCall(
                    name="quality_control_action",
                    args={
                        "action": "approve" if self.valid else "return_to_work",
                        "validation": {
                            "valid": self.valid,
                            "outcome": "all_done" if self.valid else "not_all_done",
                            "issues": [] if self.valid else ["LLM считает результат недостаточным."],
                            "fixes": [] if self.valid else ["Уточнить выполненные пункты."],
                        },
                    },
                )
            ],
        )


class FakeQualityBitrix:
    def __init__(self) -> None:
        self.calls = []

    async def get_task(self, task_id, *, select=None):
        self.calls.append(("get_task", task_id))
        return {
            "task": {
                "id": str(task_id),
                "title": "Проверить IP-камеру",
                "description": "Проверить доступность камеры и время на устройстве.",
                "status": "4",
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

    async def disapprove_task(self, task_id):
        self.calls.append(("disapprove_task", task_id))
        return {"ok": True}

    async def approve_task(self, task_id):
        self.calls.append(("approve_task", task_id))
        return {"ok": True}

    async def renew_task(self, task_id):
        self.calls.append(("renew_task", task_id))
        return {"ok": True}

    async def add_task_comment(self, *, task_id, message, author_id=None):
        self.calls.append(("add_task_comment", task_id, message))
        return {"ok": True}

    async def notify_user(self, *, user_id, message, tag="ai_server", sub_tag=""):
        self.calls.append(("notify_user", user_id, message, tag, sub_tag))
        return {"ok": True}
