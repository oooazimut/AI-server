from __future__ import annotations

import asyncio

from ai_server.agents.logistics import LogisticsSpecialist
from ai_server.models import AgentTask, ToolResult, ToolStatus, UserContext
from ai_server.orchestrators.logistics_response import render_logistics_tool_result
from ai_server.registry import get_agent_manifest
from ai_server.settings import get_settings
from tests.fakes import FakeVehicleUsageStore


def _specialist(monkeypatch) -> LogisticsSpecialist:
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    return LogisticsSpecialist.build(
        manifest,
        vehicle_usage_store=FakeVehicleUsageStore(),
        settings=get_settings(),
    )


def _task() -> AgentTask:
    return AgentTask(
        task_id="logistics-test",
        request="покажи отчёт по машинам",
        user=UserContext(id="1"),
        context={"dialog_key": "1:2026-07-23:101", "dialog_id": "chat1"},
    )


def test_logistics_registry_exposes_only_structured_tools(monkeypatch):
    specialist = _specialist(monkeypatch)

    registry = specialist.capability_registry()

    assert registry["registry_version"]
    assert {item["id"] for item in registry["tools"]} == set(specialist.manifest.tools)
    assert all(item["structured_command"] for item in registry["tools"])
    assert specialist.llm is None


def test_logistics_executes_exact_structured_command_without_model(monkeypatch):
    specialist = _specialist(monkeypatch)
    registry = specialist.capability_registry()

    result = asyncio.run(
        specialist.execute_structured_command(
            _task(),
            {
                "registry_version": registry["registry_version"],
                "tool_name": "vehicle_usage_get_report",
                "arguments": {"request_date": "2026-07-23"},
            },
        )
    )

    assert result.status == "completed"
    assert result.metadata["formatter_domain"] == "logistics"
    assert result.metadata["tool_result"]["tool"] == "vehicle_usage_get_report"
    assert result.model_usage[0].status == "not_used"


def test_logistics_rejects_free_text_and_stale_registry(monkeypatch):
    specialist = _specialist(monkeypatch)

    free_text = asyncio.run(specialist.handle(_task()))
    stale = asyncio.run(
        specialist.execute_structured_command(
            _task(),
            {
                "registry_version": "stale",
                "tool_name": "vehicle_usage_get_report",
                "arguments": {},
            },
        )
    )

    assert free_text.metadata["reason"] == "ORCHESTRATOR_STRUCTURED_COMMAND_REQUIRED"
    assert stale.metadata["reason"] == "CAPABILITY_REGISTRY_VERSION_MISMATCH"


def test_logistics_result_is_rendered_by_orchestrator():
    rendered = render_logistics_tool_result(
        tool_result=ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_get_employee_period_report",
            data={
                "employee_name": "Борисов Андрей",
                "date_from": "2026-07-20",
                "date_to": "2026-07-23",
                "days": [{"status_date": "2026-07-23", "status": "on_car"}],
            },
        )
    )

    assert rendered.status == "completed"
    assert "Борисов Андрей" in rendered.answer
    assert "on_car" in rendered.answer
