from __future__ import annotations

import sqlite3

from ai_server.agents.logistics import LogisticsSpecialist
from ai_server.agents.logistics_llm import LogisticsLLMToolCall
from ai_server.models import AgentTask, UserContext
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.vehicle_usage import VehicleUsageStore, VehicleUsageToolset
from tests.fakes import FakeEmbeddingProvider, FakeLogisticsLLM


def test_logistics_specialist_saves_llm_parsed_draft(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("VEHICLE_USAGE_STAFF_ROSTER", "1|15|Иван Петров;2|16|Олег Сидоров")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = VehicleUsageStore(tmp_path / "vehicle_usage.sqlite")
    fake_llm = FakeLogisticsLLM(
        tool_call_steps=[
            [LogisticsLLMToolCall(name="vehicle_usage_context", args={"request_date": "2026-06-05"})],
            [
                LogisticsLLMToolCall(
                    name="vehicle_usage_save_draft",
                    args={
                        "request_date": "2026-06-05",
                        "response_text": "Иван на Ларгусе, Олег в офисе",
                        "parsed": {
                            "date": "2026-06-05",
                            "people": [
                                {"staff_order": 1, "full_name": "Иван Петров", "status": "shift", "vehicle_id": 1},
                                {"staff_order": 2, "full_name": "Олег Сидоров", "status": "shift"},
                            ],
                            "vehicles": [{"vehicle_id": 1, "employee_name": "Иван Петров"}],
                        },
                    },
                )
            ],
            [LogisticsLLMToolCall(name="none")],
        ],
        final_answer="Сохранил черновик.",
    )
    specialist = LogisticsSpecialist(
        manifest,
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        tools=VehicleUsageToolset(store=store, user_id=9, dialog_id="chat9"),
        llm=fake_llm,
    )

    result = anyio_run(
        specialist.handle(
            AgentTask(
                task_id="log-1",
                request="Иван на Ларгусе, Олег в офисе",
                user=UserContext(id="9", raw={"dialog_id": "chat9"}),
            )
        )
    )

    assert result.answer == "Сохранил черновик."
    assert any(action.name == "logistics_vehicle_usage_context" for action in result.actions_taken)
    assert any(action.name == "logistics_vehicle_usage_save_draft" for action in result.actions_taken)
    with sqlite3.connect(store.path) as db:
        row = db.execute("SELECT status, parsed_json FROM vehicle_usage_requests").fetchone()
    assert row[0] == "pending_confirmation"
    assert "Иван Петров" in row[1]


def test_logistics_specialist_saves_confirmed_report(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("VEHICLE_USAGE_STAFF_ROSTER", "1|15|Иван Петров")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = VehicleUsageStore(tmp_path / "vehicle_usage.sqlite")
    fake_llm = FakeLogisticsLLM(
        tool_call_steps=[
            [LogisticsLLMToolCall(name="vehicle_usage_context", args={"request_date": "2026-06-05"})],
            [
                LogisticsLLMToolCall(
                    name="vehicle_usage_save_report",
                    args={
                        "request_date": "2026-06-05",
                        "source_text": "подтверждаю",
                        "parsed": {
                            "date": "2026-06-05",
                            "people": [{"staff_order": 1, "full_name": "Иван Петров", "status": "shift"}],
                            "vehicles": [{"vehicle_id": 1, "employee_name": "Иван Петров"}],
                        },
                    },
                )
            ],
            [LogisticsLLMToolCall(name="none")],
        ],
        final_answer="Сохранил утренний отчет.",
    )
    specialist = LogisticsSpecialist(
        manifest,
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        tools=VehicleUsageToolset(store=store, user_id=9, dialog_id="chat9"),
        llm=fake_llm,
    )

    result = anyio_run(
        specialist.handle(AgentTask(task_id="log-2", request="подтверждаю", user=UserContext(id="9")))
    )

    assert result.answer == "Сохранил утренний отчет."
    with sqlite3.connect(store.path) as db:
        request = db.execute("SELECT status FROM vehicle_usage_requests").fetchone()
        status = db.execute("SELECT status FROM employee_daily_statuses").fetchone()
        assignment = db.execute("SELECT vehicle_id, employee_id FROM vehicle_daily_assignments").fetchone()
    assert request[0] == "answered"
    assert status[0] == "shift"
    assert assignment == (1, 1)


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)
