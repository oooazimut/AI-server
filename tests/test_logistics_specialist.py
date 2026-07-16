from __future__ import annotations

import json
from datetime import datetime, timedelta

from ai_server.agents.logistics import (
    LogisticsLLMDecision,
    LogisticsLLMService,
    LogisticsLLMToolCall,
    LogisticsSpecialist,
)
from ai_server.agents.logistics.llm import _decision_system_prompt
from ai_server.agents.logistics.specialist import VehicleUsageSettings
from ai_server.agents.logistics.tools import (
    VehicleCancelReportTool,
    VehicleContextTool,
    VehicleGetOperatorsTool,
    VehicleGetReportTool,
    VehicleReferenceTool,
    VehicleSaveDraftTool,
    VehicleSaveReportTool,
    VehicleSetOperatorsTool,
    VehicleStartDayTool,
)
from ai_server.models import ActionRecord, AgentResult, AgentTask, ToolResult, ToolStatus, UserContext
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.vehicle_usage import SentRequestData, StaffMember, resolve_vehicle_usage_operator_ids
from tests.fakes import FakeEmbeddingProvider, FakeLogisticsLLM, FakeVehicleUsageStore, RecordingLLMClient

_FAKE_RETRIEVER = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())

_VU_SETTINGS = VehicleUsageSettings(
    manager_user_id=5,
    max_reminders=3,
    reminder_interval_minutes=30,
    dry_run=True,
)


def _specialist(
    manifest,
    store: FakeVehicleUsageStore,
    *,
    llm=None,
    vu_settings: VehicleUsageSettings | None = None,
) -> LogisticsSpecialist:
    return LogisticsSpecialist(
        manifest,
        retriever=_FAKE_RETRIEVER,
        agent_tools=[
            VehicleContextTool(store),
            VehicleGetReportTool(store),
            VehicleSaveDraftTool(store),
            VehicleSaveReportTool(store),
            VehicleCancelReportTool(store),
        ],
        llm=llm or FakeLogisticsLLM(),
        vu_settings=vu_settings,
    )


def test_logistics_specialist_forwards_dialog_history_to_decide(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    llm = FakeLogisticsLLM()
    history = [
        {"role": "user", "content": "кто сегодня на смене"},
        {"role": "assistant", "content": "Уточните дату смены."},
    ]
    store = FakeVehicleUsageStore()

    specialist = _specialist(manifest, store, llm=llm)
    anyio_run(
        specialist.handle(
            AgentTask(
                task_id="log-history",
                request="на сегодня",
                user=UserContext(id="9"),
                context={"dialog_history": history},
            )
        )
    )

    assert llm.decide_calls[0]["dialog_history"] == history
    assert llm.decide_calls[0]["task"].context == {"dialog_history": history}


def test_logistics_llm_decide_payload_includes_dialog_history_and_raw_context(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    client = RecordingLLMClient(
        '{"status":"completed","answer":"","confidence":0.7,"tool_calls":[{"name":"none","args":{},"summary":""}]}'
    )
    history = [{"role": "user", "content": "кто сегодня на смене"}]
    scheduler_context = {"event": "vehicle_usage_reminder_due", "request_date": "2026-06-05", "dialog_history": history}

    anyio_run(
        LogisticsLLMService(client).decide(
            manifest=manifest,
            task=AgentTask(task_id="t1", request="на сегодня", context=scheduler_context),
            retrieval_hits=[],
            tool_definitions=[],
            dialog_history=history,
        )
    )

    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["dialog_history"] == history
    assert payload["context"] == scheduler_context


def test_logistics_llm_compose_formats_get_report_without_saving(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    client = RecordingLLMClient('{"status":"completed","answer":"wrong"}')
    service = LogisticsLLMService(client)

    result = anyio_run(
        service.compose(
            manifest=manifest,
            task=AgentTask(task_id="show-report", request="покажи отчет по машинам за 2026-07-02"),
            decision=LogisticsLLMDecision(
                status="completed",
                answer="",
                tool_calls=[LogisticsLLMToolCall(name="vehicle_usage_get_report", args={"request_date": "2026-07-02"})],
            ),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="vehicle_usage_get_report",
                    data={
                        "report_date": "2026-07-02",
                        "source": "vehicle_usage_requests.parsed_json",
                        "employee_statuses": [
                            {"full_name": "Борисов Андрей", "status": "на авто", "vehicle": "Авто 2"},
                        ],
                        "vehicle_assignments": [
                            {"vehicle_name": "Авто 2", "status": "в работе", "drivers": ["Борисов Андрей"]},
                        ],
                    },
                )
            ],
        )
    )

    assert not client.calls
    assert "Отчет по машинам за 2026-07-02" in result.answer
    assert "Борисов Андрей" in result.answer
    assert "Авто 2" in result.answer
    assert "сохрани" not in result.answer.casefold()


def test_logistics_llm_compose_translates_normalized_vehicle_usage_statuses(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    client = RecordingLLMClient('{"status":"completed","answer":"wrong"}')
    service = LogisticsLLMService(client)

    result = anyio_run(
        service.compose(
            manifest=manifest,
            task=AgentTask(task_id="show-normalized-report", request="покажи отчет по машинам за 2026-07-07"),
            decision=LogisticsLLMDecision(
                status="completed",
                answer="",
                tool_calls=[LogisticsLLMToolCall(name="vehicle_usage_get_report", args={"request_date": "2026-07-07"})],
            ),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="vehicle_usage_get_report",
                    data={
                        "report_date": "2026-07-07",
                        "source": "normalized_tables",
                        "employee_statuses": [
                            {"full_name": "Борисов Андрей", "status": "worked", "vehicle_name": "Авто 2"},
                            {"full_name": "Ивашин Павел", "status": "car", "car_assigned": "Авто 3"},
                            {"full_name": "Абдураимова Галина", "status": "on_leave"},
                            {"full_name": "Марат", "status": "in_office"},
                            {"full_name": "Никаненок Алексейи", "status": "on_shift"},
                        ],
                        "vehicle_assignments": [
                            {
                                "vehicle_id": 2,
                                "vehicle_name": "Авто 2",
                                "status": "in_use",
                                "drivers": ["Борисов Андрей"],
                            },
                            {
                                "vehicle_id": 5,
                                "vehicle_name": "Авто 5",
                                "status": "idle",
                                "drivers": [],
                                "notes": "простой",
                            },
                        ],
                    },
                )
            ],
        )
    )

    assert "Борисов Андрей — работал / Авто 2" in result.answer
    assert "Ивашин Павел — работал / Авто 3" in result.answer
    assert "Абдураимова Галина — отпуск" in result.answer
    assert "Марат — работал" in result.answer
    assert "Никаненок Алексейи — работал" in result.answer
    assert "Авто 2 — Борисов Андрей / в работе" in result.answer
    assert "Авто 5 — простой" in result.answer
    assert "Авто 5 — простой / простой" not in result.answer
    assert "in_use" not in result.answer
    assert "idle" not in result.answer
    assert "on_leave" not in result.answer
    assert "in_office" not in result.answer
    assert "on_shift" not in result.answer
    assert " car" not in result.answer


def test_logistics_llm_compose_formats_update_report_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    client = RecordingLLMClient('{"status":"completed","answer":"wrong"}')
    service = LogisticsLLMService(client)

    result = anyio_run(
        service.compose(
            manifest=manifest,
            task=AgentTask(task_id="update-report", request="исправь отчет за 2026-07-07"),
            decision=LogisticsLLMDecision(
                status="completed",
                answer="",
                tool_calls=[LogisticsLLMToolCall(name="vehicle_usage_update_report", args={})],
            ),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="vehicle_usage_update_report",
                    data={"report_date": "2026-07-07", "employee_updates": 0, "vehicle_updates": 1},
                )
            ],
        )
    )

    assert not client.calls
    assert "Отчет по машинам за 2026-07-07 обновлен" in result.answer
    assert "машины: 1" in result.answer


def test_logistics_llm_compose_translates_period_summary_statuses(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    client = RecordingLLMClient('{"status":"completed","answer":"wrong"}')
    service = LogisticsLLMService(client)

    result = anyio_run(
        service.compose(
            manifest=manifest,
            task=AgentTask(task_id="vehicle-period", request="покажи отчет по Авто 3 за июль"),
            decision=LogisticsLLMDecision(
                status="completed",
                answer="",
                tool_calls=[LogisticsLLMToolCall(name="vehicle_usage_get_vehicle_period_report", args={})],
            ),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="vehicle_usage_get_vehicle_period_report",
                    data={
                        "vehicle_name": "Авто 3",
                        "date_from": "2026-07-01",
                        "date_to": "2026-07-31",
                        "days": [{"assignment_date": "2026-07-09", "status": "in_use", "drivers": ["Ивашин Павел"]}],
                        "summary": {"in_use": 1, "not_required": 1},
                    },
                )
            ],
        )
    )

    assert "в работе: 1" in result.answer
    assert "не требуется: 1" in result.answer
    assert "in_use" not in result.answer
    assert "not_required" not in result.answer


def test_logistics_llm_compose_formats_save_report_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    client = RecordingLLMClient('{"status":"completed","answer":"wrong"}')
    service = LogisticsLLMService(client)

    result = anyio_run(
        service.compose(
            manifest=manifest,
            task=AgentTask(task_id="save-report", request="подтверждаю отчет за 2026-07-08"),
            decision=LogisticsLLMDecision(
                status="completed",
                answer="",
                tool_calls=[LogisticsLLMToolCall(name="vehicle_usage_save_report", args={})],
            ),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="vehicle_usage_save_report",
                    data={
                        "request_date": "2026-07-08",
                        "staff_entries_saved": 11,
                        "vehicle_assignments_saved": 8,
                        "vehicles_saved": 6,
                    },
                )
            ],
        )
    )

    assert not client.calls
    assert "Финальный отчет по машинам за 2026-07-08 сохранен" in result.answer
    assert "сотрудники: 11" in result.answer
    assert "машины: 6" in result.answer
    assert "машины: 8" not in result.answer


def test_logistics_llm_compose_formats_incomplete_save_report_without_llm(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    client = RecordingLLMClient('{"status":"completed","answer":"wrong"}')
    service = LogisticsLLMService(client)

    result = anyio_run(
        service.compose(
            manifest=manifest,
            task=AgentTask(task_id="save-report-incomplete", request="подтверждаю отчет"),
            decision=LogisticsLLMDecision(
                status="completed",
                answer="",
                tool_calls=[LogisticsLLMToolCall(name="vehicle_usage_save_report", args={})],
            ),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="vehicle_usage_save_report",
                    data={
                        "request_date": "2026-07-16",
                        "draft_saved": True,
                        "needs_clarification": True,
                        "questions": [
                            "Уточните статус сотрудников: Karasev Alexey.",
                            "Уточните статус машин: Auto 5.",
                        ],
                    },
                )
            ],
        )
    )

    assert result.status == "needs_clarification"
    assert not client.calls
    assert "Часть отчета по машинам за 2026-07-16 сохранил как черновик" in result.answer
    assert "Уточните статус сотрудников: Karasev Alexey." in result.answer
    assert "Уточните только эти пункты." in result.answer


def test_vehicle_usage_save_report_denies_unconfigured_user(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({13}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-05",
                "source_text": "confirmed",
                "parsed": {"date": "2026-07-05", "people": [], "vehicles": []},
            },
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.DENIED
    assert store._requests == []
    assert store._day_reports == []


def test_vehicle_usage_save_report_saves_incomplete_report_as_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees(
        [
            StaffMember(order=1, name="Borisov Andrey", user_id=27),
            StaffMember(order=2, name="Karasev Alexey", user_id=28),
        ]
    )
    store._vehicles = [
        {"id": 2, "brand_model": "Auto 2", "registration_number": ""},
        {"id": 5, "brand_model": "Auto 5", "registration_number": ""},
    ]
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({1}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-16",
                "source_text": "Borisov Auto 2",
                "parsed": {
                    "date": "2026-07-16",
                    "people": [{"staff_order": 1, "full_name": "Borisov Andrey", "status": "car"}],
                    "vehicles": [{"vehicle_name": "Auto 2", "drivers": ["Borisov Andrey"]}],
                },
            },
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.OK
    assert result.data["needs_clarification"] is True
    assert store._requests[-1]["status"] == "pending_clarification"
    assert store._requests[-1]["parsed"]["validation"]["needs_clarification"] is True
    assert store._day_reports == []
    assert any(block["kind"] == "employees" and "Karasev Alexey" in block["items"] for block in result.data["missing"])
    assert any(block["kind"] == "vehicles" and "Auto 5" in block["items"] for block in result.data["missing"])


def test_vehicle_usage_save_report_marks_vehicle_drivers_worked(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, name="Борисов Андрей", user_id=27)])
    store._vehicles = [{"id": 2, "brand_model": "Авто 2", "registration_number": ""}]
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({1}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-07",
                "source_text": "confirmed",
                "parsed": {
                    "date": "2026-07-07",
                    "people": [{"staff_order": 1, "full_name": "Борисов Андрей", "status": "office"}],
                    "vehicles": [{"vehicle_name": "Авто 2", "drivers": ["Борисов Андрей"]}],
                },
            },
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.OK
    assert store._day_reports[-1]["employee_statuses"] == [(1, "worked", "")]


def test_vehicle_usage_save_report_recovers_vehicle_before_employee_phrase(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, name="Борисов Андрей", user_id=27)])
    store._vehicles = [{"id": 2, "brand_model": "Авто 2", "registration_number": ""}]
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({1}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-10",
                "source_text": "Борисов Андрей работает Авто 2 Борисов Андрей",
                "parsed": {
                    "date": "2026-07-10",
                    "people": [{"staff_order": 1, "full_name": "Борисов Андрей", "status": "работает"}],
                    "vehicles": [{"vehicle_name": "", "drivers": ["Борисов Андрей"]}],
                },
            },
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.OK
    assert store._day_reports[-1]["employee_statuses"] == [(1, "worked", "")]
    assert store._day_reports[-1]["vehicle_assignments"] == [(2, 1, "in_use", "")]


def test_vehicle_usage_save_report_does_not_infer_extra_drivers_from_long_text(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store._vehicles = [
        {"id": 1, "brand_model": "Авто 1", "registration_number": ""},
        {"id": 2, "brand_model": "Авто 2", "registration_number": ""},
        {"id": 3, "brand_model": "Авто 3", "registration_number": ""},
    ]
    store.upsert_employees(
        [
            StaffMember(order=1, name="Борисов Андрей", user_id=27),
            StaffMember(order=2, name="Карасев Алексей", user_id=28),
            StaffMember(order=3, name="Ивашин Павел", user_id=29),
            StaffMember(order=4, name="Смородин Андрей", user_id=30),
            StaffMember(order=5, name="Тищенко Алексей", user_id=31),
        ]
    )
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({1}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-13",
                "source_text": (
                    "Борисов Андрей Авто 2 Ивашин Павел Авто 3 Карасев Алексей Авто 2 "
                    "Смородин Андрей Авто 3 Тищенко Алексей Авто 1"
                ),
                "parsed": {
                    "date": "2026-07-13",
                    "people": [
                        {"staff_order": 1, "full_name": "Борисов Андрей", "status": "работает"},
                        {"staff_order": 2, "full_name": "Карасев Алексей", "status": "работает"},
                        {"staff_order": 3, "full_name": "Ивашин Павел", "status": "работает"},
                        {"staff_order": 4, "full_name": "Смородин Андрей", "status": "работает"},
                        {"staff_order": 5, "full_name": "Тищенко Алексей", "status": "работает"},
                    ],
                    "vehicles": [
                        {"vehicle_name": "Авто 2", "drivers": ["Борисов Андрей", "Карасев Алексей"]},
                        {"vehicle_name": "Авто 3", "drivers": ["Ивашин Павел", "Смородин Андрей"]},
                        {"vehicle_name": "Авто 1", "drivers": ["Тищенко Алексей"]},
                    ],
                },
            },
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.OK
    assert store._day_reports[-1]["employee_statuses"] == [
        (1, "worked", ""),
        (2, "worked", ""),
        (3, "worked", ""),
        (4, "worked", ""),
        (5, "worked", ""),
    ]
    assert store._day_reports[-1]["vehicle_assignments"] == [
        (2, 1, "in_use", ""),
        (2, 2, "in_use", ""),
        (3, 3, "in_use", ""),
        (3, 4, "in_use", ""),
        (1, 5, "in_use", ""),
    ]


def test_vehicle_usage_save_report_treats_person_vehicle_as_worked_on_vehicle(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, name="Борисов Андрей", user_id=27)])
    store._vehicles = [{"id": 2, "brand_model": "Авто 2", "registration_number": ""}]
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({1}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-11",
                "source_text": "Борисов Андрей Авто 2",
                "parsed": {
                    "date": "2026-07-11",
                    "people": [],
                    "vehicles": [{"vehicle_name": "Авто 2", "drivers": ["Борисов Андрей"]}],
                },
            },
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.OK
    assert store._day_reports[-1]["employee_statuses"] == [(1, "worked", "")]
    assert store._day_reports[-1]["vehicle_assignments"] == [(2, 1, "in_use", "")]


def test_vehicle_usage_save_report_treats_worked_person_vehicle_equally(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, name="Борисов Андрей", user_id=27)])
    store._vehicles = [{"id": 2, "brand_model": "Авто 2", "registration_number": ""}]
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({1}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-12",
                "source_text": "Борисов Андрей работает Авто 2 Борисов Андрей",
                "parsed": {
                    "date": "2026-07-12",
                    "people": [{"staff_order": 1, "full_name": "Борисов Андрей", "status": "работает"}],
                    "vehicles": [{"vehicle_name": "", "drivers": ["Борисов Андрей"]}],
                },
            },
            user_id=1,
            dialog_id="1",
        )
    )

    assert result.status == ToolStatus.OK
    assert store._day_reports[-1]["employee_statuses"] == [(1, "worked", "")]
    assert store._day_reports[-1]["vehicle_assignments"] == [(2, 1, "in_use", "")]


def test_vehicle_usage_admin_can_replace_operators(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    tool = VehicleSetOperatorsTool(store, admin_user_ids=frozenset({1}))

    result = anyio_run(tool.execute({"operator_user_ids": [13, 15, 13]}, user_id=1, dialog_id="1"))

    assert result.status == ToolStatus.OK
    assert result.data == {"operator_user_ids": [13, 15]}
    assert store.vehicle_usage_operator_ids() == {13, 15}


def test_vehicle_usage_get_operators_returns_only_operator_names(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees(
        [
            StaffMember(order=1, name="Абдураимова Галина", user_id=1),
            StaffMember(order=2, name="Коверга Дмитрий", user_id=13),
            StaffMember(order=3, name="Технический Пользователь", user_id=99),
        ]
    )
    store.set_vehicle_usage_operators(operator_user_ids=[13, 1], actor_user_id=1)
    tool = VehicleGetOperatorsTool(store)

    result = anyio_run(tool.execute({}, user_id=1, dialog_id="1"))

    assert result.status == ToolStatus.OK
    assert result.data == {
        "operator_user_ids": [1, 13],
        "operators": [
            {"user_id": 1, "full_name": "Абдураимова Галина"},
            {"user_id": 13, "full_name": "Коверга Дмитрий"},
        ],
    }


def test_vehicle_usage_reference_returns_staff_vehicles_and_operators(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, name="Borisov Andrey", user_id=27)])
    store._vehicles = [{"id": 2, "brand_model": "Auto 2", "registration_number": ""}]
    store.set_vehicle_usage_operators(operator_user_ids=[13], actor_user_id=1)
    tool = VehicleReferenceTool(store)

    result = anyio_run(tool.execute({}, user_id=1, dialog_id="1"))

    assert result.status == ToolStatus.OK
    assert result.data["staff_roster"] == [{"display_order": 1, "full_name": "Borisov Andrey", "user_id": 27}]
    assert result.data["vehicles"] == [{"id": 2, "brand_model": "Auto 2", "registration_number": ""}]
    assert result.data["operator_user_ids"] == [13]


def test_vehicle_save_report_drafts_unknown_vehicle_reference(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, name="Borisov Andrey", user_id=27)])
    store._vehicles = [{"id": 2, "brand_model": "Auto 2", "registration_number": ""}]
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({13}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-15",
                "source_text": "Borisov Andrey Auto 99",
                "parsed": {
                    "date": "2026-07-15",
                    "people": [{"staff_order": 1, "full_name": "Borisov Andrey", "status": "worked"}],
                    "vehicles": [{"vehicle_name": "Auto 99", "status": "in_use", "drivers": ["Borisov Andrey"]}],
                },
            },
            user_id=13,
            dialog_id="13",
        )
    )

    assert result.status == ToolStatus.OK
    assert result.data["draft_saved"] is True
    assert result.data["needs_clarification"] is True
    assert any(block["kind"] == "unknown_vehicle_references" for block in result.data["missing"])
    assert store._requests[-1]["status"] == "pending_clarification"
    assert store._day_reports == []


def test_logistics_llm_compose_formats_get_operators_without_saving(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    client = RecordingLLMClient('{"status":"completed","answer":"wrong"}')
    service = LogisticsLLMService(client)

    result = anyio_run(
        service.compose(
            manifest=manifest,
            task=AgentTask(task_id="operators", request="кто операторы отчета по машинам"),
            decision=LogisticsLLMDecision(
                status="completed",
                answer="",
                tool_calls=[LogisticsLLMToolCall(name="vehicle_usage_get_operators", args={})],
            ),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="vehicle_usage_get_operators",
                    data={
                        "operator_user_ids": [1, 13],
                        "operators": [
                            {"user_id": 1, "full_name": "Абдураимова Галина"},
                            {"user_id": 13, "full_name": "Коверга Дмитрий"},
                        ],
                    },
                )
            ],
        )
    )

    assert not client.calls
    assert "Операторы отчета по машинам" in result.answer
    assert "Абдураимова Галина (Bitrix ID 1)" in result.answer
    assert "Коверга Дмитрий (Bitrix ID 13)" in result.answer


def test_logistics_prompt_scopes_operator_list_to_vehicle_report_panel():
    prompt = _decision_system_prompt()

    assert "список операторов отчета по машинам/людям" in prompt
    assert "просит список операторов, вызови vehicle_usage_get_operators" not in prompt


def test_vehicle_usage_non_admin_cannot_replace_operators(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    tool = VehicleSetOperatorsTool(store, admin_user_ids=frozenset({1}))

    result = anyio_run(tool.execute({"operator_user_ids": [13]}, user_id=2, dialog_id="2"))

    assert result.status == ToolStatus.DENIED
    assert store.vehicle_usage_operator_ids() == set()


def test_vehicle_usage_operator_resolver_prefers_store_over_fallback(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.set_vehicle_usage_operators(operator_user_ids=[13, 1, 13], actor_user_id=1)

    assert resolve_vehicle_usage_operator_ids(store, frozenset({99})) == [1, 13]


def test_vehicle_usage_operator_resolver_uses_fallback_when_store_empty(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()

    assert resolve_vehicle_usage_operator_ids(store, frozenset({13, 1})) == [1, 13]


def test_vehicle_usage_reminders_are_per_operator_and_cancel_by_date(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    specialist = LogisticsSpecialist(
        manifest,
        retriever=_FAKE_RETRIEVER,
        agent_tools=[],
        llm=FakeLogisticsLLM(),
        vu_settings=_VU_SETTINGS,
    )

    reminder = specialist._build_scheduled_tasks(
        AgentTask(
            task_id="morning",
            user=UserContext(id="13"),
            request="vehicle_usage_morning",
            context={
                "event": "vehicle_usage_morning",
                "request_date": "2026-07-06",
                "channel_id": "bitrix24",
                "recipient_id": "13",
            },
        ),
        AgentResult(status="completed", agent_id="logistics", answer="send report"),
    )

    assert reminder[0].job_id == "vu_reminder_2026-07-06_13"
    assert reminder[0].task is not None
    assert reminder[0].task.context["recipient_id"] == "13"

    cancel = specialist._build_scheduled_tasks(
        AgentTask(
            task_id="save",
            user=UserContext(id="13"),
            request="done",
            context={"request_date": "2026-07-06", "recipient_id": "13"},
        ),
        AgentResult(
            status="completed",
            agent_id="logistics",
            answer="saved",
            actions_taken=[ActionRecord(name="vehicle_usage_save_report", status="ok")],
        ),
    )

    assert cancel[0].cancel is True
    assert cancel[0].job_id == "vu_reminder_2026-07-06"


def test_logistics_specialist_saves_llm_parsed_draft(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()
    store.upsert_employees(
        [StaffMember(order=1, user_id=15, name="Иван Петров"), StaffMember(order=2, user_id=16, name="Олег Сидоров")]
    )
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

    result = anyio_run(
        _specialist(manifest, store, llm=fake_llm).handle(
            AgentTask(
                task_id="log-1",
                request="Иван на Ларгусе, Олег в офисе",
                user=UserContext(id="9", raw={"dialog_id": "chat9"}),
            )
        )
    )

    assert result.answer == "Сохранил черновик."
    assert any(action.name == "vehicle_usage_context" for action in result.actions_taken)
    assert any(action.name == "vehicle_usage_save_draft" for action in result.actions_taken)
    assert store._requests
    saved = store._requests[-1]
    assert saved["status"] == "pending_confirmation"
    people = saved["parsed"].get("people", [])
    assert any(p.get("full_name") == "Иван Петров" for p in people)


def test_logistics_specialist_saves_confirmed_report(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, user_id=15, name="Иван Петров")])
    store._vehicles = [{"id": 1, "brand_model": "Авто 1", "registration_number": ""}]
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

    result = anyio_run(
        _specialist(manifest, store, llm=fake_llm).handle(
            AgentTask(task_id="log-2", request="подтверждаю", user=UserContext(id="9"))
        )
    )

    assert result.answer == "Сохранил утренний отчет."
    assert len(fake_llm.decide_calls) == 2
    assert result.metadata["fast_return"] is True
    assert result.metadata["terminal_tool"] == "vehicle_usage_save_report"
    assert any(action.name == "logistics_fast_return" for action in result.actions_taken)
    assert store._requests
    assert store._requests[-1]["status"] == "answered"
    assert store._day_reports
    report = store._day_reports[-1]
    assert report["employee_statuses"]
    assert report["employee_statuses"][0][1] == "worked"
    assert report["vehicle_assignments"]
    assert report["vehicle_assignments"][0][0] == 1


def test_logistics_specialist_saves_report_from_production_style_payload(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()
    store.upsert_employees(
        [
            StaffMember(order=1, user_id=15, name="Борисов Андрей"),
            StaffMember(order=2, user_id=16, name="Карасев Алексей"),
        ]
    )
    store._vehicles = [{"id": 2, "brand_model": "Авто 2", "registration_number": ""}]
    fake_llm = FakeLogisticsLLM(
        tool_call_steps=[
            [LogisticsLLMToolCall(name="vehicle_usage_context", args={"request_date": "2026-07-03"})],
            [
                LogisticsLLMToolCall(
                    name="vehicle_usage_save_report",
                    args={
                        "request_date": "2026-07-03",
                        "source_text": "отчет подтвержден",
                        "parsed": {
                            "staff": [
                                {"full_name": "Борисов Андрей", "status": "worked"},
                                {"full_name": "Карасев Алексей", "status": "worked"},
                            ],
                            "vehicles": [
                                {
                                    "vehicle_name": "Авто 2",
                                    "status": "in_use",
                                    "drivers": ["Борисов Андрей", "Карасев Алексей"],
                                }
                            ],
                        },
                    },
                )
            ],
            [LogisticsLLMToolCall(name="none")],
        ],
        final_answer="Отчет сохранен.",
    )

    anyio_run(
        _specialist(manifest, store, llm=fake_llm).handle(
            AgentTask(task_id="prod-style-report", request="подтверждаю", user=UserContext(id="9"))
        )
    )

    report = store._day_reports[-1]
    assert report["employee_statuses"] == [(1, "worked", ""), (2, "worked", "")]
    assert (2, 1, "in_use", "") in report["vehicle_assignments"]
    assert (2, 2, "in_use", "") in report["vehicle_assignments"]


def test_logistics_specialist_infers_vehicle_entries_from_report_text(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()
    store.upsert_employees(
        [
            StaffMember(order=1, user_id=15, name="Borisov Andrey"),
            StaffMember(order=2, user_id=16, name="Karasev Alexey"),
        ]
    )
    store._vehicles = [
        {"id": 2, "brand_model": "Auto 2", "registration_number": ""},
        {"id": 5, "brand_model": "Auto 5", "registration_number": ""},
    ]
    fake_llm = FakeLogisticsLLM(
        tool_call_steps=[
            [LogisticsLLMToolCall(name="vehicle_usage_context", args={"request_date": "2026-07-06"})],
            [
                LogisticsLLMToolCall(
                    name="vehicle_usage_save_report",
                    args={
                        "request_date": "2026-07-06",
                        "source_text": "\n".join(
                            [
                                "Employees:",
                                "- Borisov Andrey - car",
                                "- Karasev Alexey - car",
                                "Vehicles:",
                                "- Auto 2 - Borisov Andrey, Karasev Alexey / in use",
                                "- Auto 5 - idle",
                            ]
                        ),
                        "parsed": {
                            "date": "2026-07-06",
                            "people": [
                                {"full_name": "Borisov Andrey", "status": "car"},
                                {"full_name": "Karasev Alexey", "status": "car"},
                            ],
                        },
                    },
                )
            ],
            [LogisticsLLMToolCall(name="none")],
        ],
        final_answer="Saved.",
    )

    anyio_run(
        _specialist(manifest, store, llm=fake_llm).handle(
            AgentTask(task_id="infer-vehicles", request="confirm", user=UserContext(id="9"))
        )
    )

    assert store._requests[-1]["parsed"]["vehicles"][0]["vehicle_name"] == "Auto 2"
    report = store._day_reports[-1]
    assert report["employee_statuses"] == [(1, "worked", ""), (2, "worked", "")]
    assert (2, 1, "in_use", "") in report["vehicle_assignments"]
    assert (2, 2, "in_use", "") in report["vehicle_assignments"]
    assert (5, None, "idle", "") in report["vehicle_assignments"]


def test_logistics_specialist_cancels_report_as_day_off(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, user_id=15, name="Иван Петров")])
    fake_llm = FakeLogisticsLLM(
        tool_call_steps=[
            [LogisticsLLMToolCall(name="vehicle_usage_context", args={"request_date": "2026-07-05"})],
            [
                LogisticsLLMToolCall(
                    name="vehicle_usage_cancel_day",
                    args={"request_date": "2026-07-05", "reason": "Сегодня выходной, отчет не требуется."},
                )
            ],
            [LogisticsLLMToolCall(name="none")],
        ],
        final_answer="День отмечен как выходной.",
    )

    result = anyio_run(
        _specialist(manifest, store, llm=fake_llm, vu_settings=_VU_SETTINGS).handle(
            AgentTask(
                task_id="cancel-day",
                request="Отчет не требуется сегодня выходной",
                user=UserContext(id="9", raw={"dialog_id": "chat9"}),
                context={"request_date": "2026-07-05"},
            )
        )
    )

    assert store._requests[-1]["status"] == "cancelled_day_off"
    assert store._day_reports[-1]["employee_statuses"] == [(1, "day_off", "Сегодня выходной, отчет не требуется.")]
    assert len(result.scheduled_tasks) == 1
    assert result.scheduled_tasks[0].cancel is True


def test_manual_start_schedules_reminder_from_sent_at(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()
    store.set_vehicle_usage_operators(operator_user_ids=[13], actor_user_id=1)
    fake_llm = FakeLogisticsLLM(
        tool_call_steps=[
            [
                LogisticsLLMToolCall(
                    name="vehicle_usage_start_day",
                    args={"request_date": "2026-07-15", "message": "Заполните отчет по машинам за сегодня."},
                )
            ],
            [LogisticsLLMToolCall(name="none")],
        ],
        final_answer="Запустил отчет по машинам.",
    )

    result = anyio_run(
        LogisticsSpecialist(
            manifest,
            retriever=_FAKE_RETRIEVER,
            agent_tools=[VehicleStartDayTool(store, allowed_user_ids=frozenset({13}))],
            llm=fake_llm,
            vu_settings=_VU_SETTINGS,
        ).handle(
            AgentTask(
                task_id="manual-start",
                request="сегодня рабочий день, запусти отчет",
                user=UserContext(id="13", raw={"dialog_id": "13"}),
                context={"channel_id": "bitrix24", "recipient_id": "13", "dialog_id": "13"},
            )
        )
    )

    start_action = next(action for action in result.actions_taken if action.name == "vehicle_usage_start_day")
    sent_at = datetime.fromisoformat(start_action.details["data"]["sent_at"])
    assert len(result.scheduled_tasks) == 1
    sched = result.scheduled_tasks[0]
    assert sched.job_id == "vu_reminder_2026-07-15_13"
    assert sched.task is not None
    assert sched.task.context["event"] == "vehicle_usage_reminder_due"
    assert sched.task.context["reminder_count"] == 1
    assert sched.task.context["started_at"] == start_action.details["data"]["sent_at"]
    assert datetime.fromisoformat(sched.trigger["run_date"]) == sent_at + timedelta(minutes=30)


def test_auto_close_pending_draft_finalizes_unknowns(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees(
        [StaffMember(order=1, user_id=15, name="Иван Петров"), StaffMember(order=2, user_id=16, name="Олег Сидоров")]
    )
    store.create_sent_request(
        SentRequestData(
            request_date="2026-07-15",
            user_id=13,
            dialog_id="13",
            message="Заполните отчет по машинам за сегодня.",
            sent_at="2026-07-15T13:00:00+03:00",
            reminder_count=0,
            source="manual",
        )
    )
    store.save_draft(
        request_date="2026-07-15",
        user_id=13,
        dialog_id="13",
        response_text="Иван Петров работал",
        parsed={
            "date": "2026-07-15",
            "people": [{"staff_order": 1, "full_name": "Иван Петров", "status": "worked"}],
            "vehicles": [{"vehicle_id": 1, "vehicle_name": "Авто 1", "status": "idle"}],
        },
        status="pending_clarification",
    )

    result = store.auto_close_unanswered_day(
        report_date="2026-07-15",
        reason="No complete vehicle usage response was received by cutoff.",
    )

    assert result["status"] == "finalized_unknown"
    report = store.get_day_report(report_date="2026-07-15")
    statuses = {row[0]: row[1] for row in report["employee_statuses"]}
    vehicle_statuses = {row[0]: row[2] for row in report["vehicle_assignments"]}
    assert statuses[1] == "worked"
    assert statuses[2] == "unknown"
    assert vehicle_statuses[1] == "idle"
    assert vehicle_statuses[2] == "unknown"
    assert vehicle_statuses[3] == "unknown"


def test_auto_close_scheduled_pending_draft_keeps_previous_behavior(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.create_sent_request(
        SentRequestData(
            request_date="2026-07-15",
            user_id=13,
            dialog_id="13",
            message="Заполните отчет по машинам за сегодня.",
            sent_at="2026-07-15T08:30:00+03:00",
            reminder_count=0,
            source="scheduled",
        )
    )
    store.save_draft(
        request_date="2026-07-15",
        user_id=13,
        dialog_id="13",
        response_text="частичный отчет",
        parsed={"date": "2026-07-15", "people": [], "vehicles": []},
        status="pending_clarification",
    )

    result = store.auto_close_unanswered_day(
        report_date="2026-07-15",
        reason="No complete vehicle usage response was received by cutoff.",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "useful_response_exists"
    assert result["request_status"] == "pending_clarification"


def test_morning_task_schedules_reminder(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()
    fake_llm = FakeLogisticsLLM(final_answer="Привет, заполните отчёт по автомобилям за сегодня.")

    result = anyio_run(
        _specialist(manifest, store, llm=fake_llm, vu_settings=_VU_SETTINGS).handle(
            AgentTask(
                task_id="morning-1",
                request="Сгенерируй утренний отчёт.",
                context={
                    "channel_id": "bitrix24",
                    "recipient_id": "chat77",
                    "event": "vehicle_usage_morning",
                    "request_date": "2026-06-23",
                },
            )
        )
    )

    assert result.answer == "Привет, заполните отчёт по автомобилям за сегодня."
    assert len(result.scheduled_tasks) == 1
    sched = result.scheduled_tasks[0]
    assert sched.cancel is False
    assert sched.agent_id == "logistics"
    assert sched.job_id == "vu_reminder_2026-06-23_chat77"
    assert sched.task is not None
    assert sched.task.context["event"] == "vehicle_usage_reminder_due"
    assert sched.task.context["channel_id"] == "bitrix24"
    assert sched.task.context["recipient_id"] == "chat77"
    assert sched.task.context["reminder_count"] == 1


def test_logistics_compose_prioritizes_save_report_over_operator_lookup(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    service = LogisticsLLMService(RecordingLLMClient('{"status":"completed","answer":"wrong"}'))

    result = anyio_run(
        service.compose(
            manifest=manifest,
            task=AgentTask(task_id="save", request="отчет"),
            decision=LogisticsLLMDecision(status="completed", answer="", tool_calls=[]),
            tool_results=[
                ToolResult(
                    status="ok",
                    tool="vehicle_usage_get_operators",
                    data={"operators": [{"user_id": 13, "full_name": "Коверга Дмитрий"}]},
                ),
                ToolResult(
                    status="ok",
                    tool="vehicle_usage_save_report",
                    data={
                        "request_date": "2026-07-16",
                        "staff_entries_saved": 11,
                        "vehicles_saved": 6,
                        "vehicle_assignments_saved": 7,
                    },
                ),
            ],
        )
    )

    assert result.status == "completed"
    assert "Финальный отчет по машинам за 2026-07-16 сохранен." in result.answer
    assert "Операторы отчета" not in result.answer


def test_vehicle_usage_save_report_matches_minor_employee_name_typos(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=8, user_id=29, name="Никаненок Алексейи")])
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({13}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-16",
                "source_text": "Никоненок Авто 3",
                "parsed": {
                    "date": "2026-07-16",
                    "people": [{"full_name": "Никаненок Алексей", "status": "На авто"}],
                    "vehicles": [
                        {"vehicle_name": "Авто 1", "status": "idle", "drivers": []},
                        {"vehicle_name": "Авто 2", "status": "idle", "drivers": []},
                        {"vehicle_name": "Авто 3", "status": "in_use", "drivers": ["Никоненок Алексей"]},
                    ],
                },
            },
            user_id=13,
            dialog_id="13",
        )
    )

    assert result.status == "ok"
    assert not result.data.get("needs_clarification")
    assert result.data["vehicle_assignments_saved"] == 3


def test_vehicle_usage_save_report_counts_unique_vehicles_separately(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees(
        [
            StaffMember(order=1, name="Борисов Андрей", user_id=27),
            StaffMember(order=2, name="Карасев Алексей", user_id=25),
        ]
    )
    store._vehicles = [{"id": 2, "brand_model": "Авто 2", "registration_number": ""}]
    tool = VehicleSaveReportTool(store, allowed_user_ids=frozenset({13}))

    result = anyio_run(
        tool.execute(
            {
                "request_date": "2026-07-16",
                "source_text": "Борисов и Карасев Авто 2",
                "parsed": {
                    "date": "2026-07-16",
                    "people": [
                        {"full_name": "Борисов Андрей", "status": "worked"},
                        {"full_name": "Карасев Алексей", "status": "worked"},
                    ],
                    "vehicles": [
                        {
                            "vehicle_name": "Авто 2",
                            "status": "in_use",
                            "drivers": ["Борисов Андрей", "Карасев Алексей"],
                        }
                    ],
                },
            },
            user_id=13,
            dialog_id="13",
        )
    )

    assert result.status == "ok"
    assert result.data["vehicles_saved"] == 1
    assert result.data["vehicle_assignments_saved"] == 2


def test_reminder_increments_count(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()

    result = anyio_run(
        _specialist(manifest, store, vu_settings=_VU_SETTINGS).handle(
            AgentTask(
                task_id="reminder-2",
                request="Напоминание.",
                context={
                    "channel_id": "bitrix24",
                    "recipient_id": "chat77",
                    "event": "vehicle_usage_reminder_due",
                    "request_date": "2026-06-23",
                    "reminder_count": 1,
                },
            )
        )
    )

    if result.scheduled_tasks:
        sched = result.scheduled_tasks[0]
        assert sched.task is not None
        assert sched.task.context["reminder_count"] == 2


def test_max_reminders_stops_scheduling(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()

    result = anyio_run(
        _specialist(manifest, store, vu_settings=_VU_SETTINGS).handle(
            AgentTask(
                task_id="reminder-max",
                request="Напоминание.",
                context={
                    "event": "vehicle_usage_reminder_due",
                    "request_date": "2026-06-23",
                    "reminder_count": 3,  # already at max
                },
            )
        )
    )

    assert result.scheduled_tasks == []


def test_save_report_cancels_reminder(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, user_id=15, name="Иван Петров")])
    fake_llm = FakeLogisticsLLM(
        tool_call_steps=[
            [LogisticsLLMToolCall(name="vehicle_usage_context", args={"request_date": "2026-06-23"})],
            [
                LogisticsLLMToolCall(
                    name="vehicle_usage_save_report",
                    args={
                        "request_date": "2026-06-23",
                        "source_text": "подтверждаю",
                        "parsed": {
                            "date": "2026-06-23",
                            "people": [{"staff_order": 1, "full_name": "Иван Петров", "status": "shift"}],
                            "vehicles": [],
                        },
                    },
                )
            ],
            [LogisticsLLMToolCall(name="none")],
        ],
        final_answer="Отчёт сохранён.",
    )

    result = anyio_run(
        _specialist(manifest, store, llm=fake_llm, vu_settings=_VU_SETTINGS).handle(
            AgentTask(
                task_id="save-report-1",
                request="подтверждаю",
                user=UserContext(id="9"),
                context={"request_date": "2026-06-23"},
            )
        )
    )

    assert result.answer == "Отчёт сохранён."
    assert len(result.scheduled_tasks) == 1
    cancel = result.scheduled_tasks[0]
    assert cancel.cancel is True
    assert cancel.job_id == "vu_reminder_2026-06-23"
    assert cancel.agent_id == "logistics"


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)
