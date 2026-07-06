from __future__ import annotations

import json

from ai_server.agents.logistics import (
    LogisticsLLMDecision,
    LogisticsLLMService,
    LogisticsLLMToolCall,
    LogisticsSpecialist,
)
from ai_server.agents.logistics.specialist import VehicleUsageSettings
from ai_server.agents.logistics.tools import (
    VehicleCancelReportTool,
    VehicleContextTool,
    VehicleGetOperatorsTool,
    VehicleGetReportTool,
    VehicleSaveDraftTool,
    VehicleSaveReportTool,
    VehicleSetOperatorsTool,
)
from ai_server.models import ActionRecord, AgentResult, AgentTask, ToolResult, ToolStatus, UserContext
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.vehicle_usage import StaffMember, resolve_vehicle_usage_operator_ids
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
                            {"full_name": "Абдураимова Галина", "status": "on_leave"},
                            {"full_name": "Марат", "status": "in_office"},
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
    assert "Абдураимова Галина — отпуск" in result.answer
    assert "Марат — работал" in result.answer
    assert "Авто 2 — Борисов Андрей / в работе" in result.answer
    assert "Авто 5 — простой" in result.answer
    assert "Авто 5 — простой / простой" not in result.answer
    assert "in_use" not in result.answer
    assert "idle" not in result.answer
    assert "on_leave" not in result.answer
    assert "in_office" not in result.answer


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
                    },
                )
            ],
        )
    )

    assert not client.calls
    assert "Финальный отчет по машинам за 2026-07-08 сохранен" in result.answer
    assert "сотрудники: 11" in result.answer
    assert "машины: 8" in result.answer


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


def test_vehicle_usage_save_report_marks_vehicle_drivers_worked(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    store = FakeVehicleUsageStore()
    store.upsert_employees([StaffMember(order=1, name="Борисов Андрей", user_id=27)])
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
