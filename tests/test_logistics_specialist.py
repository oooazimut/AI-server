from __future__ import annotations

import json

from ai_server.agents.logistics import LogisticsLLMService, LogisticsLLMToolCall, LogisticsSpecialist
from ai_server.agents.logistics.specialist import VehicleUsageSettings
from ai_server.agents.logistics.tools import VehicleContextTool, VehicleSaveDraftTool, VehicleSaveReportTool
from ai_server.models import AgentTask, UserContext
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.vehicle_usage import StaffMember
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
            VehicleSaveDraftTool(store),
            VehicleSaveReportTool(store),
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
    assert report["employee_statuses"][0][1] == "shift"
    assert report["vehicle_assignments"]
    assert report["vehicle_assignments"][0][0] == 1


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
    assert sched.job_id == "vu_reminder_2026-06-23"
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
