from __future__ import annotations

import json
import sqlite3

from ai_server.agents.logistics import LogisticsSpecialist
from ai_server.agents.logistics_llm import LogisticsLLMService, LogisticsLLMToolCall
from ai_server.models import AgentTask, UserContext
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.vehicle_usage import StaffMember, VehicleUsageStore, VehicleUsageToolset
from tests.fakes import FakeEmbeddingProvider, FakeLogisticsLLM, RecordingLLMClient

_FAKE_RETRIEVER = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())


def test_logistics_specialist_forwards_dialog_history_to_decide(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    llm = FakeLogisticsLLM()
    history = [
        {"role": "user", "content": "кто сегодня на смене"},
        {"role": "assistant", "content": "Уточните дату смены."},
    ]

    specialist = LogisticsSpecialist(
        manifest,
        retriever=_FAKE_RETRIEVER,
        tools=VehicleUsageToolset(store=VehicleUsageStore(tmp_path / "vehicle_usage.sqlite"), user_id=9),
        llm=llm,
    )
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


def test_logistics_llm_decide_payload_includes_dialog_history_and_raw_context():
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


def test_logistics_specialist_saves_llm_parsed_draft(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = VehicleUsageStore(tmp_path / "vehicle_usage.sqlite")
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
    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    store = VehicleUsageStore(tmp_path / "vehicle_usage.sqlite")
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
    specialist = LogisticsSpecialist(
        manifest,
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        tools=VehicleUsageToolset(store=store, user_id=9, dialog_id="chat9"),
        llm=fake_llm,
    )

    result = anyio_run(specialist.handle(AgentTask(task_id="log-2", request="подтверждаю", user=UserContext(id="9"))))

    assert result.answer == "Сохранил утренний отчет."
    with sqlite3.connect(store.path) as db:
        request = db.execute("SELECT status FROM vehicle_usage_requests").fetchone()
        status = db.execute("SELECT status FROM employee_daily_statuses").fetchone()
        assignment = db.execute("SELECT vehicle_id, employee_id FROM vehicle_daily_assignments").fetchone()
    assert request[0] == "answered"
    assert status[0] == "shift"
    assert assignment == (1, 1)


def test_logistics_morning_handler_delivers_and_records_request(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("VEHICLE_USAGE_ENABLED", "true")
    monkeypatch.setenv("VEHICLE_USAGE_DRY_RUN", "false")
    monkeypatch.setenv("VEHICLE_USAGE_MANAGER_USER_ID", "9")
    monkeypatch.setenv("VEHICLE_USAGE_DIALOG_ID", "chat9")
    monkeypatch.setenv("VEHICLE_USAGE_REMINDER_INTERVAL_MINUTES", "30")
    monkeypatch.setenv("VEHICLE_USAGE_MAX_REMINDERS", "3")
    from ai_server.settings import get_settings

    store = VehicleUsageStore(tmp_path / "vehicle_usage.sqlite")
    fake_llm = FakeLogisticsLLM(
        tool_call_steps=[
            [LogisticsLLMToolCall(name="vehicle_usage_context", args={"request_date": "2026-06-05"})],
            [LogisticsLLMToolCall(name="none")],
        ],
        final_answer="Нужен утренний отчет.",
    )
    delivered = []

    async def _deliver(dialog_id: str, message: str) -> None:
        delivered.append((dialog_id, message))

    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    # No real scheduler — schedule_job is a safe no-op when _scheduler is None
    specialist = LogisticsSpecialist(
        manifest,
        retriever=_FAKE_RETRIEVER,
        tools=VehicleUsageToolset(store=store, user_id=9, dialog_id="chat9"),
        llm=fake_llm,
        deliver_fn=_deliver,
        settings=get_settings(),
    )
    anyio_run(specialist._run_and_deliver(reminder_count=0))

    assert delivered == [("chat9", "Нужен утренний отчет.")]
    with sqlite3.connect(store.path) as db:
        row = db.execute("SELECT status, reminder_count, message FROM vehicle_usage_requests").fetchone()
    assert row == ("sent", 1, "Нужен утренний отчет.")


def test_logistics_escalates_after_max_reminders(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("VEHICLE_USAGE_ENABLED", "true")
    monkeypatch.setenv("VEHICLE_USAGE_DRY_RUN", "false")
    monkeypatch.setenv("VEHICLE_USAGE_MANAGER_USER_ID", "9")
    monkeypatch.setenv("VEHICLE_USAGE_DIALOG_ID", "chat9")
    monkeypatch.setenv("VEHICLE_USAGE_ADMIN_NOTIFY_USER_IDS", "1,9")
    monkeypatch.setenv("VEHICLE_USAGE_MAX_REMINDERS", "3")
    from datetime import datetime

    from ai_server.settings import get_settings
    from ai_server.utils import MOSCOW_TZ

    today = datetime.now(MOSCOW_TZ).date().isoformat()
    store = VehicleUsageStore(tmp_path / "vehicle_usage.sqlite")
    store.create_sent_request(
        request_date=today,
        user_id=9,
        dialog_id="chat9",
        message="Нужен отчет.",
        sent_at=f"{today}T08:00:00+03:00",
        reminder_count=3,
    )
    fake_llm = FakeLogisticsLLM(
        tool_call_steps=[
            [LogisticsLLMToolCall(name="vehicle_usage_context", args={"request_date": "2026-06-05"})],
            [LogisticsLLMToolCall(name="none")],
        ],
        final_answer="Отчет по машинам не получен.",
    )
    notified: list[int] = []

    async def _notify(user_id: int, message: str) -> None:
        notified.append(user_id)

    manifest = get_agent_manifest("logistics")
    assert manifest is not None
    specialist = LogisticsSpecialist(
        manifest,
        retriever=_FAKE_RETRIEVER,
        tools=VehicleUsageToolset(store=store, user_id=9, dialog_id="chat9"),
        llm=fake_llm,
        notify_fn=_notify,
        settings=get_settings(),
    )
    anyio_run(specialist._run_and_deliver(reminder_count=3))

    assert notified == [1, 9]
    with sqlite3.connect(store.path) as db:
        row = db.execute("SELECT escalated_at FROM vehicle_usage_requests").fetchone()
    assert row[0]


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)


class FakeVehicleBitrix:
    def __init__(self) -> None:
        self.messages = []
        self.notifications = []

    async def send_bot_message(self, dialog_id, message, *, bot_id=None, keyboard=None):
        self.messages.append({"dialog_id": dialog_id, "message": message, "bot_id": bot_id, "keyboard": keyboard})
        return 1

    async def notify_user(self, *, user_id, message, tag="ai_server", sub_tag=""):
        self.notifications.append({"user_id": user_id, "message": message, "tag": tag, "sub_tag": sub_tag})
        return 1
