from datetime import datetime, timedelta, timezone
import sqlite3

from fastapi.testclient import TestClient

from ai_server.channels.bitrix import BitrixWebhookProcessor
from ai_server.integrations.bitrix.dialog_state import (
    BitrixPendingActionService,
    DialogStateStore,
    PendingBitrixAction,
    make_dialog_key,
)
from ai_server.integrations.bitrix.events import parse_incoming_message
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.main import app
from ai_server.models import ActionRecord, AgentResult
from ai_server.workers.bitrix.webhook_event_queue import WebhookEventQueue


def _bitrix_v2_message_payload() -> dict:
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "auth": {"application_token": "secret-token"},
        "data": {
            "bot": {"id": 42},
            "chat": {"id": 77, "dialogId": "chat99"},
            "message": {"id": 123, "authorId": 9, "text": "Покажи задачи в Битриксе"},
            "user": {"id": 9},
        },
    }


def test_parse_bitrix_v2_message():
    incoming = parse_incoming_message(_bitrix_v2_message_payload())

    assert incoming.event_type == "ONIMBOTV2MESSAGEADD"
    assert incoming.bot_id == 42
    assert incoming.dialog_id == "chat99"
    assert incoming.message_id == 123
    assert incoming.user_id == 9
    assert incoming.text == "Покажи задачи в Битриксе"


def test_webhook_event_queue_is_compatible_and_sanitizes_payload(tmp_path):
    queue = WebhookEventQueue(tmp_path / "webhook_event_queue.sqlite")
    queue.ensure_schema()

    event_id, inserted = queue.enqueue(_bitrix_v2_message_payload(), event_type="ONIMBOTV2MESSAGEADD")
    duplicate_id, duplicate_inserted = queue.enqueue(_bitrix_v2_message_payload(), event_type="ONIMBOTV2MESSAGEADD")

    assert inserted is True
    assert duplicate_inserted is False
    assert duplicate_id == event_id
    assert queue.stats()["pending"] == 1

    event = queue.claim_next()
    assert event is not None
    assert event["partition_key"] == "dialog:chat99"
    assert "auth" not in event["payload"]

    queue.mark_done(event_id, {"handled": True})
    assert queue.stats()["done"] == 1


def test_bitrix_events_endpoint_enqueues_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("AGENT_DRY_RUN", "true")

    with TestClient(app) as client:
        response = client.post(
            "/bitrix/events?secret=test-secret",
            json=_bitrix_v2_message_payload(),
        )
        duplicate = client.post(
            "/bitrix/events?secret=test-secret",
            json=_bitrix_v2_message_payload(),
        )
        status = client.get("/bitrix/webhook-events/status")

    assert response.status_code == 200
    assert response.json()["queued"] is True
    assert duplicate.status_code == 200
    assert duplicate.json()["duplicate"] is True
    assert status.json()["queue"]["pending"] == 1


def test_bitrix_events_endpoint_rejects_wrong_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")

    with TestClient(app) as client:
        response = client.post("/bitrix/events?secret=wrong", json=_bitrix_v2_message_payload())

    assert response.status_code == 403


def test_bitrix_webhook_processor_delegates_message_to_orchestrator(monkeypatch):
    monkeypatch.setenv("AGENT_DRY_RUN", "true")

    class FakeOrchestrator:
        async def handle(self, task):
            assert task.source == "bitrix24_chat"
            assert task.user.id == "9"
            assert task.user.raw["dialog_id"] == "chat99"
            assert "Битриксе" in task.request
            return AgentResult(
                status="completed",
                agent_id="internal_orchestrator",
                answer="Готово",
                confidence=0.9,
                handoff_to=["bitrix24"],
            )

    processor = BitrixWebhookProcessor(orchestrator=FakeOrchestrator())

    result = anyio_run(processor.process(_bitrix_v2_message_payload()))

    assert result["handled"] is True
    assert result["reply_sent"] is False
    assert result["handoff_to"] == ["bitrix24"]


def test_dialog_state_store_keeps_legacy_pending_action_shape(tmp_path):
    store = DialogStateStore(tmp_path / "dialog_state.sqlite")
    key = "chat:77:user:9"
    store.save_raw(
        key,
        {
            "summary": "old summary",
            "turns": [{"role": "user", "content": "создай задачу"}],
            "pending_action": {
                "method": "tasks.task.add",
                "params": {"fields": {"TITLE": "Тест"}},
                "summary": "создать задачу",
                "created_by": "9",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            "pending_shell_action": {"command": "ignored"},
        },
    )

    state = store.load(key)
    assert state.summary == "old summary"
    assert state.pending_action is not None
    assert state.pending_action.method == "tasks.task.add"
    assert state.pending_action.created_by == 9

    store.clear_pending(key)
    assert store.load(key).pending_action is None


def test_bitrix_webhook_processor_saves_pending_action_from_specialist(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DRY_RUN", "true")
    store = DialogStateStore(tmp_path / "dialog_state.sqlite")
    fake_bitrix = FakeBitrixClient()

    class FakeOrchestrator:
        async def handle(self, task):
            return AgentResult(
                status="needs_human",
                agent_id="internal_orchestrator",
                answer="Нужно подтверждение.",
                actions_requiring_approval=[
                    ActionRecord(
                        name="bitrix_api",
                        status="approval_required",
                        details={
                            "method": "tasks.task.add",
                            "params": {"fields": {"TITLE": "Тестовая задача"}},
                            "summary": "создать задачу",
                        },
                    )
                ],
                confidence=0.8,
            )

    processor = BitrixWebhookProcessor(
        bitrix=fake_bitrix,
        orchestrator=FakeOrchestrator(),
        pending_actions=BitrixPendingActionService(
            store=store,
            bitrix=fake_bitrix,
            audit_log_path=tmp_path / "bitrix_write_audit.jsonl",
        ),
    )

    result = anyio_run(processor.process(_bitrix_v2_message_payload()))
    key = make_dialog_key(chat_id=77, dialog_id="chat99", user_id=9)
    state = store.load(key)

    assert result["pending_action_saved"] is True
    assert result["dialog_key"] == key
    assert state.pending_action is not None
    assert state.pending_action.method == "tasks.task.add"
    assert state.pending_action.params["fields"]["TITLE"] == "Тестовая задача"


def test_bitrix_pending_action_confirm_executes_and_clears_state(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DRY_RUN", "false")
    monkeypatch.setenv("AGENT_WRITE_ALLOWED_USER_IDS", "9")
    monkeypatch.setenv("BITRIX_OAUTH_REQUIRED_FOR_WRITES", "false")
    store = DialogStateStore(tmp_path / "dialog_state.sqlite")
    fake_bitrix = FakeBitrixClient()
    key = make_dialog_key(chat_id=77, dialog_id="chat99", user_id=9)
    store.set_pending(
        key,
        PendingBitrixAction(
            method="tasks.task.add",
            params={"fields": {"TITLE": "Тестовая задача"}},
            summary="создать задачу",
            created_by=9,
        ),
    )
    payload = _bitrix_v2_message_payload()
    payload["data"]["message"]["text"] = "да"

    class ForbiddenOrchestrator:
        async def handle(self, task):
            raise AssertionError("pending confirmation should not reach orchestrator")

    processor = BitrixWebhookProcessor(
        bitrix=fake_bitrix,
        orchestrator=ForbiddenOrchestrator(),
        pending_actions=BitrixPendingActionService(
            store=store,
            bitrix=fake_bitrix,
            audit_log_path=tmp_path / "bitrix_write_audit.jsonl",
        ),
    )

    result = anyio_run(processor.process(payload))

    assert result["agent_result_status"] == "executed"
    assert fake_bitrix.calls == [("tasks.task.add", {"fields": {"TITLE": "Тестовая задача"}})]
    assert fake_bitrix.messages[0][1].startswith("Готово")
    assert store.load(key).pending_action is None
    assert '"status": "executed"' in (tmp_path / "bitrix_write_audit.jsonl").read_text(encoding="utf-8")


def test_bitrix_pending_action_cancel_clears_state_without_call(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DRY_RUN", "false")
    store = DialogStateStore(tmp_path / "dialog_state.sqlite")
    fake_bitrix = FakeBitrixClient()
    key = make_dialog_key(chat_id=77, dialog_id="chat99", user_id=9)
    store.set_pending(
        key,
        PendingBitrixAction(
            method="tasks.task.add",
            params={"fields": {"TITLE": "Тестовая задача"}},
            summary="создать задачу",
            created_by=9,
        ),
    )
    payload = _bitrix_v2_message_payload()
    payload["data"]["message"]["text"] = "отмена"

    processor = BitrixWebhookProcessor(
        bitrix=fake_bitrix,
        pending_actions=BitrixPendingActionService(
            store=store,
            bitrix=fake_bitrix,
            audit_log_path=tmp_path / "bitrix_write_audit.jsonl",
        ),
    )

    result = anyio_run(processor.process(payload))

    assert result["agent_result_status"] == "cancelled"
    assert fake_bitrix.calls == []
    assert store.load(key).pending_action is None
    assert '"status": "cancelled"' in (tmp_path / "bitrix_write_audit.jsonl").read_text(encoding="utf-8")


def test_bitrix_oauth_service_reads_migrated_sqlite(tmp_path):
    db_path = tmp_path / "bitrix_oauth.sqlite"
    service = BitrixOAuthService(db_path)
    service.ensure_schema()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=2)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO bitrix_oauth_tokens (
                user_id, domain, member_id, client_endpoint, server_endpoint,
                access_token, refresh_token, scope, expires_at, updated_at, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                9,
                "example.bitrix24.ru",
                "member",
                "https://example.bitrix24.ru/rest/",
                "https://example.bitrix24.ru",
                "access",
                "refresh",
                "tasks,user",
                expires_at.isoformat(),
                datetime.now(timezone.utc).isoformat(),
                "test",
            ),
        )

    status = service.public_status()
    token = service.get_token(9)

    assert status["linked_users_count"] == 1
    assert token is not None
    assert token.user_id == 9
    assert token.access_token == "access"


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)


class FakeBitrixClient:
    def __init__(self) -> None:
        self.calls = []
        self.messages = []

    async def call(self, method, payload=None, *, base_url=None):
        self.calls.append((method, payload or {}))
        return {"result": {"id": 123}}

    async def send_bot_message(self, dialog_id, message, *, bot_id=None, keyboard=None):
        self.messages.append((dialog_id, message, bot_id, keyboard))
        return 1
