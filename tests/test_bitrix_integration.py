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
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.events import parse_incoming_message
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.main import app
from ai_server.models import ActionRecord, AgentResult, ToolResult
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.workers.bitrix.webhook_event_queue import WebhookEventQueue
from scripts.create_bitrix_dev_chat import chat_reference, sanitize_result
from tests.fakes import FakeEmbeddingProvider


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


def test_bitrix_client_create_bot_chat_builds_v2_payload(monkeypatch):
    monkeypatch.setenv("BITRIX_BOT_ID", "42")
    monkeypatch.setenv("BITRIX_BOT_TOKEN", "bot-token")

    client = RecordingCreateChatClient()

    result = anyio_run(
        client.create_bot_chat(
            title=" AI dev ",
            user_ids=[1, 9, 9],
            description="Dev contour",
            message="Ready",
        )
    )

    assert result == {"chatId": 555, "dialogId": "chat555"}
    assert client.calls == [
        (
            "imbot.v2.Chat.add",
            {
                "botId": 42,
                "botToken": "bot-token",
                "fields": {
                    "title": "AI dev",
                    "color": "mint",
                    "userIds": [1, 9],
                    "description": "Dev contour",
                    "message": "Ready",
                },
            },
        )
    ]


def test_create_bitrix_dev_chat_helpers_extract_reference_and_redact_tokens():
    raw = {
        "chat": {"id": 3955, "dialogId": "chat3955"},
        "callInfo": {"token": "secret-call-token", "chatId": 3955},
        "access_token": "secret-access",
    }

    assert chat_reference(raw) == {"chat_id": 3955, "dialog_id": "chat3955"}
    assert sanitize_result(raw)["callInfo"]["token"] == "<redacted>"
    assert sanitize_result(raw)["access_token"] == "<redacted>"


def test_bitrix_task_create_chat_flow_saves_and_confirms_pending_action(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("AGENT_DRY_RUN", "true")
    store = DialogStateStore(tmp_path / "dialog_state.sqlite")
    fake_bitrix = FakeBitrixClient()
    processor = BitrixWebhookProcessor(
        bitrix=fake_bitrix,
        bitrix_tools=FakeBitrixTools(),
        bitrix_retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        pending_actions=BitrixPendingActionService(
            store=store,
            bitrix=fake_bitrix,
            audit_log_path=tmp_path / "bitrix_write_audit.jsonl",
        ),
    )
    payload = _bitrix_v2_message_payload()
    payload["data"]["message"]["text"] = "Создай задачу на Иванова проверить IP-камеру завтра"

    draft_result = anyio_run(processor.process(payload))
    key = make_dialog_key(chat_id=77, dialog_id="chat99", user_id=9)
    pending = store.load(key).pending_action

    assert draft_result["handled"] is True
    assert draft_result["agent_result_status"] == "needs_human"
    assert draft_result["pending_action_saved"] is True
    assert pending is not None
    assert pending.method == "tasks.task.add"
    fields = pending.params["fields"]
    assert fields["TITLE"] == "проверить IP-камеру"
    assert fields["RESPONSIBLE_ID"] == 15
    assert fields["CREATED_BY"] == 9
    assert "DEADLINE" in fields

    monkeypatch.setenv("AGENT_DRY_RUN", "false")
    monkeypatch.setenv("AGENT_WRITE_ALLOWED_USER_IDS", "9")
    monkeypatch.setenv("BITRIX_OAUTH_REQUIRED_FOR_WRITES", "false")
    payload["data"]["message"]["text"] = "да"

    confirm_result = anyio_run(processor.process(payload))

    assert confirm_result["agent_result_status"] == "executed"
    assert fake_bitrix.calls == [("tasks.task.add", {"fields": fields})]
    assert fake_bitrix.messages[0][0] == "chat99"
    assert fake_bitrix.messages[0][1].startswith("Готово")
    assert store.load(key).pending_action is None


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


class RecordingCreateChatClient(BitrixClient):
    def __init__(self) -> None:
        super().__init__(base_url="https://example.bitrix24.ru/rest/1/webhook/")
        self.calls = []

    async def result(self, method, payload=None, *, base_url=None):
        self.calls.append((method, payload or {}))
        return {"chatId": 555, "dialogId": "chat555"}


class FakeBitrixTools:
    async def resolve_user(self, query: str, *, limit: int = 5):
        assert query == "Иванова"
        return ToolResult(
            status="ok",
            tool="resolve_user",
            data={
                "query": query,
                "candidate": {"id": 15, "label": "Иванов Иван"},
                "candidates": [{"id": 15, "label": "Иванов Иван"}],
            },
        )

    async def resolve_project(self, query: str, *, limit: int = 5):
        raise AssertionError("project resolver should not be called")

    def portal_search_contract(self, args):
        raise AssertionError("portal search should not be called")
