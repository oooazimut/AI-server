import sqlite3
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.bitrix_llm import BitrixLLMToolCall
from ai_server.attachments import StoredAttachment
from ai_server.channels.bitrix import BitrixWebhookProcessor
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.dialog_state import (
    BitrixPendingActionService,
    DialogStateStore,
    PendingBitrixAction,
    apply_write_policy,
    make_dialog_key,
)
from ai_server.integrations.bitrix.events import parse_incoming_message
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, _token_endpoint_from_server
from ai_server.main import app
from ai_server.models import ActionRecord, AgentResult, ModelUsageRecord, ToolResult
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import load_agent_manifests
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.specialists import manifest_by_id
from ai_server.technical_footer import ProviderBalanceSnapshot, TechnicalFooterService
from ai_server.tools.bitrix import BitrixToolset
from ai_server.transcription import TranscriptionResult
from ai_server.workers.bitrix.webhook_event_queue import WebhookEventQueue
from scripts.create_bitrix_dev_chat import chat_reference, sanitize_result
from tests.fakes import (
    FakeBitrixLLM,
    FakeEmbeddingProvider,
    FakeInternalOrchestratorLLM,
)


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


def test_bitrix_webhook_processor_transcribes_voice_before_orchestrator(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DRY_RUN", "true")
    audio = StoredAttachment(
        file_id=501,
        name="voice.ogg",
        content_type="audio/ogg",
        size=4,
        path=str(tmp_path / "voice.ogg"),
        is_audio=True,
    )

    class FakeAttachmentService:
        async def download_message_files(self, message):
            assert message.files[0].id == 501
            return [audio]

    class FakeTranscriber:
        async def transcribe(self, attachment):
            assert attachment.file_id == 501
            return TranscriptionResult(
                text="Создай задачу по камере",
                model="fake_stt",
                attachment=attachment,
                raw={"ok": True},
            )

    class FakeOrchestrator:
        async def handle(self, task):
            assert task.request == "Создай задачу по камере"
            assert task.files[0]["id"] == 501
            assert task.files[1]["file_id"] == 501
            assert task.context["transcriptions"][0]["text"] == "Создай задачу по камере"
            return AgentResult(
                status="completed",
                agent_id="internal_orchestrator",
                answer="Готово",
                confidence=0.9,
                handoff_to=["bitrix24"],
            )

    payload = _bitrix_v2_message_payload()
    payload["data"]["message"] = {
        "id": 123,
        "authorId": 9,
        "text": "",
        "files": [{"id": 501, "name": "voice.ogg", "type": "voice"}],
    }
    processor = BitrixWebhookProcessor(
        orchestrator=FakeOrchestrator(),
        attachment_service=FakeAttachmentService(),
        transcriber=FakeTranscriber(),
    )

    result = anyio_run(processor.process(payload))

    assert result["handled"] is True
    assert result["transcriptions"][0]["text"] == "Создай задачу по камере"


def test_bitrix_webhook_processor_appends_admin_technical_footer(monkeypatch):
    monkeypatch.setenv("AGENT_DRY_RUN", "false")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ENABLED", "true")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ALLOWED_USER_IDS", "9")
    fake_bitrix = FakeBitrixClient()

    class FakeOrchestrator:
        async def handle(self, task):
            return AgentResult(
                status="completed",
                agent_id="internal_orchestrator",
                answer="Готово",
                model_usage=[
                    ModelUsageRecord(
                        agent_id="internal_orchestrator",
                        provider="deepseek",
                        model="deepseek-v4-flash",
                    )
                ],
                confidence=0.9,
            )

    processor = BitrixWebhookProcessor(
        bitrix=fake_bitrix,
        orchestrator=FakeOrchestrator(),
        technical_footer=TechnicalFooterService(balance_registry={"deepseek": FakeDeepSeekBalance()}),
    )

    result = anyio_run(processor.process(_bitrix_v2_message_payload()))

    assert result["reply_sent"] is True
    assert "Тех: LLM: internal_orchestrator deepseek deepseek-v4-flash" in fake_bitrix.messages[0][1]
    assert "DeepSeek: доступен; баланс $12.34." in fake_bitrix.messages[0][1]


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
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AGENT_DRY_RUN", "true")
    monkeypatch.setenv("BITRIX_OAUTH_REQUIRED_FOR_WRITES", "false")
    store = DialogStateStore(tmp_path / "dialog_state.sqlite")
    fake_bitrix = FakeBitrixClient()
    key = make_dialog_key(chat_id=77, dialog_id="chat99", user_id=9)
    pending_service = BitrixPendingActionService(
        store=store,
        bitrix=fake_bitrix,
        audit_log_path=tmp_path / "bitrix_write_audit.jsonl",
        dry_run=False,
    )
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
    payload["data"]["message"]["text"] = "ну давай, запускай это"
    manifests = load_agent_manifests()
    specialist = Bitrix24Specialist(
        manifest_by_id(manifests, "bitrix24"),
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        tools=BitrixToolset(client=fake_bitrix, pending_actions=pending_service, dialog_key=key, user_id=9),
        llm=FakeBitrixLLM(
            tool_calls=[BitrixLLMToolCall(name="bitrix_api", args={"action": "confirm_pending"})],
            final_answer="Подтвердил, задача создана.",
        ),
    )
    processor = BitrixWebhookProcessor(
        bitrix=fake_bitrix,
        orchestrator=InternalOrchestrator(
            manifests,
            specialists={"bitrix24": specialist},
            orchestrator_llm=FakeInternalOrchestratorLLM(),
        ),
        pending_actions=pending_service,
    )

    result = anyio_run(processor.process(payload))

    assert result["agent_result_status"] == "completed"
    assert fake_bitrix.calls == [("tasks.task.add", {"fields": {"TITLE": "Тестовая задача"}})]
    assert store.load(key).pending_action is None


def test_bitrix_pending_action_cancel_clears_state_without_call(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AGENT_DRY_RUN", "true")
    store = DialogStateStore(tmp_path / "dialog_state.sqlite")
    fake_bitrix = FakeBitrixClient()
    key = make_dialog_key(chat_id=77, dialog_id="chat99", user_id=9)
    pending_service = BitrixPendingActionService(
        store=store,
        bitrix=fake_bitrix,
        audit_log_path=tmp_path / "bitrix_write_audit.jsonl",
    )
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
    payload["data"]["message"]["text"] = "не надо выполнять"
    manifests = load_agent_manifests()
    specialist = Bitrix24Specialist(
        manifest_by_id(manifests, "bitrix24"),
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        tools=BitrixToolset(client=fake_bitrix, pending_actions=pending_service, dialog_key=key, user_id=9),
        llm=FakeBitrixLLM(
            tool_calls=[BitrixLLMToolCall(name="bitrix_api", args={"action": "cancel_pending"})],
            final_answer="Отменил, создавать не буду.",
        ),
    )
    processor = BitrixWebhookProcessor(
        bitrix=fake_bitrix,
        orchestrator=InternalOrchestrator(
            manifests,
            specialists={"bitrix24": specialist},
            orchestrator_llm=FakeInternalOrchestratorLLM(),
        ),
        pending_actions=pending_service,
    )

    result = anyio_run(processor.process(payload))

    assert result["agent_result_status"] == "completed"
    assert fake_bitrix.calls == []
    assert store.load(key).pending_action is None


def test_bitrix_pending_action_new_request_goes_to_orchestrator(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DRY_RUN", "true")
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
    payload["data"]["message"]["text"] = "Покажи мои открытые задачи"

    class FakeOrchestrator:
        def __init__(self) -> None:
            self.requests = []

        async def handle(self, task):
            self.requests.append(task.request)
            return AgentResult(
                status="completed",
                agent_id="internal_orchestrator",
                answer="Вот список задач.",
                confidence=0.8,
            )

    orchestrator = FakeOrchestrator()
    processor = BitrixWebhookProcessor(
        bitrix=fake_bitrix,
        orchestrator=orchestrator,
        pending_actions=BitrixPendingActionService(
            store=store,
            bitrix=fake_bitrix,
            audit_log_path=tmp_path / "bitrix_write_audit.jsonl",
        ),
    )

    result = anyio_run(processor.process(payload))

    assert result["agent_result_status"] == "completed"
    assert orchestrator.requests == ["Покажи мои открытые задачи"]
    assert store.load(key).pending_action is not None
    assert fake_bitrix.calls == []


def test_quality_control_live_webhook_requires_actor(monkeypatch):
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "true")
    monkeypatch.setenv("QUALITY_CONTROL_DRY_RUN", "false")
    monkeypatch.delenv("QUALITY_CONTROL_ACTOR_USER_ID", raising=False)
    fake_bitrix = FakeBitrixClient()
    payload = {
        "event": "ONTASKUPDATE",
        "data": {"FIELDS_AFTER": {"ID": 8413}},
    }

    result = anyio_run(BitrixWebhookProcessor(bitrix=fake_bitrix).process(payload))

    assert result["handled"] is False
    assert result["quality_control"]["reason"] == "quality_actor_not_configured"
    assert fake_bitrix.calls == []


def test_task_add_write_policy_translates_internal_no_deadline_marker():
    params = {
        "fields": {
            "TITLE": "Тестовая задача",
            "RESPONSIBLE_ID": 9,
            "NO_DEADLINE": True,
            "DEADLINE": "",
        }
    }

    result = apply_write_policy("tasks.task.add", params)

    assert result["fields"] == {"TITLE": "Тестовая задача", "RESPONSIBLE_ID": 9, "DEADLINE": ""}


def test_bitrix_client_create_bot_chat_builds_v2_payload(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_BOT_ID", "42")
    monkeypatch.setenv("BITRIX_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("BITRIX_BOT_AUTH_MODE", "webhook")

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
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AGENT_DRY_RUN", "true")
    monkeypatch.setenv("BITRIX_OAUTH_REQUIRED_FOR_WRITES", "false")
    store = DialogStateStore(tmp_path / "dialog_state.sqlite")
    fake_bitrix = FakeBitrixClient()
    key = make_dialog_key(chat_id=77, dialog_id="chat99", user_id=9)
    pending_service = BitrixPendingActionService(
        store=store,
        bitrix=fake_bitrix,
        audit_log_path=tmp_path / "bitrix_write_audit.jsonl",
        dry_run=False,
    )
    manifests = load_agent_manifests()
    bitrix_tools = BitrixToolset(client=fake_bitrix, pending_actions=pending_service, dialog_key=key, user_id=9)
    specialist = Bitrix24Specialist(
        manifest_by_id(manifests, "bitrix24"),
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        tools=bitrix_tools,
        llm=FakeBitrixLLM(
            tool_call_steps=[
                [
                    BitrixLLMToolCall(
                        name="task_create_draft",
                        args={
                            "title": "проверить IP-камеру",
                            "responsible_id": 9,
                            "deadline_iso": "2026-06-15T19:00:00+03:00",
                        },
                    )
                ],
                [BitrixLLMToolCall(name="bitrix_api", args={"action": "confirm_pending"})],
            ],
            final_answer="Готово.",
        ),
    )
    processor = BitrixWebhookProcessor(
        bitrix=fake_bitrix,
        orchestrator=InternalOrchestrator(
            manifests,
            specialists={"bitrix24": specialist},
            orchestrator_llm=FakeInternalOrchestratorLLM(handoff_to=["bitrix24"]),
        ),
        pending_actions=pending_service,
    )
    payload = _bitrix_v2_message_payload()
    payload["data"]["message"]["text"] = "Создай задачу на себя проверить IP-камеру 15 июня"

    draft_result = anyio_run(processor.process(payload))
    saved_pending = store.load(key).pending_action

    assert draft_result["handled"] is True
    assert draft_result["agent_result_status"] == "needs_human"
    assert draft_result["pending_action_saved"] is True
    assert saved_pending is not None
    assert saved_pending.method == "tasks.task.add"
    fields = saved_pending.params["fields"]
    assert fields["TITLE"] == "проверить IP-камеру"
    assert fields["RESPONSIBLE_ID"] == 9
    assert fields["CREATED_BY"] == 9
    assert "DEADLINE" in fields

    payload["data"]["message"]["text"] = "подтверждаю создание"
    confirm_result = anyio_run(processor.process(payload))

    assert confirm_result["agent_result_status"] == "completed"
    assert fake_bitrix.calls == [("tasks.task.add", {"fields": fields})]
    assert store.load(key).pending_action is None


def test_bitrix_oauth_service_reads_migrated_sqlite(tmp_path):
    db_path = tmp_path / "bitrix_oauth.sqlite"
    service = BitrixOAuthService(db_path=db_path)
    service.ensure_schema()
    expires_at = datetime.now(UTC) + timedelta(hours=2)
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
                datetime.now(UTC).isoformat(),
                "test",
            ),
        )

    status = service.public_status()
    token = service.get_token(9)

    assert status["linked_users_count"] == 1
    assert token is not None
    assert token.user_id == 9
    assert token.access_token == "access"


def test_bitrix_oauth_service_saves_token_from_app_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "example.bitrix24.ru")
    db_path = tmp_path / "bitrix_oauth.sqlite"
    service = BitrixOAuthService(db_path=db_path)

    result = anyio_run(
        service.save_from_payload(
            {
                "auth": {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "domain": "example.bitrix24.ru",
                    "member_id": "member",
                    "scope": "tasks,user",
                    "expires_in": 3600,
                    "user_id": 9,
                }
            },
            source="bitrix_app",
        )
    )
    token = service.get_token(9)

    assert result.user_id == 9
    assert token is not None
    assert token.access_token == "access"
    assert service.public_status()["linked_users_count"] == 1
    assert service.public_status()["authorization"]["message"]


def test_bitrix_oauth_token_endpoint_handles_rest_server_endpoint(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_OAUTH_TOKEN_ENDPOINT", "")

    assert _token_endpoint_from_server("https://oauth.bitrix.info/rest/") == "https://oauth.bitrix.info/oauth/token/"


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)


class FakeBitrixClient:
    def __init__(self) -> None:
        self.calls = []
        self.messages = []
        self.task = {
            "id": "8413",
            "title": "Проверить IP-камеру",
            "description": "Перезагрузить камеру и проверить изображение.",
            "status": "3",
            "responsibleId": "9",
            "createdBy": "1",
            "groupId": "44",
            "taskControl": "Y",
        }

    async def call(self, method, payload=None, *, base_url=None):
        self.calls.append((method, payload or {}))
        return {"result": {"id": 123}}

    async def send_bot_message(self, dialog_id, message, *, bot_id=None, keyboard=None):
        self.messages.append((dialog_id, message, bot_id, keyboard))
        return 1

    async def get_task(self, task_id, *, select=None):
        task = dict(self.task)
        task["id"] = str(task_id)
        return {"task": task}

    async def add_task_result(self, task_id, text):
        payload = {"taskId": task_id, "fields": {"text": text}}
        self.calls.append(("tasks.task.result.add", payload))
        return {"id": 501, "taskId": task_id}

    async def complete_task(self, task_id):
        payload = {"taskId": task_id}
        self.calls.append(("tasks.task.complete", payload))
        self.task["status"] = "5"
        return True

    async def approve_task(self, task_id):
        self.calls.append(("tasks.task.approve", {"taskId": task_id}))
        self.task["status"] = "5"
        return True

    async def disapprove_task(self, task_id):
        self.calls.append(("tasks.task.disapprove", {"taskId": task_id}))
        return True

    async def renew_task(self, task_id):
        self.calls.append(("tasks.task.renew", {"taskId": task_id}))
        self.task["status"] = "3"
        return True

    async def add_task_comment(self, *, task_id, message, author_id=None):
        payload = {"TASKID": task_id, "FIELDS": {"POST_MESSAGE": message}}
        if author_id is not None:
            payload["FIELDS"]["AUTHOR_ID"] = author_id
        self.calls.append(("task.commentitem.add", payload))
        return 1

    async def get_user(self, user_id: int):
        return {
            "ID": str(user_id),
            "WORK_POSITION": "Руководитель",
            "IS_ADMIN": "N",
            "ACTIVE": "Y",
            "NAME": "Test",
            "LAST_NAME": "User",
            "USER_TYPE": "employee",
            "UF_DEPARTMENT": [],
        }

    async def notify_user(self, *, user_id, message, tag="ai_server", sub_tag=""):
        self.calls.append(
            (
                "im.notify.system.add",
                {"USER_ID": user_id, "MESSAGE": message, "TAG": f"{tag}:{sub_tag}" if sub_tag else tag},
            )
        )
        return 1


class RecordingCreateChatClient(BitrixClient):
    def __init__(self) -> None:
        super().__init__(base_url="https://example.bitrix24.ru/rest/1/webhook/")
        self.calls = []

    async def result(self, method, payload=None, *, base_url=None):
        self.calls.append((method, payload or {}))
        return {"chatId": 555, "dialogId": "chat555"}


class FakeBitrixTools:
    def definitions(self):
        return []

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


class FakeDeepSeekBalance:
    async def snapshot(self):
        return ProviderBalanceSnapshot(
            provider="deepseek",
            status="ok",
            lines=["DeepSeek: доступен; баланс $12.34."],
            available=True,
        )
