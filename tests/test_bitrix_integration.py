import sqlite3
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.bitrix24.tools import BitrixApiTool
from ai_server.attachments import StoredAttachment
from ai_server.integrations.bitrix.bitrix_policy import apply_write_policy
from ai_server.integrations.bitrix.chat_parser import build_agent_task_from_bitrix_chat
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.events import parse_incoming_message
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, _token_endpoint_from_server
from ai_server.main import app
from ai_server.models import AgentTask, ToolResult, ToolStatus
from ai_server.registry import get_agent_manifest
from ai_server.settings import get_settings
from ai_server.transcription import TranscriptionResult
from ai_server.workers.bitrix.webhook_event_queue import WebhookEventQueue
from scripts.create_bitrix_dev_chat import chat_reference, sanitize_result
from tests.fakes import FakeBitrixLLM


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


def test_webhook_event_queue_is_compatible_and_sanitizes_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    queue = WebhookEventQueue(tmp_path / "webhook_event_queue.sqlite", settings=get_settings())
    queue.ensure_schema()

    event_id, inserted = anyio_run(queue.enqueue(_bitrix_v2_message_payload(), event_type="ONIMBOTV2MESSAGEADD"))
    duplicate_id, duplicate_inserted = anyio_run(
        queue.enqueue(_bitrix_v2_message_payload(), event_type="ONIMBOTV2MESSAGEADD")
    )

    assert inserted is True
    assert duplicate_inserted is False
    assert duplicate_id == event_id
    assert anyio_run(queue.stats())["pending"] == 1

    event = anyio_run(queue.claim_next())
    assert event is not None
    assert event["partition_key"] == "dialog:chat99"
    assert "auth" not in event["payload"]

    anyio_run(queue.mark_done(event_id, {"handled": True}))
    assert anyio_run(queue.stats())["done"] == 1


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


def test_build_agent_task_from_bitrix_chat_builds_correct_task():
    class FakeAttachmentService:
        async def download_message_files(self, message):
            return []

    class FakeTranscriber:
        async def transcribe(self, attachment):
            raise AssertionError("no voice files expected")

    task = anyio_run(
        build_agent_task_from_bitrix_chat(
            _bitrix_v2_message_payload(),
            attachment_service=FakeAttachmentService(),
            transcriber=FakeTranscriber(),
            settings=None,
        )
    )

    assert task.source == "bitrix24_chat"
    assert task.user.id == "9"
    assert task.user.raw["dialog_id"] == "chat99"
    assert "Битриксе" in task.request
    assert task.context["channel_id"] == "bitrix24"
    assert task.context["recipient_id"] == "chat99"


def test_build_agent_task_from_bitrix_chat_transcribes_voice(tmp_path):
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

    payload = _bitrix_v2_message_payload()
    payload["data"]["message"] = {
        "id": 123,
        "authorId": 9,
        "text": "",
        "files": [{"id": 501, "name": "voice.ogg", "type": "voice"}],
    }

    task = anyio_run(
        build_agent_task_from_bitrix_chat(
            payload,
            attachment_service=FakeAttachmentService(),
            transcriber=FakeTranscriber(),
            settings=None,
        )
    )

    assert task.request == "Создай задачу по камере"
    assert task.context["transcriptions"][0]["text"] == "Создай задачу по камере"


def test_bitrix_api_tool_write_executes_directly():
    """BitrixApiTool should execute write methods directly via write_client.call()."""
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix, write_client=fake_bitrix)

    result = anyio_run(
        tool.execute(
            {"method": "tasks.task.add", "params": {"fields": {"TITLE": "Тест"}}, "summary": "создать задачу"},
            user_id=9,
        )
    )

    assert result.status == ToolStatus.OK
    assert any(method == "tasks.task.add" for method, _ in fake_bitrix.calls)


def test_bitrix_api_tool_dry_run_blocks_write():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix, write_client=fake_bitrix, dry_run=True)

    result = anyio_run(tool.execute({"method": "tasks.task.add", "params": {"fields": {"TITLE": "Тест"}}}, user_id=9))

    assert result.status == ToolStatus.DRY_RUN
    assert fake_bitrix.calls == []


def test_bitrix_api_tool_write_no_write_client_returns_not_configured():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix, write_client=None)

    result = anyio_run(tool.execute({"method": "tasks.task.add", "params": {"fields": {"TITLE": "Тест"}}}, user_id=9))

    assert result.status == ToolStatus.NOT_CONFIGURED


def test_bitrix_api_tool_write_empty_params_returns_invalid():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix, write_client=fake_bitrix)

    result = anyio_run(tool.execute({"method": "tasks.task.add", "params": {}}))

    assert result.status == ToolStatus.INVALID_TOOL_CALL


def test_bitrix_api_tool_denied_method():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix, write_client=fake_bitrix)

    result = anyio_run(tool.execute({"method": "user.delete", "params": {"ID": 9}}))

    assert result.status == ToolStatus.DENIED
    assert fake_bitrix.calls == []


def test_bitrix24_specialist_skips_quality_control_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("QUALITY_CONTROL_WEBHOOK_ENABLED", "false")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    settings = get_settings()
    manifest = get_agent_manifest("bitrix24")
    fake_bitrix = FakeBitrixClient()
    specialist = Bitrix24Specialist(
        manifest,
        bitrix_task_client=fake_bitrix,
        settings=settings,
        llm=FakeBitrixLLM(),
    )

    task = AgentTask(
        task_id="qc_test",
        request="quality_control",
        context={"bitrix_event_type": "ONTASKUPDATE", "task_id": 8413},
    )
    result = anyio_run(specialist.handle(task))

    assert result.status == "completed"
    assert "quality_control_disabled" in result.answer
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
    from ai_server.settings import get_settings

    assert (
        _token_endpoint_from_server("https://oauth.bitrix.info/rest/", get_settings())
        == "https://oauth.bitrix.info/oauth/token/"
    )


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
        from ai_server.settings import get_settings

        super().__init__(settings=get_settings(), base_url="https://example.bitrix24.ru/rest/1/webhook/")
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
