from types import SimpleNamespace

from fastapi.testclient import TestClient

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.bitrix24.tools import BitrixApiTool, BitrixWarehouseSearchTool
from ai_server.attachments import StoredAttachment
from ai_server.integrations.bitrix.chat_parser import build_agent_task_from_bitrix_chat
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.events import parse_incoming_message
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, _token_endpoint_from_server
from ai_server.main import app
from ai_server.models import AgentTask, ToolResult, ToolStatus
from ai_server.registry import get_agent_manifest
from ai_server.settings import get_settings
from ai_server.tools.bitrix_policy import apply_write_policy
from ai_server.transcription import TranscriptionResult
from scripts.create_bitrix_dev_chat import chat_reference, create_chat, sanitize_result
from tests.fakes import FakePortalSearchIndex


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


def test_bitrix_events_endpoint_enqueues_payload(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
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


def test_bitrix_events_endpoint_rejects_wrong_secret(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
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


def test_bitrix_api_tool_generic_write_is_denied_even_without_oauth_requirement():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(
        tool.execute(
            {
                "method": "crm.deal.update",
                "params": {"id": 1, "fields": {"TITLE": "Тест"}},
                "summary": "обновить сделку",
            },
            user_id=9,
        )
    )

    assert result.status == ToolStatus.DENIED
    assert fake_bitrix.calls == []


def test_bitrix_api_tool_generic_write_is_denied_before_oauth():
    fallback_bitrix = FakeBitrixClient()
    oauth_bitrix = FakeBitrixClient()
    oauth = FakeBitrixOAuth(oauth_bitrix)
    tool = BitrixApiTool(
        client=fallback_bitrix,
        bitrix_oauth=oauth,
    )

    result = anyio_run(
        tool.execute(
            {
                "method": "crm.deal.update",
                "params": {"id": 1, "fields": {"TITLE": "Тест"}},
                "summary": "обновить сделку",
            },
            user_id=9,
            dialog_id="chat99",
        )
    )

    assert result.status == ToolStatus.DENIED
    assert oauth.user_ids == []
    assert oauth_bitrix.calls == []
    assert fallback_bitrix.calls == []


def test_bitrix_api_tool_required_oauth_blocks_missing_dialog_id():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(
        tool.execute(
            {
                "method": "crm.deal.update",
                "params": {"id": 1, "fields": {"TITLE": "Тест"}},
                "summary": "обновить сделку",
            },
            user_id=9,
            dialog_id=None,
        )
    )

    assert result.status == ToolStatus.DENIED
    assert fake_bitrix.calls == []


def test_bitrix_api_tool_required_oauth_does_not_fallback_to_write_client():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(
        tool.execute(
            {
                "method": "crm.deal.update",
                "params": {"id": 1, "fields": {"TITLE": "Тест"}},
                "summary": "обновить сделку",
            },
            user_id=9,
            dialog_id="chat99",
        )
    )

    assert result.status == ToolStatus.DENIED
    assert fake_bitrix.calls == []


def test_bitrix_api_tool_dry_run_blocks_write():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(
        tool.execute({"method": "crm.deal.update", "params": {"id": 1, "fields": {"TITLE": "Тест"}}}, user_id=9)
    )

    assert result.status == ToolStatus.DENIED
    assert fake_bitrix.calls == []


def test_bitrix_api_tool_write_no_write_client_returns_not_configured():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(
        tool.execute({"method": "crm.deal.update", "params": {"id": 1, "fields": {"TITLE": "Тест"}}}, user_id=9)
    )

    assert result.status == ToolStatus.DENIED


def test_bitrix_api_tool_write_empty_params_returns_invalid():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(tool.execute({"method": "crm.deal.update", "params": {}}))

    assert result.status == ToolStatus.DENIED


def test_bitrix_api_tool_denied_method():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(tool.execute({"method": "user.delete", "params": {"ID": 9}}))

    assert result.status == ToolStatus.DENIED
    assert fake_bitrix.calls == []


def test_bitrix_api_tool_denies_direct_project_creation():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(tool.execute({"method": "sonet_group.create", "params": {"fields": {"NAME": "Проект"}}}))

    assert result.status == ToolStatus.DENIED
    assert result.error == "Use project_create_draft/project_create_confirm for Bitrix project creation."
    assert fake_bitrix.calls == []


def test_bitrix_api_tool_denies_direct_task_creation():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(tool.execute({"method": "tasks.task.add", "params": {"fields": {"TITLE": "Тест"}}}))

    assert result.status == ToolStatus.DENIED
    assert result.error == "Use task_create_draft/task_create_confirm for Bitrix task creation."
    assert fake_bitrix.calls == []


def test_bitrix_api_tool_denies_direct_task_closing_methods():
    close_methods = [
        ("tasks.task.result.add", {"taskId": 139, "fields": {"TEXT": "Готово"}}),
        ("tasks.task.complete", {"taskId": 139}),
        ("tasks.task.approve", {"taskId": 139}),
    ]

    for method, params in close_methods:
        fake_bitrix = FakeBitrixClient()
        tool = BitrixApiTool(client=fake_bitrix)

        result = anyio_run(tool.execute({"method": method, "params": params}))

        assert result.status == ToolStatus.DENIED
        assert "task_close_draft/task_close_confirm" in result.error
        assert result.data == {"method": method}
        assert fake_bitrix.calls == []


def test_bitrix_api_tool_denies_direct_calendar_event_creation():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(tool.execute({"method": "calendar.event.add", "params": {"name": "Позвонить"}}))

    assert result.status == ToolStatus.DENIED
    assert result.error == "Use calendar_event_draft/calendar_event_confirm for Bitrix calendar event creation."
    assert fake_bitrix.calls == []


def test_bitrix_api_tool_executes_exact_read_once_without_name_fallback():
    class FakeProjectClient:
        def __init__(self) -> None:
            self.calls = []

        async def result(self, method, payload=None, *, base_url=None):
            self.calls.append((method, payload or {}))
            if method == "sonet_group.get" and (payload or {}).get("FILTER", {}).get("%NAME"):
                return []
            if method == "sonet_group.get":
                return [
                    {"ID": "39", "NAME": "Логан"},
                    {"ID": "45", "NAME": "Ларгус 2"},
                    {"ID": "53", "NAME": "ларгус 3"},
                ]
            return []

    fake_bitrix = FakeProjectClient()
    tool = BitrixApiTool(client=fake_bitrix)

    result = anyio_run(tool.execute({"method": "sonet_group.get", "params": {"FILTER": {"%NAME": "Ларгус-2"}}}))

    assert result.status == ToolStatus.OK
    assert result.data["result"] == []
    assert fake_bitrix.calls == [
        ("sonet_group.get", {"FILTER": {"%NAME": "Ларгус-2"}}),
    ]


def test_bitrix_api_tool_read_uses_oauth_client_when_available():
    fallback_bitrix = FakeBitrixClient()
    oauth_bitrix = FakeBitrixClient()
    oauth = FakeBitrixOAuth(oauth_bitrix)
    tool = BitrixApiTool(client=fallback_bitrix, bitrix_oauth=oauth)

    result = anyio_run(tool.execute({"method": "catalog.store.list", "params": {}}, user_id=13))

    assert result.status == ToolStatus.OK
    assert result.data["access_actor"] == "oauth_current_user"
    assert oauth.user_ids == [13]
    assert fallback_bitrix.calls == []
    assert ("catalog.store.list", {}) in oauth_bitrix.calls


def test_bitrix_api_tool_oauth_read_denies_without_user_id():
    fallback_bitrix = FakeBitrixClient()
    oauth_bitrix = FakeBitrixClient()
    tool = BitrixApiTool(client=fallback_bitrix, bitrix_oauth=FakeBitrixOAuth(oauth_bitrix))

    result = anyio_run(tool.execute({"method": "catalog.store.list", "params": {}}))

    assert result.status == ToolStatus.DENIED
    assert fallback_bitrix.calls == []
    assert oauth_bitrix.calls == []


def test_bitrix_warehouse_search_tool_finds_store_and_products():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixWarehouseSearchTool(client=fake_bitrix)

    result = anyio_run(
        tool.execute(
            {
                "query": "Borisov warehouse",
                "store_id": 10,
                "list_all": False,
                "include_products": True,
                "limit": 5,
                "product_limit": 5,
                "product_offset": 0,
            }
        )
    )

    assert result.status == ToolStatus.OK
    assert result.data["matches"][0]["id"] == 10
    assert result.data["products"]["items"][0]["product_id"] == 1001
    assert result.data["products"]["items"][0]["product_name"] == "Cable"
    assert result.data["products"]["items"][0]["product_url"] == "/shop/documents-catalog/7/product/1001/"
    assert ("catalog.store.list", {}) in fake_bitrix.calls
    assert any(method == "catalog.storeproduct.list" for method, _ in fake_bitrix.calls)


def test_bitrix_warehouse_search_tool_lists_all_stores_without_literal_name_search():
    fake_bitrix = FakeBitrixClient()
    tool = BitrixWarehouseSearchTool(client=fake_bitrix)

    result = anyio_run(
        tool.execute(
            {
                "query": "все",
                "list_all": True,
                "include_products": False,
                "limit": 10,
                "product_limit": 50,
                "product_offset": 0,
            }
        )
    )

    assert result.status == ToolStatus.OK
    assert [item["id"] for item in result.data["matches"]] == [10, 11]
    assert result.data["list_all"] is True
    assert "products" not in result.data


def test_bitrix_warehouse_search_tool_uses_oauth_client_for_live_reads():
    fallback_bitrix = FakeBitrixClient()
    oauth_bitrix = FakeBitrixClient()
    oauth = FakeBitrixOAuth(oauth_bitrix)
    tool = BitrixWarehouseSearchTool(client=fallback_bitrix, bitrix_oauth=oauth)

    result = anyio_run(
        tool.execute(
                {
                    "query": "Borisov warehouse",
                    "store_id": 10,
                    "list_all": False,
                    "include_products": True,
                    "limit": 5,
                    "product_limit": 5,
                    "product_offset": 0,
                },
            user_id=13,
        )
    )

    assert result.status == ToolStatus.OK
    assert result.data["source"] == "live_bitrix_rest"
    assert result.data["access_actor"] == "oauth_current_user"
    assert oauth.user_ids == [13]
    assert fallback_bitrix.calls == []
    assert ("catalog.store.list", {}) in oauth_bitrix.calls
    assert any(method == "catalog.storeproduct.list" for method, _ in oauth_bitrix.calls)
    assert any(method == "catalog.product.list" for method, _ in oauth_bitrix.calls)


def test_bitrix_warehouse_search_tool_oauth_read_denies_live_lookup_without_user_id():
    fallback_bitrix = FakeBitrixClient()
    oauth_bitrix = FakeBitrixClient()
    tool = BitrixWarehouseSearchTool(client=fallback_bitrix, bitrix_oauth=FakeBitrixOAuth(oauth_bitrix))

    result = anyio_run(
        tool.execute(
            {
                "query": "Borisov warehouse",
                "store_id": 10,
                "list_all": False,
                "include_products": True,
                "limit": 5,
                "product_limit": 5,
                "product_offset": 0,
            }
        )
    )

    assert result.status == ToolStatus.DENIED
    assert fallback_bitrix.calls == []
    assert oauth_bitrix.calls == []


def test_bitrix_warehouse_search_tool_live_verifies_stock_instead_of_serving_stale_snapshot():
    fake_bitrix = FakeBitrixClient()
    index = FakePortalSearchIndex()
    index.upsert_item(
        entity_type="catalog_store",
        entity_id=10,
        title="Borisov warehouse",
        body="Borisov address",
        metadata={},
    )
    index.upsert_item(
        entity_type="catalog_store_stock",
        entity_id="10:1001",
        title="Cable - Borisov warehouse",
        body="Store: Borisov warehouse\nProduct: Cable\nAmount: 3",
        url="https://example.test/shop/documents-catalog/7/product/1001/",
        metadata={
            "store_id": 10,
            "store_title": "Borisov warehouse",
            "store_address": "Borisov address",
            "product_id": 1001,
            "product_name": "Cable",
            "iblock_id": 7,
            "amount": "3",
            "product_url": "https://example.test/shop/documents-catalog/7/product/1001/",
            "positive_amount": True,
        },
    )
    tool = BitrixWarehouseSearchTool(client=fake_bitrix, portal_search=index)

    result = anyio_run(
        tool.execute(
            {
                "query": "Borisov warehouse",
                "store_id": 10,
                "list_all": False,
                "include_products": True,
                "limit": 5,
                "product_limit": 5,
                "product_offset": 0,
            }
        )
    )

    assert result.status == ToolStatus.OK
    assert result.data["source"] == "postgres_portal_snapshot"
    assert result.data["products"]["items"][0]["product_id"] == 1001
    assert result.data["products"]["items"][0]["product_name"] == "Cable"
    assert result.data["products"]["items"][0]["amount"] == "3"
    assert fake_bitrix.calls == []


def test_bitrix_warehouse_uses_snapshot_before_live_oauth():
    fallback_bitrix = FakeBitrixClient()
    oauth_bitrix = FakeBitrixClient()
    oauth = FakeBitrixOAuth(oauth_bitrix)
    index = FakePortalSearchIndex()
    index.upsert_item(
        entity_type="catalog_store",
        entity_id=10,
        title="Borisov warehouse",
        body="Borisov address",
        metadata={},
    )
    tool = BitrixWarehouseSearchTool(client=fallback_bitrix, portal_search=index, bitrix_oauth=oauth)

    result = anyio_run(
        tool.execute(
            {
                "query": "Borisov warehouse",
                "store_id": 10,
                "list_all": False,
                "include_products": False,
                "limit": 5,
                "product_limit": 50,
                "product_offset": 0,
            },
            user_id=13,
        )
    )

    assert result.status == ToolStatus.OK
    assert result.data["source"] == "postgres_portal_snapshot"
    assert result.data["access_actor"] == "postgres_snapshot"
    assert oauth.user_ids == []
    assert fallback_bitrix.calls == []
    assert oauth_bitrix.calls == []


def test_bitrix_warehouse_snapshot_does_not_require_user_oauth_for_read():
    fallback_bitrix = FakeBitrixClient()
    oauth_bitrix = FakeBitrixClient()
    index = FakePortalSearchIndex()
    index.upsert_item(
        entity_type="catalog_store",
        entity_id=10,
        title="Borisov warehouse",
        body="Borisov address",
        metadata={},
    )
    tool = BitrixWarehouseSearchTool(
        client=fallback_bitrix,
        portal_search=index,
        bitrix_oauth=FakeBitrixOAuth(oauth_bitrix),
    )

    result = anyio_run(
        tool.execute(
            {
                "query": "Borisov warehouse",
                "store_id": 10,
                "list_all": False,
                "include_products": False,
                "limit": 5,
                "product_limit": 50,
                "product_offset": 0,
            }
        )
    )

    assert result.status == ToolStatus.OK
    assert result.data["source"] == "postgres_portal_snapshot"
    assert fallback_bitrix.calls == []
    assert oauth_bitrix.calls == []


def test_bitrix_warehouse_search_tool_filters_non_available_products_before_limit():
    class FakeWarehouseClient(FakeBitrixClient):
        async def result(self, method, payload=None, *, base_url=None):
            self.calls.append((method, payload or {}))
            if method == "catalog.store.list":
                return {"stores": [{"id": 10, "title": "Borisov warehouse", "address": "Borisov"}]}
            if method == "catalog.storeproduct.list":
                return {
                    "storeProducts": [
                        {"storeId": 10, "productId": 1001, "amount": "0"},
                        {"storeId": 10, "productId": 1002},
                        {"storeId": 10, "productId": 1003, "amount": "2"},
                        {"storeId": 10, "productId": 1004, "amount": "5"},
                        {"storeId": 10, "productId": 1005, "amount": ""},
                    ]
                }
            if method == "catalog.product.list":
                return {
                    "products": [
                        {"id": 1003, "iblockId": 7, "name": "Cable"},
                        {"id": 1004, "iblockId": 7, "name": "Switch"},
                    ]
                }
            return {}

    fake_bitrix = FakeWarehouseClient()
    tool = BitrixWarehouseSearchTool(client=fake_bitrix)

    result = anyio_run(
        tool.execute(
            {
                "query": "Borisov warehouse",
                "store_id": 10,
                "list_all": False,
                "include_products": True,
                "limit": 5,
                "product_limit": 1,
                "product_offset": 0,
            }
        )
    )

    assert result.status == ToolStatus.OK
    products = result.data["products"]
    assert [item["product_id"] for item in products["items"]] == [1003]
    assert products["filtered_non_positive_count"] == 3
    assert products["available_items_seen"] == 2
    assert products["has_more"] is True


def test_bitrix_warehouse_search_reads_every_page_then_filters_sorts_and_limits():
    class PagedWarehouseClient(FakeBitrixClient):
        async def collect_paged(self, method, payload=None, *, list_key=None, limit=None, base_url=None):
            self.calls.append((method, payload or {}))
            if method == "catalog.store.list":
                return [{"id": 10, "title": "Main warehouse", "address": "Main"}]
            if method == "catalog.storeproduct.list":
                return [
                    {"storeId": 10, "productId": product_id, "amount": "1"}
                    for product_id in range(1, 121)
                ]
            if method == "catalog.product.list":
                ids = list((payload or {}).get("filter", {}).get("id") or [])
                return [
                    {
                        "id": product_id,
                        "iblockId": 7,
                        "name": "Амортизатор" if product_id == 120 else f"Товар {121 - product_id:03d}",
                    }
                    for product_id in ids
                ]
            return []

    client = PagedWarehouseClient()
    tool = BitrixWarehouseSearchTool(client=client)
    all_products = anyio_run(
        tool.execute(
            {
                "query": "Main warehouse",
                "store_id": 10,
                "list_all": False,
                "include_products": True,
                "limit": 10,
                "product_limit": 50,
                "product_offset": 0,
            }
        )
    )
    products = all_products.data["products"]

    assert products["total_rows_seen"] == 120
    assert products["available_items_with_names"] == 120
    assert products["has_more"] is True
    assert len(products["items"]) == 50
    assert [item["product_name"] for item in products["items"][:3]] == [
        "Амортизатор",
        "Товар 002",
        "Товар 003",
    ]

    filtered = anyio_run(
        tool.execute(
            {
                "query": "Main warehouse",
                "store_id": 10,
                "product_query": "амортизатор",
                "list_all": False,
                "include_products": True,
                "limit": 10,
                "product_limit": 50,
                "product_offset": 0,
            }
        )
    )

    assert filtered.data["products"]["available_items_with_names"] == 1
    assert [item["product_id"] for item in filtered.data["products"]["items"]] == [120]


def test_bitrix24_specialist_rejects_any_free_text(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    settings = get_settings()
    manifest = get_agent_manifest("bitrix24")
    fake_bitrix = FakeBitrixClient()
    specialist = Bitrix24Specialist(
        manifest,
        bitrix_task_client=fake_bitrix,
        settings=settings,
    )

    task = AgentTask(
        task_id="free_text_test",
        request="покажи задачи",
    )
    result = anyio_run(specialist.handle(task))

    assert result.status == "failed"
    assert result.metadata["reason"] == "ORCHESTRATOR_STRUCTURED_COMMAND_REQUIRED"
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


def test_create_bitrix_dev_chat_passes_settings_to_client(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    constructed = []

    class FakeBitrixClient:
        def __init__(self, *, settings):
            constructed.append(settings)

        async def create_bot_chat(self, **kwargs):
            return {"chat": {"id": 8745, "dialogId": "chat8745"}, "kwargs": kwargs}

    monkeypatch.setattr("scripts.create_bitrix_dev_chat.BitrixClient", FakeBitrixClient)
    args = SimpleNamespace(
        title="AI Dev Tester",
        user_ids=[55],
        description="test chat",
        color="mint",
        message="ready",
        bot_id=231,
        owner_id=None,
    )

    result = anyio_run(create_chat(args))

    assert constructed
    assert constructed[0].bitrix_bot_auth_mode == get_settings().bitrix_bot_auth_mode
    assert result["chat"] == {"chat_id": 8745, "dialog_id": "chat8745"}
    assert result["result"]["kwargs"]["user_ids"] == [55]


def test_bitrix_oauth_service_saves_token_from_app_payload(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "example.bitrix24.ru")
    # redis.asyncio.from_url is patched by conftest to return fakeredis
    service = BitrixOAuthService(redis_url="redis://localhost/15")

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
    token = anyio_run(service.get_token(9))
    status = anyio_run(service.public_status())

    assert result.user_id == 9
    assert token is not None
    assert token.access_token == "access"
    assert status["linked_users_count"] == 1
    assert status["authorization"]["message"]


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

    async def result(self, method, payload=None, *, base_url=None):
        self.calls.append((method, payload or {}))
        if method == "catalog.store.list":
            return {
                "stores": [
                    {"id": 10, "title": "Borisov warehouse", "address": "Borisov"},
                    {"id": 11, "title": "Minsk warehouse", "address": "Minsk"},
                ]
            }
        if method == "catalog.storeproduct.list":
            return {"storeProducts": [{"storeId": 10, "productId": 1001, "amount": "7"}]}
        if method == "catalog.product.list":
            return {"products": [{"id": 1001, "iblockId": 7, "name": "Cable"}]}
        return {"id": 123}

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


class FakeBitrixOAuth:
    def __init__(self, client) -> None:
        self.client = client
        self.user_ids = []

    async def client_for_user(self, user_id: int):
        self.user_ids.append(user_id)
        return self.client


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
