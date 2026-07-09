from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ai_server.agents.bitrix24 import Bitrix24Specialist, BitrixLLMToolCall
from ai_server.agents.bitrix24.tools import PortalSearchTool
from ai_server.integrations.bitrix.portal_search import (
    sync_disk_delta_index,
    sync_portal_content_index,
    sync_portal_index,
)
from ai_server.main import app
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
from ai_server.workers.bitrix.search_webhook_indexer import (
    prepare_search_webhook_job,
    process_search_webhook_job,
)
from tests.fakes import FakeBitrixLLM, FakeEmbeddingProvider, FakePortalSearchIndex


def _create_index() -> FakePortalSearchIndex:
    index = FakePortalSearchIndex()
    index.upsert_item(
        entity_type="disk_file",
        entity_id="101",
        title="Договор транзит экспресс.docx",
        body="Текст договора с компанией Транзит-Экспресс и приложениями.",
        url="https://example.test/docs/101",
        metadata={"content_index_status": "indexed", "path": "/Договоры"},
    )
    index.upsert_item(
        entity_type="task",
        entity_id="202",
        title="Проверить камеру",
        body="Задача по IP-камере на складе.",
        url="https://example.test/tasks/202",
        metadata={},
    )
    return index


class _FakeBitrixFiles:
    def __init__(self, *, allowed_files: set[int] | None = None, allowed_attachments: set[int] | None = None) -> None:
        self.allowed_files = allowed_files if allowed_files is not None else {101}
        self.allowed_attachments = allowed_attachments if allowed_attachments is not None else set()

    async def get_disk_file_download_url(self, file_id: int) -> str:
        if file_id not in self.allowed_files:
            raise RuntimeError("access denied")
        return f"https://example.test/download/{file_id}"

    async def get_attached_object(self, attached_object_id: int):
        if attached_object_id not in self.allowed_attachments:
            raise RuntimeError("access denied")
        return {"ID": attached_object_id, "DOWNLOAD_URL": f"https://example.test/attached/{attached_object_id}"}


def test_portal_search_index_searches_old_schema():
    index = _create_index()

    results = index.search("транзит договор", limit=5)
    stats = index.stats()

    assert results
    assert results[0].entity_type == "disk_file"
    assert results[0].entity_id == "101"
    assert stats.total_items == 2
    assert stats.by_type["disk_file"] == 1
    assert stats.content_by_status["indexed"] == 1


def test_portal_search_tool_returns_results():
    import asyncio

    index = _create_index()
    tool = PortalSearchTool(portal_search=index, bitrix_files=_FakeBitrixFiles())

    result = asyncio.run(tool.execute({"query": "транзит договор", "scope": "documents", "limit": 5}))

    assert result.status == "ok"
    assert result.data["results"][0]["entity_type"] == "disk_file"
    assert "Нашёл по порталу" in result.data["summary"]


def test_portal_search_tool_requires_live_access_check_for_documents():
    import asyncio

    tool = PortalSearchTool(portal_search=_create_index())

    result = asyncio.run(tool.execute({"query": "транзит договор", "scope": "documents", "limit": 5}))

    assert result.status == "denied"
    assert "live access check" in result.error


def test_portal_search_tool_filters_inaccessible_documents():
    import asyncio

    index = _create_index()
    index.upsert_item(
        entity_type="disk_file",
        entity_id="303",
        title="Скрытый договор.docx",
        body="Текст договора с компанией Транзит-Экспресс.",
        url="https://example.test/docs/303",
        metadata={"disk_object_id": 303},
    )
    tool = PortalSearchTool(portal_search=index, bitrix_files=_FakeBitrixFiles(allowed_files={101}))

    result = asyncio.run(tool.execute({"query": "транзит договор", "scope": "documents", "limit": 5}))

    assert result.status == "ok"
    assert [item["entity_id"] for item in result.data["results"]] == ["101"]
    assert result.data["access_checked"] is True
    assert result.data["access_filtered_count"] == 1


def test_portal_search_tool_reports_missing_index():
    import asyncio

    tool = PortalSearchTool(portal_search=FakePortalSearchIndex(exists=False))

    result = asyncio.run(tool.execute({"query": "договор", "scope": "documents"}))

    assert result.status == "not_configured"
    assert "missing" in result.data["message"].lower()


def test_portal_search_tool_supports_store_scope():
    import asyncio

    index = _create_index()
    index.upsert_item(
        entity_type="catalog_store",
        entity_id="12",
        title="Borisov warehouse",
        body="Main stock location",
        metadata={},
    )
    tool = PortalSearchTool(portal_search=index)

    result = asyncio.run(tool.execute({"query": "Borisov", "scope": "stores", "limit": 5}))

    assert result.status == "ok"
    assert result.data["results"][0]["entity_type"] == "catalog_store"


def test_portal_search_tool_denies_unrestricted_all_scope():
    import asyncio

    tool = PortalSearchTool(portal_search=_create_index())

    result = asyncio.run(tool.execute({"query": "камера", "scope": "all", "limit": 5}, user_id=13))

    assert result.status == "denied"
    assert "focused non-task scope" in result.error


def test_portal_search_tool_denies_task_scope():
    import asyncio

    tool = PortalSearchTool(portal_search=_create_index())

    result = asyncio.run(tool.execute({"query": "камера", "scope": "tasks", "limit": 5}, user_id=13))

    assert result.status == "denied"
    assert "bitrix_task_search" in result.error


def test_bitrix_search_endpoint(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    index = _create_index()

    with TestClient(app) as client:
        app.state.portal_search = index
        response = client.get("/bitrix/search", params={"q": "транзит договор", "scope": "documents"})
        status = client.get("/bitrix/search/status")

    assert response.status_code == 200
    assert response.json()["results"][0]["entity_type"] == "disk_file"
    assert status.json()["total_items"] == 2


def test_bitrix_specialist_uses_portal_search_for_document_requests():
    manifest = get_agent_manifest("bitrix24")
    assert manifest is not None
    index = _create_index()
    specialist = Bitrix24Specialist(
        manifest,
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        agent_tools=[PortalSearchTool(portal_search=index, bitrix_files=_FakeBitrixFiles())],
        llm=FakeBitrixLLM(
            tool_calls=[
                BitrixLLMToolCall(
                    name="portal_search",
                    args={"query": "договор Транзит", "scope": "documents", "limit": 5},
                )
            ]
        ),
    )

    result = anyio_run(specialist.handle(AgentTask(task_id="t1", request="Найди договор Транзит на портале")))

    action = next(item for item in result.actions_taken if item.name == "portal_search")
    assert action.status == "ok"
    assert action.details["data"]["results"][0]["entity_id"] == "101"


def test_portal_metadata_sync_indexes_tasks_projects_and_disk(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    index = FakePortalSearchIndex()
    bitrix = FakePortalBitrix()

    stats = anyio_run(sync_portal_index(bitrix, index, settings=get_settings()))

    assert stats.tasks == 1
    assert stats.projects == 1
    assert stats.task_attachments == 1
    assert stats.catalog_products == 2
    assert stats.catalog_stores == 1
    assert stats.catalog_stock_rows == 1
    assert stats.disk_items == 3
    task = index.get_item(entity_type="task", entity_id=202)
    assert task is not None
    assert "понаблюдать" in task.body.casefold()
    assert "вы добавлены наблюдателем" not in task.body.casefold()
    assert task.metadata["comments_indexed"] is True
    assert task.metadata["comments_count"] == 1
    assert task.metadata["responsible_label"] == "Марат"
    assert index.get_item(entity_type="project", entity_id=17) is not None
    stock = index.get_item(entity_type="catalog_store_stock", entity_id="12:1001")
    assert stock is not None
    assert stock.metadata["store_title"] == "Borisov warehouse"
    assert stock.metadata["product_name"] == "Junction box"
    assert stock.metadata["amount"] == "3"
    assert stock.url == "https://asutp-expert.bitrix24.ru/shop/documents-catalog/7/product/1001/"
    assert index.get_item(entity_type="catalog_store_stock", entity_id="12:1002") is None
    assert index.get_item(entity_type="disk_file", entity_id=501) is not None
    assert index.search("план склад", entity_types={"disk_file"})
    assert index.search("понаблюдать", entity_types={"task"})
    assert index.search("Borisov Junction", entity_types={"catalog_store_stock"})
    assert any(method == "batch" for method, _payload in bitrix.calls)


def test_portal_delta_sync_updates_folder_and_deletes_missing_children(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    index = FakePortalSearchIndex()
    index.upsert_item(
        entity_type="disk_storage",
        entity_id=10,
        title="Общий диск",
        metadata={"root_object_id": 500},
    )
    index.upsert_item(
        entity_type="disk_file",
        entity_id=999,
        title="Старый файл.txt",
        metadata={"parent_id": 500},
    )

    stats = anyio_run(
        sync_disk_delta_index(
            FakePortalBitrix(),
            index,
            cursor_type=None,
            cursor_id=None,
            folder_limit=10,
            child_limit=10,
            settings=get_settings(),
        )
    )

    assert stats.folders_scanned == 1
    assert stats.items_seen == 2
    assert stats.deleted == 1
    assert index.get_item(entity_type="disk_file", entity_id=501) is not None
    assert index.get_item(entity_type="disk_file", entity_id=999) is None


def test_portal_content_sync_indexes_downloaded_text(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("SEARCH_CONTENT_MAX_FILES", "10")
    index = FakePortalSearchIndex()
    index.upsert_item(
        entity_type="disk_file",
        entity_id=501,
        title="План склада.txt",
        body="Диск: Общий диск\nПуть: Общий диск/План склада.txt",
        metadata={"disk_object_id": 501, "size": 128},
        source_updated_at="2026-06-02T10:00:00+03:00",
    )

    stats = anyio_run(sync_portal_content_index(FakePortalBitrix(), index, settings=get_settings()))

    item = index.get_item(entity_type="disk_file", entity_id=501)
    assert item is not None
    assert stats.indexed == 1
    assert item.metadata["content_index_status"] == "indexed"
    assert "секретное слово альфа" in item.body.lower()
    assert index.search("альфа", entity_types={"disk_file"})


def test_search_webhook_indexer_upserts_and_deletes_file(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("SEARCH_WEBHOOK_INDEXER_ENABLED", "true")
    monkeypatch.setenv("SEARCH_WEBHOOK_CONTENT_ENABLED", "false")
    settings = get_settings()
    index = FakePortalSearchIndex()
    status: dict[str, object] = {}

    job, prepared = prepare_search_webhook_job(
        {"event": "ONDISKFILEUPDATE", "data": {"FIELDS_AFTER": {"ID": "777"}}}, settings=settings
    )

    assert job is not None
    assert prepared["handled"] is True

    result = anyio_run(
        process_search_webhook_job(
            FakePortalBitrix(),
            index,
            job,
            status=status,
            settings=settings,
        )
    )

    assert result["reason"] == "metadata_indexed"
    assert index.get_item(entity_type="disk_file", entity_id=777) is not None

    delete_job, _ = prepare_search_webhook_job({"event": "ONDISKFILEDELETE", "FILE_ID": "777"}, settings=settings)
    assert delete_job is not None
    delete_result = anyio_run(
        process_search_webhook_job(
            FakePortalBitrix(),
            index,
            delete_job,
            status=status,
            settings=settings,
        )
    )

    assert delete_result["reason"] == "deleted"
    assert index.get_item(entity_type="disk_file", entity_id=777) is None


class FakePortalBitrix:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def list_all_tasks(self, **kwargs):
        return [
            {
                "ID": 202,
                "TITLE": "Проверить камеру на складе",
                "DESCRIPTION": "IP-камера, регистратор и склад",
                "STATUS": 2,
                "RESPONSIBLE_ID": 9,
                "RESPONSIBLE": {"name": "Марат"},
                "CREATED_BY": 1,
                "CREATOR": {"name": "Валерий Кулинич"},
                "GROUP_ID": 17,
                "DEADLINE": "2026-06-10T09:00:00+03:00",
                "CREATED_DATE": "2026-06-01T09:00:00+03:00",
                "CHANGED_DATE": "2026-06-02T09:00:00+03:00",
                "ACCOMPLICES": [11],
                "AUDITORS": [13],
                "UF_TASK_WEBDAV_FILES": ["n701"],
            }
        ]

    async def result(self, method: str, payload: dict):
        self.calls.append((method, payload))
        if method == "batch":
            assert "task_0" in payload["cmd"]
            assert "TASKID=202" in payload["cmd"]["task_0"]
            return {
                "result": {
                    "task_0": [
                        {
                            "POST_MESSAGE": "[USER=13]Дмитрий[/USER], вы добавлены наблюдателем.\nНеобходимо указать крайний срок, иначе задача не будет выполнена вовремя."
                        },
                        {"POST_MESSAGE": "[B]Внес изменения[/B], нужно понаблюдать хотя бы до завтра."},
                    ]
                },
                "result_error": [],
            }
        if method == "task.commentitem.getlist":
            raise AssertionError("comment sync should use batch")
        raise AssertionError(method)

    async def get_attached_object(self, attached_object_id: int):
        return {
            "ID": attached_object_id,
            "OBJECT_ID": 501,
            "NAME": "Фото камеры.jpg",
            "SIZE": 1024,
            "CREATE_TIME": "2026-06-02T09:05:00+03:00",
            "DOWNLOAD_URL": "https://example.test/download/701",
        }

    async def search_projects(self, query: str = "", *, limit: int = 10):
        return [
            {
                "ID": 17,
                "NAME": "Склад",
                "DESCRIPTION": "Проект склада",
                "OWNER_ID": 1,
                "ACTIVE": "Y",
                "PROJECT": "Y",
                "DATE_UPDATE": "2026-06-02T08:00:00+03:00",
            }
        ][:limit]

    async def list_catalogs(self):
        return [{"iblockId": 7}]

    async def list_catalog_products(self, iblock_id: int, *, limit: int | None = None):
        products = [
            {
                "id": 1001,
                "iblockId": iblock_id,
                "name": "Junction box",
                "previewText": "Electrical catalog item",
                "detailText": "",
            },
            {
                "id": 1002,
                "iblockId": iblock_id,
                "name": "Selector",
                "previewText": "",
                "detailText": "",
            },
        ]
        return products[:limit] if limit else products

    async def list_catalog_stores(self, *, limit: int | None = None):
        stores = [{"id": 12, "title": "Borisov warehouse", "address": "Russian, 8", "description": ""}]
        return stores[:limit] if limit else stores

    async def list_catalog_store_products(self, store_id: object, *, limit: int | None = None):
        rows = [
            {"storeId": store_id, "productId": 1001, "amount": "3"},
            {"storeId": store_id, "productId": 1002, "amount": "0"},
        ]
        return rows[:limit] if limit else rows

    async def list_disk_storages(self, *, limit: int | None = None):
        storages = [{"ID": 10, "ROOT_OBJECT_ID": 500, "NAME": "Общий диск"}]
        return storages[:limit] if limit else storages

    async def list_disk_folder_children_all(
        self,
        *,
        folder_id: int,
        filter_: dict | None = None,
        limit: int | None = None,
    ):
        if folder_id == 500:
            children = [
                {
                    "ID": 501,
                    "NAME": "План склада.pdf",
                    "TYPE": "file",
                    "DETAIL_URL": "/docs/file/501/",
                    "UPDATE_TIME": "2026-06-02T10:00:00+03:00",
                    "SIZE": 2048,
                },
                {
                    "ID": 502,
                    "NAME": "Чертежи",
                    "TYPE": "folder",
                    "DETAIL_URL": "/docs/folder/502/",
                    "UPDATE_TIME": "2026-06-02T10:01:00+03:00",
                },
            ]
            return children[:limit] if limit else children
        return []

    async def get_disk_file(self, file_id: int):
        return {
            "ID": file_id,
            "NAME": "Схема подключения.txt",
            "TYPE": "file",
            "DETAIL_URL": "/docs/file/777/",
            "STORAGE_NAME": "Общий диск",
            "PATH": "Общий диск/Схемы",
            "STORAGE_ID": 10,
            "PARENT_ID": 500,
            "UPDATE_TIME": "2026-06-02T11:00:00+03:00",
            "SIZE": 4096,
        }

    async def get_disk_file_download_url(self, file_id: int):
        return f"fake://disk/{file_id}"

    async def download_file_from_url(self, url: str, destination: Path, *, max_bytes: int):
        data = "Содержимое файла: секретное слово Альфа и данные склада.".encode()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return len(data)


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)
