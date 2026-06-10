import json
from pathlib import Path

from fastapi.testclient import TestClient

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.bitrix_llm import BitrixLLMToolCall
from ai_server.integrations.bitrix.portal_search import (
    PortalSearchIndex,
    sync_disk_delta_index,
    sync_portal_content_index,
    sync_portal_index,
)
from ai_server.main import app
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.bitrix import BitrixToolset
from ai_server.workers.bitrix.search_webhook_indexer import (
    prepare_search_webhook_job,
    process_search_webhook_job,
)
from tests.fakes import FakeBitrixLLM, FakeEmbeddingProvider


def _create_index(path: Path) -> PortalSearchIndex:
    index = PortalSearchIndex(path)
    index.ensure_schema()
    with index._connect() as connection:
        connection.execute(
            """
            INSERT INTO portal_search_items (
                entity_type, entity_id, title, body, url, search_text,
                metadata_json, source_updated_at, last_seen_at, indexed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "disk_file",
                "101",
                "Договор транзит экспресс.docx",
                "Текст договора с компанией Транзит-Экспресс и приложениями.",
                "https://example.test/docs/101",
                "disk_file 101 договор транзит экспресс.docx текст договора с компанией транзит-экспресс и приложениями.",
                json.dumps({"content_index_status": "indexed", "path": "/Договоры"}, ensure_ascii=False),
                "2026-06-01T10:00:00+03:00",
                "2026-06-01T10:00:00+03:00",
                "2026-06-01T10:00:00+03:00",
            ),
        )
        connection.execute(
            """
            INSERT INTO portal_search_items (
                entity_type, entity_id, title, body, url, search_text,
                metadata_json, source_updated_at, last_seen_at, indexed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "task",
                "202",
                "Проверить камеру",
                "Задача по IP-камере на складе.",
                "https://example.test/tasks/202",
                "task 202 проверить камеру задача по ip-камере на складе.",
                "{}",
                "2026-06-01T10:00:00+03:00",
                "2026-06-01T10:00:00+03:00",
                "2026-06-01T10:00:00+03:00",
            ),
        )
    return index


def test_portal_search_index_searches_old_schema(tmp_path):
    index = _create_index(tmp_path / "search_index.sqlite")

    results = index.search("транзит договор", limit=5)
    stats = index.stats()

    assert results
    assert results[0].entity_type == "disk_file"
    assert results[0].entity_id == "101"
    assert stats.total_items == 2
    assert stats.by_type["disk_file"] == 1
    assert stats.content_by_status["indexed"] == 1


def test_portal_search_tool_returns_results(tmp_path):
    index = _create_index(tmp_path / "search_index.sqlite")
    toolset = BitrixToolset(portal_search=index)

    result = toolset.portal_search_contract({"query": "транзит договор", "scope": "documents", "limit": 5})

    assert result.status == "ok"
    assert result.data["results"][0]["entity_type"] == "disk_file"
    assert "Нашёл по порталу" in result.data["summary"]


def test_portal_search_tool_reports_missing_index(tmp_path):
    toolset = BitrixToolset(portal_search=PortalSearchIndex(tmp_path / "missing.sqlite"))

    result = toolset.portal_search_contract({"query": "договор", "scope": "documents"})

    assert result.status == "not_configured"
    assert "missing" in result.data["message"].lower()


def test_bitrix_search_endpoint(monkeypatch, tmp_path):
    var_dir = tmp_path / "var"
    _create_index(var_dir / "search_index.sqlite")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(var_dir))

    with TestClient(app) as client:
        response = client.get("/bitrix/search", params={"q": "транзит договор", "scope": "documents"})
        status = client.get("/bitrix/search/status")

    assert response.status_code == 200
    assert response.json()["results"][0]["entity_type"] == "disk_file"
    assert status.json()["total_items"] == 2


def test_bitrix_specialist_uses_portal_search_for_document_requests(tmp_path):
    manifest = get_agent_manifest("bitrix24")
    assert manifest is not None
    index = _create_index(tmp_path / "search_index.sqlite")
    specialist = Bitrix24Specialist(
        manifest,
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        tools=BitrixToolset(portal_search=index),
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


def test_portal_metadata_sync_indexes_tasks_projects_and_disk(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    index = PortalSearchIndex(tmp_path / "search_index.sqlite")
    bitrix = FakePortalBitrix()

    stats = anyio_run(sync_portal_index(bitrix, index))

    assert stats.tasks == 1
    assert stats.projects == 1
    assert stats.task_attachments == 1
    assert stats.disk_items == 3
    assert index.get_item(entity_type="task", entity_id=202) is not None
    assert index.get_item(entity_type="project", entity_id=17) is not None
    assert index.get_item(entity_type="disk_file", entity_id=501) is not None
    assert index.search("план склад", entity_types={"disk_file"})


def test_portal_delta_sync_updates_folder_and_deletes_missing_children(tmp_path):
    index = PortalSearchIndex(tmp_path / "search_index.sqlite")
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
        )
    )

    assert stats.folders_scanned == 1
    assert stats.items_seen == 2
    assert stats.deleted == 1
    assert index.get_item(entity_type="disk_file", entity_id=501) is not None
    assert index.get_item(entity_type="disk_file", entity_id=999) is None


def test_portal_content_sync_indexes_downloaded_text(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("SEARCH_CONTENT_MAX_FILES", "10")
    index = PortalSearchIndex(tmp_path / "search_index.sqlite")
    index.upsert_item(
        entity_type="disk_file",
        entity_id=501,
        title="План склада.txt",
        body="Диск: Общий диск\nПуть: Общий диск/План склада.txt",
        metadata={"disk_object_id": 501, "size": 128},
        source_updated_at="2026-06-02T10:00:00+03:00",
    )

    stats = anyio_run(sync_portal_content_index(FakePortalBitrix(), index))

    item = index.get_item(entity_type="disk_file", entity_id=501)
    assert item is not None
    assert stats.indexed == 1
    assert item.metadata["content_index_status"] == "indexed"
    assert "секретное слово альфа" in item.body.lower()
    assert index.search("альфа", entity_types={"disk_file"})


def test_search_webhook_indexer_upserts_and_deletes_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SEARCH_WEBHOOK_INDEXER_ENABLED", "true")
    monkeypatch.setenv("SEARCH_WEBHOOK_CONTENT_ENABLED", "false")
    index = PortalSearchIndex(tmp_path / "search_index.sqlite")
    status: dict[str, object] = {}

    job, prepared = prepare_search_webhook_job({"event": "ONDISKFILEUPDATE", "data": {"FIELDS_AFTER": {"ID": "777"}}})

    assert job is not None
    assert prepared["handled"] is True

    result = anyio_run(
        process_search_webhook_job(
            FakePortalBitrix(),
            index,
            job,
            status=status,
        )
    )

    assert result["reason"] == "metadata_indexed"
    assert index.get_item(entity_type="disk_file", entity_id=777) is not None

    delete_job, _ = prepare_search_webhook_job({"event": "ONDISKFILEDELETE", "FILE_ID": "777"})
    assert delete_job is not None
    delete_result = anyio_run(
        process_search_webhook_job(
            FakePortalBitrix(),
            index,
            delete_job,
            status=status,
        )
    )

    assert delete_result["reason"] == "deleted"
    assert index.get_item(entity_type="disk_file", entity_id=777) is None


class FakePortalBitrix:
    async def list_all_tasks(self, **kwargs):
        return [
            {
                "ID": 202,
                "TITLE": "Проверить камеру на складе",
                "DESCRIPTION": "IP-камера, регистратор и склад",
                "STATUS": 2,
                "RESPONSIBLE_ID": 9,
                "CREATED_BY": 1,
                "GROUP_ID": 17,
                "DEADLINE": "2026-06-10T09:00:00+03:00",
                "CHANGED_DATE": "2026-06-02T09:00:00+03:00",
                "UF_TASK_WEBDAV_FILES": ["n701"],
            }
        ]

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
