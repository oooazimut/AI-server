import json
from pathlib import Path

from fastapi.testclient import TestClient

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.main import app
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.bitrix import BitrixToolset
from tests.fakes import FakeEmbeddingProvider


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
    )

    result = anyio_run(specialist.handle(AgentTask(task_id="t1", request="Найди договор Транзит на портале")))

    action = next(item for item in result.actions_taken if item.name == "portal_search")
    assert action.status == "ok"
    assert action.details["data"]["results"][0]["entity_id"] == "101"


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)

