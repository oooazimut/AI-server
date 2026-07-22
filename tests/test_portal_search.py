from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from ai_server.agents.bitrix24 import Bitrix24Specialist, BitrixLLMToolCall
from ai_server.agents.bitrix24.tools import PortalSearchTool
from ai_server.integrations.bitrix.client import BitrixApiError
from ai_server.integrations.bitrix.portal_search import (
    sync_disk_delta_index,
    sync_portal_content_index,
    sync_portal_index,
    sync_task_item,
)
from ai_server.integrations.bitrix.task_close_control import (
    TASK_CLOSE_DECISION_CONTROLLED,
    TASK_CLOSE_DECISION_IGNORED_BEFORE_START,
    TASK_CLOSE_DECISION_IGNORED_USER_NOT_CONTROLLED,
    task_close_event_key,
)
from ai_server.integrations.bitrix.task_close_direct_queue import (
    TASK_CLOSE_DIRECT_STATUS_ACTIVE,
    direct_close_state_key,
)
from ai_server.integrations.bitrix.task_close_reports import task_close_report_state_key
from ai_server.main import app
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
from ai_server.workers.bitrix.search_webhook_indexer import (
    prepare_search_webhook_job,
    process_search_webhook_job,
)
from tests.fakes import FakeBitrixLLM, FakeEmbeddingProvider, FakeOrchestratorStore, FakePortalSearchIndex


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
        self.calls: list[tuple[str, int]] = []

    async def get_disk_file_download_url(self, file_id: int) -> str:
        self.calls.append(("get_disk_file_download_url", file_id))
        if file_id not in self.allowed_files:
            raise RuntimeError("access denied")
        return f"https://example.test/download/{file_id}"

    async def get_attached_object(self, attached_object_id: int):
        self.calls.append(("get_attached_object", attached_object_id))
        if attached_object_id not in self.allowed_attachments:
            raise RuntimeError("access denied")
        return {"ID": attached_object_id, "DOWNLOAD_URL": f"https://example.test/attached/{attached_object_id}"}


class _FakeBitrixOAuth:
    def __init__(self, client: _FakeBitrixFiles) -> None:
        self.client = client
        self.user_ids: list[int] = []

    async def client_for_user(self, user_id: int):
        self.user_ids.append(user_id)
        return self.client


class _FakeLiveBitrixFiles(_FakeBitrixFiles):
    def __init__(self, *, fail_live: bool = False) -> None:
        super().__init__(allowed_files=set(range(1, 1000)))
        self.fail_live = fail_live
        self.live_calls: list[tuple[str, int | None]] = []

    async def list_disk_storages(self, *, limit: int | None = None):
        self.live_calls.append(("list_disk_storages", limit))
        if self.fail_live:
            raise RuntimeError("live unavailable")
        return [{"ID": 1, "NAME": "Shared", "ROOT_OBJECT_ID": 100}]

    async def list_disk_folder_children_all(self, *, folder_id: int, filter_=None, limit: int | None = None):
        self.live_calls.append(("list_disk_folder_children_all", folder_id))
        if self.fail_live:
            raise RuntimeError("live unavailable")
        return [
            {"ID": 501, "NAME": "invoice alpha.pdf", "TYPE": "file", "DETAIL_URL": "/disk/501"},
            {"ID": 502, "NAME": "unrelated.txt", "TYPE": "file", "DETAIL_URL": "/disk/502"},
        ]


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


def test_portal_search_tool_paginates_dialog_bound_results_without_duplicates():
    import asyncio

    index = FakePortalSearchIndex()
    allowed = set(range(1, 13))
    for item_id in sorted(allowed):
        index.upsert_item(
            entity_type="disk_file",
            entity_id=item_id,
            title=f"alpha document {item_id:02d}",
            metadata={"disk_object_id": item_id},
        )
    store = FakeOrchestratorStore()
    tool = PortalSearchTool(
        portal_search=index,
        bitrix_files=_FakeBitrixFiles(allowed_files=allowed),
        state_store=store,
    )

    first = asyncio.run(tool.execute({"query": "alpha", "scope": "documents"}, dialog_key="dialog-1"))
    second = asyncio.run(tool.execute({"continuation": "next"}, dialog_key="dialog-1"))

    first_ids = [item["entity_id"] for item in first.data["results"]]
    second_ids = [item["entity_id"] for item in second.data["results"]]
    assert first.status == "ok"
    assert second.status == "ok"
    assert first.data["total"] == second.data["total"] == 12
    assert first.data["range_start"] == 1
    assert first.data["range_end"] == 10
    assert first.data["remaining"] == 2
    assert first.data["pages"] == 2
    assert second.data["offset"] == 10
    assert second.data["range_start"] == 11
    assert second.data["range_end"] == 12
    assert second.data["has_more"] is False
    assert set(first_ids).isdisjoint(second_ids)


def test_portal_search_tool_show_all_is_bounded_to_fifty():
    import asyncio

    index = FakePortalSearchIndex()
    allowed = set(range(1, 56))
    for item_id in sorted(allowed):
        index.upsert_item(
            entity_type="disk_file",
            entity_id=item_id,
            title=f"alpha document {item_id:02d}",
            metadata={"disk_object_id": item_id},
        )
    tool = PortalSearchTool(portal_search=index, bitrix_files=_FakeBitrixFiles(allowed_files=allowed))

    result = asyncio.run(tool.execute({"query": "alpha", "scope": "documents", "show_all": True}))

    assert result.status == "ok"
    assert result.data["limit"] == 50
    assert result.data["shown"] == 50
    assert result.data["total"] == 55
    assert result.data["remaining"] == 5


def test_portal_search_missing_index_uses_current_user_live_search_and_updates_index():
    import asyncio

    index = FakePortalSearchIndex(exists=False)
    fallback = _FakeBitrixFiles()
    live = _FakeLiveBitrixFiles()
    oauth = _FakeBitrixOAuth(live)
    tool = PortalSearchTool(
        portal_search=index,
        bitrix_files=fallback,
        bitrix_oauth=oauth,
        live_fallback_enabled=True,
        index_max_age_seconds=1,
    )

    result = asyncio.run(tool.execute({"query": "invoice alpha", "scope": "documents"}, user_id=13))

    assert result.status == "ok"
    assert result.data["index_state"] == "missing"
    assert result.data["source_mode"] == "bitrix_live_current_user"
    assert result.data["access_actor"] == "oauth_current_user"
    assert result.data["stale_results_suppressed"] == 0
    assert [item["entity_id"] for item in result.data["results"]] == ["501"]
    assert result.data["results"][0]["source"] == "bitrix_live_current_user"
    assert fallback.calls == []
    assert oauth.user_ids == [13]
    assert live.live_calls
    assert index.get_item(entity_type="disk_file", entity_id="501") is not None


def test_portal_search_stale_index_suppresses_snapshot_when_current_user_live_check_fails():
    import asyncio
    from dataclasses import replace

    class _StaleIndex(FakePortalSearchIndex):
        def stats(self):
            return replace(super().stats(), last_indexed_at="2000-01-01T00:00:00+00:00")

    index = _StaleIndex()
    index.upsert_item(
        entity_type="disk_file",
        entity_id=101,
        title="alpha stale document",
        metadata={"disk_object_id": 101},
    )
    live = _FakeLiveBitrixFiles(fail_live=True)
    tool = PortalSearchTool(
        portal_search=index,
        bitrix_oauth=_FakeBitrixOAuth(live),
        live_fallback_enabled=True,
        index_max_age_seconds=1,
    )

    result = asyncio.run(tool.execute({"query": "alpha", "scope": "documents"}, user_id=13))

    assert result.status == "error"
    assert result.data["index_state"] == "stale"
    assert result.data["stale_results_suppressed"] == 1
    assert result.data["results"] == []
    assert "live verification failed" in result.error


def test_portal_search_uses_durable_indexer_success_instead_of_item_max_timestamp(tmp_path):
    import asyncio
    import json
    from datetime import UTC, datetime

    state_path = tmp_path / "search-indexer-state.json"
    state_path.write_text(
        json.dumps(
            {
                "last_metadata_sync_at": datetime.now(UTC).isoformat(),
                "last_delta_sync_at": None,
                "consecutive_errors": 0,
                "last_error": None,
            }
        ),
        encoding="utf-8",
    )
    tool = PortalSearchTool(
        portal_search=_create_index(),
        bitrix_files=_FakeBitrixFiles(),
        index_max_age_seconds=3600,
        index_freshness_path=state_path,
    )

    result = asyncio.run(tool.execute({"query": "транзит договор", "scope": "documents"}))

    assert result.status == "ok"
    assert result.data["index_state"] == "fresh"
    assert result.data["index_freshness_source"] == "indexer_state:last_metadata_sync_at"
    assert result.data["source_mode"] == "bitrix_postgresql"


def test_portal_search_indexer_without_success_stays_stale_after_live_item_refresh(tmp_path):
    import asyncio
    import json

    index = _create_index()
    state_path = tmp_path / "search-indexer-state.json"
    state_path.write_text(
        json.dumps(
            {
                "last_metadata_sync_at": None,
                "last_delta_sync_at": None,
                "consecutive_errors": 0,
                "last_error": None,
            }
        ),
        encoding="utf-8",
    )
    live = _FakeLiveBitrixFiles()
    tool = PortalSearchTool(
        portal_search=index,
        bitrix_oauth=_FakeBitrixOAuth(live),
        live_fallback_enabled=True,
        index_max_age_seconds=3600,
        index_freshness_path=state_path,
    )

    first = asyncio.run(tool.execute({"query": "invoice alpha", "scope": "documents"}, user_id=13))
    second = asyncio.run(tool.execute({"query": "invoice alpha", "scope": "documents"}, user_id=13))

    assert first.status == second.status == "ok"
    assert first.data["index_state"] == second.data["index_state"] == "stale"
    assert first.data["index_freshness_source"] == "indexer_state_missing_success"
    assert second.data["index_freshness_source"] == "indexer_state_missing_success"
    assert first.data["source_mode"] == second.data["source_mode"] == "bitrix_live_current_user"


def test_portal_search_tool_uses_oauth_client_for_document_access_check():
    import asyncio

    index = _create_index()
    fallback_files = _FakeBitrixFiles()
    oauth_files = _FakeBitrixFiles()
    oauth = _FakeBitrixOAuth(oauth_files)
    tool = PortalSearchTool(portal_search=index, bitrix_files=fallback_files, bitrix_oauth=oauth)

    result = asyncio.run(
        tool.execute({"query": "С‚СЂР°РЅР·РёС‚ РґРѕРіРѕРІРѕСЂ", "scope": "documents", "limit": 5}, user_id=13)
    )

    assert result.status == "ok"
    if not result.data["results"]:
        result = asyncio.run(tool.execute({"query": "docx", "scope": "documents", "limit": 5}, user_id=13))
    assert result.data["access_checked"] is True
    assert result.data["access_actor"] == "oauth_current_user"
    assert oauth.user_ids
    assert set(oauth.user_ids) == {13}
    assert fallback_files.calls == []
    assert oauth_files.calls == [("get_disk_file_download_url", 101)]


def test_portal_search_tool_oauth_document_access_denies_without_user_id():
    import asyncio

    index = _create_index()
    fallback_files = _FakeBitrixFiles()
    oauth_files = _FakeBitrixFiles()
    tool = PortalSearchTool(
        portal_search=index,
        bitrix_files=fallback_files,
        bitrix_oauth=_FakeBitrixOAuth(oauth_files),
    )

    result = asyncio.run(tool.execute({"query": "С‚СЂР°РЅР·РёС‚ РґРѕРіРѕРІРѕСЂ", "scope": "documents"}))

    assert result.status == "denied"
    assert fallback_files.calls == []
    assert oauth_files.calls == []


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


def test_portal_search_tool_uses_oauth_actor_for_store_scope():
    import asyncio

    index = _create_index()
    index.upsert_item(
        entity_type="catalog_store",
        entity_id="12",
        title="Borisov warehouse",
        body="Main stock location",
        metadata={},
    )
    oauth_files = _FakeBitrixFiles()
    oauth = _FakeBitrixOAuth(oauth_files)
    tool = PortalSearchTool(portal_search=index, bitrix_oauth=oauth)

    result = asyncio.run(tool.execute({"query": "Borisov", "scope": "stores", "limit": 5}, user_id=13))

    assert result.status == "ok"
    assert result.data["access_actor"] == "oauth_current_user"
    assert result.data["results"][0]["entity_type"] == "catalog_store"
    assert oauth.user_ids == [13]
    assert oauth_files.calls == []


def test_portal_search_tool_oauth_store_scope_denies_without_user_id():
    import asyncio

    index = _create_index()
    index.upsert_item(
        entity_type="catalog_store",
        entity_id="12",
        title="Borisov warehouse",
        body="Main stock location",
        metadata={},
    )
    oauth_files = _FakeBitrixFiles()
    tool = PortalSearchTool(portal_search=index, bitrix_oauth=_FakeBitrixOAuth(oauth_files))

    result = asyncio.run(tool.execute({"query": "Borisov", "scope": "stores", "limit": 5}))

    assert result.status == "denied"
    assert oauth_files.calls == []


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
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" in task.body
    assert task.metadata["comments_indexed"] is True
    assert task.metadata["comments_count"] == 1
    assert task.metadata["task_results_indexed"] is True
    assert task.metadata["task_results_count"] == 1
    assert task.metadata["ai_close_incomplete"] is True
    assert task.metadata["ai_close_marker"] == "AI_SERVER_TASK_CLOSE_INCOMPLETE"
    assert task.metadata["ai_close_problem_types"] == ["not_done", "unconfirmed"]
    assert task.metadata["ai_close_has_not_done"] is True
    assert task.metadata["ai_close_has_unconfirmed"] is True
    assert task.metadata["ai_close_marker_source"] == "task_result"
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
    assert index.search("AI_SERVER_TASK_CLOSE_INCOMPLETE", entity_types={"task"})
    assert index.search("Borisov Junction", entity_types={"catalog_store_stock"})
    assert any(method == "batch" for method, _payload in bitrix.calls)


def test_portal_metadata_sync_queues_controlled_direct_closed_tasks(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    index = FakePortalSearchIndex()
    index.set_task_close_control_setting(
        key="control_enabled_from",
        value="2026-07-12T00:00:00+03:00",
        updated_by=1,
    )
    index.upsert_task_close_controlled_user(
        user_id=13,
        active=True,
        updated_by=1,
        controlled_from="2026-07-12T12:15:00+03:00",
    )

    class DirectClosedBitrix(FakePortalBitrix):
        async def list_all_tasks(self, **kwargs):
            return [
                {
                    "ID": 402,
                    "TITLE": "Later closed direct task",
                    "DESCRIPTION": "Second direct close",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CHANGED_DATE": "2026-07-12T12:22:00+03:00",
                    "CLOSED_DATE": "2026-07-12T12:20:00+03:00",
                    "CLOSED_BY": 13,
                    "UF_TASK_WEBDAV_FILES": [],
                },
                {
                    "ID": 401,
                    "TITLE": "Earlier closed direct task",
                    "DESCRIPTION": "First direct close",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CHANGED_DATE": "2026-07-12T12:12:00+03:00",
                    "CLOSED_DATE": "2026-07-12T12:10:00+03:00",
                    "CLOSED_BY": 13,
                    "UF_TASK_WEBDAV_FILES": [],
                },
            ]

        async def result(self, method: str, payload: dict):
            self.calls.append((method, payload))
            if method == "batch":
                return {"result": {key: [] for key in payload["cmd"]}, "result_error": []}
            raise AssertionError(method)

    anyio_run(sync_portal_index(DirectClosedBitrix(), index, settings=get_settings()))

    early_key = task_close_event_key(task_id=401, closed_at="2026-07-12T12:10:00+03:00")
    late_key = task_close_event_key(task_id=402, closed_at="2026-07-12T12:20:00+03:00")
    early_event = index.get_task_close_control_event(task_id=401, close_event_key=early_key)
    late_event = index.get_task_close_control_event(task_id=402, close_event_key=late_key)
    assert early_event is not None
    assert late_event is not None
    assert early_event["decision"] == TASK_CLOSE_DECISION_IGNORED_BEFORE_START
    assert late_event["decision"] == TASK_CLOSE_DECISION_CONTROLLED
    assert early_event["payload"]["controlled_from"] == "2026-07-12T12:15:00+03:00"
    assert early_event["payload"]["effective_control_start"] == "2026-07-12T12:15:00+03:00"

    early_state = index.get_task_close_processing_state(task_id=401, state_key=direct_close_state_key(early_key))
    late_state = index.get_task_close_processing_state(task_id=402, state_key=direct_close_state_key(late_key))
    assert early_state is None
    assert late_state is not None
    assert late_state["status"] == TASK_CLOSE_DIRECT_STATUS_ACTIVE


def test_targeted_task_webhook_refresh_queues_controlled_direct_close(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    index = FakePortalSearchIndex()
    index.set_task_close_control_setting(
        key="control_enabled_from",
        value="2026-07-12T00:00:00+03:00",
        updated_by=1,
    )
    index.upsert_task_close_controlled_user(user_id=13, active=True, updated_by=1)

    class TargetedClosedBitrix(FakePortalBitrix):
        async def get_task(self, task_id: int, *, select=None):
            return {
                "task": {
                    "ID": task_id,
                    "TITLE": "Closed from webhook",
                    "DESCRIPTION": "Point one\nPoint two",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CHANGED_DATE": "2026-07-12T16:01:00+03:00",
                    "CLOSED_DATE": "2026-07-12T16:00:00+03:00",
                    "CLOSED_BY": 13,
                    "UF_TASK_WEBDAV_FILES": [],
                }
            }

        async def result(self, method: str, payload: dict):
            if method in {"task.commentitem.getlist", "tasks.task.result.list"}:
                return []
            return await super().result(method, payload)

    anyio_run(sync_task_item(TargetedClosedBitrix(), index, task_id=420, settings=get_settings()))

    close_key = task_close_event_key(task_id=420, closed_at="2026-07-12T16:00:00+03:00")
    event = index.get_task_close_control_event(task_id=420, close_event_key=close_key)
    state = index.get_task_close_processing_state(task_id=420, state_key=direct_close_state_key(close_key))
    assert event is not None
    assert event["decision"] == TASK_CLOSE_DECISION_CONTROLLED
    assert state is not None
    assert state["status"] == TASK_CLOSE_DIRECT_STATUS_ACTIVE
    assert state["payload"]["task_points"] == ["Point one", "Point two"]


def test_portal_metadata_sync_keeps_uncontrolled_direct_close_ignored_after_user_added(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    index = FakePortalSearchIndex()
    index.set_task_close_control_setting(
        key="control_enabled_from",
        value="2026-07-12T00:00:00+03:00",
        updated_by=1,
    )

    class DirectClosedBitrix(FakePortalBitrix):
        async def list_all_tasks(self, **kwargs):
            return [
                {
                    "ID": 410,
                    "TITLE": "Direct close by uncontrolled user",
                    "DESCRIPTION": "Closed before user was added to control",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 99,
                    "CREATED_BY": 1,
                    "CHANGED_DATE": "2026-07-12T14:00:00+03:00",
                    "CLOSED_DATE": "2026-07-12T13:55:00+03:00",
                    "CLOSED_BY": 99,
                    "UF_TASK_WEBDAV_FILES": [],
                }
            ]

        async def result(self, method: str, payload: dict):
            self.calls.append((method, payload))
            if method == "batch":
                return {"result": {key: [] for key in payload["cmd"]}, "result_error": []}
            raise AssertionError(method)

    anyio_run(sync_portal_index(DirectClosedBitrix(), index, settings=get_settings()))
    index.upsert_task_close_controlled_user(user_id=99, active=True, updated_by=1)
    anyio_run(sync_portal_index(DirectClosedBitrix(), index, settings=get_settings()))

    close_key = task_close_event_key(task_id=410, closed_at="2026-07-12T13:55:00+03:00")
    event = index.get_task_close_control_event(task_id=410, close_event_key=close_key)
    state = index.get_task_close_processing_state(task_id=410, state_key=direct_close_state_key(close_key))
    assert event is not None
    assert event["decision"] == TASK_CLOSE_DECISION_IGNORED_USER_NOT_CONTROLLED
    assert state is None


def test_portal_metadata_sync_ignores_direct_close_before_control_start(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    index = FakePortalSearchIndex()
    index.set_task_close_control_setting(
        key="control_enabled_from",
        value="2026-07-12T00:00:00+03:00",
        updated_by=1,
    )
    index.upsert_task_close_controlled_user(
        user_id=13,
        active=True,
        updated_by=1,
        controlled_from="2026-07-12T00:00:00+03:00",
    )

    class OldDirectClosedBitrix(FakePortalBitrix):
        async def list_all_tasks(self, **kwargs):
            return [
                {
                    "ID": 411,
                    "TITLE": "Old direct close",
                    "DESCRIPTION": "Closed before control launch",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CHANGED_DATE": "2026-07-10T12:05:00+03:00",
                    "CLOSED_DATE": "2026-07-10T12:00:00+03:00",
                    "CLOSED_BY": 13,
                    "UF_TASK_WEBDAV_FILES": [],
                }
            ]

        async def result(self, method: str, payload: dict):
            self.calls.append((method, payload))
            if method == "batch":
                return {"result": {key: [] for key in payload["cmd"]}, "result_error": []}
            raise AssertionError(method)

    anyio_run(sync_portal_index(OldDirectClosedBitrix(), index, settings=get_settings()))

    close_key = task_close_event_key(task_id=411, closed_at="2026-07-10T12:00:00+03:00")
    event = index.get_task_close_control_event(task_id=411, close_event_key=close_key)
    state = index.get_task_close_processing_state(task_id=411, state_key=direct_close_state_key(close_key))
    assert event is not None
    assert event["decision"] == TASK_CLOSE_DECISION_IGNORED_BEFORE_START
    assert state is None


def test_portal_metadata_sync_does_not_queue_task_with_ai_close_report(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    index = FakePortalSearchIndex()
    index.set_task_close_control_setting(
        key="control_enabled_from",
        value="2026-07-12T00:00:00+03:00",
        updated_by=1,
    )
    index.upsert_task_close_controlled_user(user_id=13, active=True, updated_by=1)

    class ReportedClosedBitrix(FakePortalBitrix):
        async def list_all_tasks(self, **kwargs):
            return [
                {
                    "ID": 412,
                    "TITLE": "Already closed through AI workflow",
                    "DESCRIPTION": "Has protected report file",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CHANGED_DATE": "2026-07-12T15:00:00+03:00",
                    "CLOSED_DATE": "2026-07-12T14:55:00+03:00",
                    "CLOSED_BY": 13,
                    "UF_TASK_WEBDAV_FILES": ["n812"],
                }
            ]

        async def result(self, method: str, payload: dict):
            self.calls.append((method, payload))
            if method == "batch":
                return {"result": {key: [] for key in payload["cmd"]}, "result_error": []}
            raise AssertionError(method)

        async def get_attached_object(self, attached_object_id: int):
            return {
                "ID": attached_object_id,
                "OBJECT_ID": 62357,
                "NAME": "AI-close-412.txt",
                "SIZE": 269,
                "CREATE_TIME": "2026-07-12T14:56:00+03:00",
                "DOWNLOAD_URL": "https://example.test/download/812",
            }

    anyio_run(sync_portal_index(ReportedClosedBitrix(), index, settings=get_settings()))

    close_key = task_close_event_key(task_id=412, closed_at="2026-07-12T14:55:00+03:00")
    task = index.get_item(entity_type="task", entity_id=412)
    assert task is not None
    assert task.metadata["task_close_control_decision"] == "skipped_ai_close_report_present"
    assert index.get_task_close_control_event(task_id=412, close_event_key=close_key) is None
    assert index.get_task_close_processing_state(task_id=412, state_key=direct_close_state_key(close_key)) is None


def test_portal_metadata_sync_marks_ai_close_report_attachment(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    index = FakePortalSearchIndex()

    class ReportAttachmentBitrix(FakePortalBitrix):
        async def list_all_tasks(self, **kwargs):
            tasks = await super().list_all_tasks(**kwargs)
            task = dict(tasks[0])
            task.update(
                {
                    "ID": 303,
                    "TITLE": "Проверить архив",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CLOSED_DATE": "2026-06-02T11:00:00+03:00",
                    "UF_TASK_WEBDAV_FILES": ["n801"],
                }
            )
            return [task]

        async def result(self, method: str, payload: dict):
            self.calls.append((method, payload))
            if method == "batch":
                if "task_result_0" in payload["cmd"]:
                    return {"result": {"task_result_0": []}, "result_error": []}
                return {"result": {"task_0": []}, "result_error": []}
            raise AssertionError(method)

        async def get_attached_object(self, attached_object_id: int):
            return {
                "ID": attached_object_id,
                "OBJECT_ID": 62357,
                "NAME": "AI-close-303.txt",
                "SIZE": 269,
                "CREATE_TIME": "2026-06-02T11:01:00+03:00",
                "DOWNLOAD_URL": "https://example.test/download/801",
            }

        async def download_file_from_url(self, url: str, destination: Path, *, max_bytes: int):
            data = b"AI task close report\nStatus: unconfirmed\nAI marker: AI_SERVER_TASK_CLOSE_INCOMPLETE\n"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            return len(data)

    stats = anyio_run(sync_portal_index(ReportAttachmentBitrix(), index, settings=get_settings()))

    assert stats.tasks == 1
    assert stats.task_attachments == 1
    task = index.get_item(entity_type="task", entity_id=303)
    assert task is not None
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" in task.body
    assert task.metadata["ai_close_incomplete"] is True
    assert task.metadata["ai_close_marker"] == "AI_SERVER_TASK_CLOSE_INCOMPLETE"
    assert task.metadata["ai_close_problem_types"] == ["unconfirmed"]
    assert task.metadata["ai_close_marker_source"] == "task_attachment"
    attachment = index.get_item(entity_type="task_attachment", entity_id=801)
    assert attachment is not None
    assert attachment.metadata["ai_close_report"] is True
    assert attachment.metadata["ai_close_problem_types"] == ["unconfirmed"]
    assert index.search("AI_SERVER_TASK_CLOSE_INCOMPLETE", entity_types={"task"})


def test_portal_metadata_sync_uses_report_content_status_over_legacy_file_name(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    index = FakePortalSearchIndex()

    class ReportAttachmentBitrix(FakePortalBitrix):
        async def list_all_tasks(self, **kwargs):
            tasks = await super().list_all_tasks(**kwargs)
            task = dict(tasks[0])
            task.update(
                {
                    "ID": 303,
                    "TITLE": "Проверить архив",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CLOSED_DATE": "2026-06-02T11:00:00+03:00",
                    "UF_TASK_WEBDAV_FILES": ["n801"],
                }
            )
            return [task]

        async def result(self, method: str, payload: dict):
            self.calls.append((method, payload))
            if method == "batch":
                if "task_result_0" in payload["cmd"]:
                    return {"result": {"task_result_0": []}, "result_error": []}
                return {"result": {"task_0": []}, "result_error": []}
            raise AssertionError(method)

        async def get_attached_object(self, attached_object_id: int):
            return {
                "ID": attached_object_id,
                "OBJECT_ID": 62357,
                "NAME": "AI-close-303-unconfirmed.txt",
                "SIZE": 269,
                "CREATE_TIME": "2026-06-02T11:01:00+03:00",
                "DOWNLOAD_URL": "https://example.test/download/801",
            }

        async def download_file_from_url(self, url: str, destination: Path, *, max_bytes: int):
            data = b"AI task close report\nStatus: ok\n"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            return len(data)

    anyio_run(sync_portal_index(ReportAttachmentBitrix(), index, settings=get_settings()))

    task = index.get_item(entity_type="task", entity_id=303)
    assert task is not None
    assert task.metadata["ai_close_report_files"][0]["status"] == "ok"
    assert task.metadata["ai_close_incomplete"] is False
    assert task.metadata["ai_close_problem_types"] == []
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" not in task.body
    attachment = index.get_item(entity_type="task_attachment", entity_id=801)
    assert attachment is not None
    assert attachment.metadata["ai_close_problem_types"] == []


def test_portal_metadata_sync_alerts_when_ai_close_report_attachment_disappears(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("BITRIX_TASK_CLOSE_REPORT_ADMIN_USER_IDS", "1")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    index = FakePortalSearchIndex()

    class ReportAttachmentBitrix(FakePortalBitrix):
        def __init__(self, *, with_report: bool = True) -> None:
            super().__init__()
            self.with_report = with_report
            self.messages: list[tuple[str, str]] = []

        async def list_all_tasks(self, **kwargs):
            tasks = await super().list_all_tasks(**kwargs)
            task = dict(tasks[0])
            task.update(
                {
                    "ID": 303,
                    "TITLE": "РџСЂРѕРІРµСЂРёС‚СЊ Р°СЂС…РёРІ",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CLOSED_DATE": "2026-06-02T11:00:00+03:00",
                    "UF_TASK_WEBDAV_FILES": ["n801"] if self.with_report else [],
                }
            )
            return [task]

        async def result(self, method: str, payload: dict):
            self.calls.append((method, payload))
            if method == "batch":
                if "task_result_0" in payload["cmd"]:
                    return {"result": {"task_result_0": []}, "result_error": []}
                return {"result": {"task_0": []}, "result_error": []}
            raise AssertionError(method)

        async def get_attached_object(self, attached_object_id: int):
            return {
                "ID": attached_object_id,
                "OBJECT_ID": 62357,
                "NAME": "AI-close-303-unconfirmed.txt",
                "SIZE": 269,
                "CREATE_TIME": "2026-06-02T11:01:00+03:00",
                "DOWNLOAD_URL": "https://example.test/download/801",
            }

        async def send_bot_message(self, dialog_id: str, message: str, *, bot_id=None, keyboard=None):
            self.messages.append((dialog_id, message))
            return {"message_id": 1}

    anyio_run(sync_portal_index(ReportAttachmentBitrix(with_report=True), index, settings=get_settings()))
    bitrix = ReportAttachmentBitrix(with_report=False)
    anyio_run(sync_portal_index(bitrix, index, settings=get_settings()))

    task = index.get_item(entity_type="task", entity_id=303)
    assert task is not None
    assert task.metadata["ai_close_report_missing"] is True
    assert task.metadata["ai_close_report_incident_status"] == "pending"
    assert task.metadata["ai_close_report_missing_files"][0]["name"] == "AI-close-303-unconfirmed.txt"
    assert task.metadata["ai_close_incomplete"] is True
    assert "AI_SERVER_TASK_CLOSE_INCOMPLETE" in task.body
    assert bitrix.messages
    assert bitrix.messages[0][0] == "1"
    assert "AI-close-303-unconfirmed.txt" in bitrix.messages[0][1]
    assert "1 " in bitrix.messages[0][1]
    assert "2 " in bitrix.messages[0][1]


def test_portal_metadata_sync_skips_accepted_missing_ai_close_report(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("BITRIX_TASK_CLOSE_REPORT_ADMIN_USER_IDS", "1")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    index = FakePortalSearchIndex()

    class ReportAttachmentBitrix(FakePortalBitrix):
        def __init__(self, *, with_report: bool = True) -> None:
            super().__init__()
            self.with_report = with_report
            self.messages: list[tuple[str, str]] = []

        async def list_all_tasks(self, **kwargs):
            tasks = await super().list_all_tasks(**kwargs)
            task = dict(tasks[0])
            task.update(
                {
                    "ID": 303,
                    "TITLE": "Проверить архив",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CLOSED_DATE": "2026-06-02T11:00:00+03:00",
                    "UF_TASK_WEBDAV_FILES": ["n801"] if self.with_report else [],
                }
            )
            return [task]

        async def result(self, method: str, payload: dict):
            self.calls.append((method, payload))
            if method == "batch":
                if "task_result_0" in payload["cmd"]:
                    return {"result": {"task_result_0": []}, "result_error": []}
                return {"result": {"task_0": []}, "result_error": []}
            raise AssertionError(method)

        async def get_attached_object(self, attached_object_id: int):
            return {
                "ID": attached_object_id,
                "OBJECT_ID": 62357,
                "NAME": "AI-close-303-unconfirmed.txt",
                "SIZE": 269,
                "CREATE_TIME": "2026-06-02T11:01:00+03:00",
                "DOWNLOAD_URL": "https://example.test/download/801",
            }

        async def send_bot_message(self, dialog_id: str, message: str, *, bot_id=None, keyboard=None):
            self.messages.append((dialog_id, message))
            return {"message_id": 1}

    anyio_run(sync_portal_index(ReportAttachmentBitrix(with_report=True), index, settings=get_settings()))
    task = index.get_item(entity_type="task", entity_id=303)
    assert task is not None
    report_file = task.metadata["ai_close_report_files"][0]
    index.upsert_task_close_processing_state(
        task_id=303,
        state_key=task_close_report_state_key(report_file),
        status="accepted_missing",
        payload={"accepted_file": report_file},
        actor_user_id=1,
    )

    bitrix = ReportAttachmentBitrix(with_report=False)
    anyio_run(sync_portal_index(bitrix, index, settings=get_settings()))

    task = index.get_item(entity_type="task", entity_id=303)
    assert task is not None
    assert bitrix.messages == []
    assert "ai_close_report_missing" not in task.metadata
    assert task.metadata["ai_close_report_files"] == []


def test_portal_metadata_sync_auto_restores_missing_ai_close_report(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    index = FakePortalSearchIndex()

    class ReportAttachmentBitrix(FakePortalBitrix):
        def __init__(self, *, with_report: bool = True) -> None:
            super().__init__()
            self.with_report = with_report
            self.addfile_payloads: list[dict] = []

        async def list_all_tasks(self, **kwargs):
            tasks = await super().list_all_tasks(**kwargs)
            task = dict(tasks[0])
            task.update(
                {
                    "ID": 303,
                    "TITLE": "РџСЂРѕРІРµСЂРёС‚СЊ Р°СЂС…РёРІ",
                    "STATUS": 5,
                    "RESPONSIBLE_ID": 13,
                    "CREATED_BY": 1,
                    "CLOSED_DATE": "2026-06-02T11:00:00+03:00",
                    "UF_TASK_WEBDAV_FILES": ["n801"] if self.with_report else [],
                }
            )
            return [task]

        async def result(self, method: str, payload: dict):
            self.calls.append((method, payload))
            if method == "batch":
                if "task_result_0" in payload["cmd"]:
                    return {"result": {"task_result_0": []}, "result_error": []}
                return {"result": {"task_0": []}, "result_error": []}
            if method == "task.item.addfile":
                self.addfile_payloads.append(payload)
                return {"ATTACHMENT_ID": 901, "FILE_ID": 62357, "NAME": payload["fileParameters"]["NAME"]}
            raise AssertionError(method)

        async def get_attached_object(self, attached_object_id: int):
            return {
                "ID": attached_object_id,
                "OBJECT_ID": 62357,
                "NAME": "AI-close-303-unconfirmed.txt",
                "SIZE": 269,
                "CREATE_TIME": "2026-06-02T11:01:00+03:00",
                "DOWNLOAD_URL": "https://example.test/download/801",
            }

        async def send_bot_message(self, dialog_id: str, message: str, *, bot_id=None, keyboard=None):
            return {"message_id": 1}

        async def get_disk_file_download_url(self, file_id: int) -> str:
            return f"fake://disk/{file_id}"

        async def download_file_from_url(self, url: str, destination: Path, *, max_bytes: int):
            data = b"AI task close report\nStatus: unconfirmed\n"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            return len(data)

    anyio_run(sync_portal_index(ReportAttachmentBitrix(with_report=True), index, settings=get_settings()))
    anyio_run(sync_portal_index(ReportAttachmentBitrix(with_report=False), index, settings=get_settings()))
    task = index.get_item(entity_type="task", entity_id=303)
    assert task is not None
    metadata = dict(task.metadata)
    metadata["ai_close_report_auto_restore_after"] = "2000-01-01T00:00:00+03:00"
    index.update_item_body_metadata(entity_type="task", entity_id=303, body=task.body, metadata=metadata)

    bitrix = ReportAttachmentBitrix(with_report=False)
    anyio_run(sync_portal_index(bitrix, index, settings=get_settings()))

    assert bitrix.addfile_payloads
    assert bitrix.addfile_payloads[0]["fileParameters"]["NAME"] == "AI-close-303-unconfirmed.txt"
    task = index.get_item(entity_type="task", entity_id=303)
    assert task is not None
    assert task.metadata["ai_close_report_missing"] is False
    assert task.metadata["ai_close_report_incident_status"] == "restored"
    assert task.metadata["ai_close_report_files"][0]["attached_object_id"] == 901


def test_portal_metadata_sync_does_not_restore_ai_close_report_when_task_disappears(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("BITRIX_TASK_CLOSE_REPORT_ADMIN_USER_IDS", "1")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    index = FakePortalSearchIndex()
    index.upsert_item(
        entity_type="task",
        entity_id=303,
        title="Deleted task",
        body="",
        metadata={
            "ai_close_report_files": [
                {
                    "attached_object_id": 801,
                    "disk_object_id": 62357,
                    "name": "AI-close-303.txt",
                    "status": "unconfirmed",
                }
            ],
            "ai_close_report_missing": True,
            "ai_close_report_incident_status": "pending",
            "ai_close_report_auto_restore_after": "2000-01-01T00:00:00+03:00",
        },
    )

    class DeletedTaskBitrix(FakePortalBitrix):
        def __init__(self) -> None:
            super().__init__()
            self.addfile_payloads: list[dict] = []
            self.messages: list[tuple[str, str]] = []

        async def list_all_tasks(self, **kwargs):
            return []

        async def result(self, method: str, payload: dict):
            if method == "task.item.addfile":
                self.addfile_payloads.append(payload)
                return {"ATTACHMENT_ID": 901, "FILE_ID": 62357, "NAME": payload["fileParameters"]["NAME"]}
            return await super().result(method, payload)

        async def send_bot_message(self, dialog_id: str, message: str, *, bot_id=None, keyboard=None):
            self.messages.append((dialog_id, message))
            return {"message_id": 1}

    bitrix = DeletedTaskBitrix()
    anyio_run(sync_portal_index(bitrix, index, settings=get_settings()))

    assert bitrix.addfile_payloads == []
    assert bitrix.messages == []


def test_portal_metadata_sync_deduplicates_disk_roots(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    monkeypatch.setenv("SEARCH_INDEX_MAX_DISK_ITEMS", "10")

    class DuplicateRootBitrix(FakePortalBitrix):
        def __init__(self):
            super().__init__()
            self.folder_calls: dict[int, int] = {}

        async def list_disk_storages(self, *, limit: int | None = None):
            storages = [
                {"ID": 10, "ROOT_OBJECT_ID": 500, "NAME": "РћР±С‰РёР№ РґРёСЃРє"},
                {"ID": 11, "ROOT_OBJECT_ID": 500, "NAME": "Р”СѓР±Р»СЊ РѕР±С‰РµРіРѕ РґРёСЃРєР°"},
            ]
            return storages[:limit] if limit else storages

        async def list_disk_folder_children_all(
            self,
            *,
            folder_id: int,
            filter_: dict | None = None,
            limit: int | None = None,
        ):
            self.folder_calls[folder_id] = self.folder_calls.get(folder_id, 0) + 1
            return await super().list_disk_folder_children_all(folder_id=folder_id, filter_=filter_, limit=limit)

    index = FakePortalSearchIndex()
    bitrix = DuplicateRootBitrix()

    stats = anyio_run(sync_portal_index(bitrix, index, settings=get_settings()))

    assert stats.disk_items == 4
    assert bitrix.folder_calls[500] == 1
    assert index.get_item(entity_type="disk_storage", entity_id=10) is not None
    assert index.get_item(entity_type="disk_storage", entity_id=11) is not None
    assert index.get_item(entity_type="disk_file", entity_id=501) is not None
    assert index.get_item(entity_type="disk_folder", entity_id=502) is not None


def test_portal_metadata_sync_continues_after_disk_folder_502(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "asutp-expert.bitrix24.ru")
    monkeypatch.setenv("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", "10")
    monkeypatch.setenv("SEARCH_INDEX_MAX_DISK_ITEMS", "20")

    class OneBrokenDiskBitrix(FakePortalBitrix):
        async def list_disk_storages(self, *, limit: int | None = None):
            storages = [
                {"ID": 20, "ROOT_OBJECT_ID": 600, "NAME": "Broken Disk"},
                {"ID": 10, "ROOT_OBJECT_ID": 500, "NAME": "Shared Disk"},
            ]
            return storages[:limit] if limit else storages

        async def list_disk_folder_children_all(
            self,
            *,
            folder_id: int,
            filter_: dict | None = None,
            limit: int | None = None,
        ):
            if folder_id == 600:
                raise BitrixApiError("disk.folder.getchildren", "HTTP_502", "Bad Gateway")
            return await super().list_disk_folder_children_all(folder_id=folder_id, filter_=filter_, limit=limit)

    index = FakePortalSearchIndex()
    index.upsert_item(entity_type="disk_file", entity_id=999, title="Old file", metadata={"parent_id": 600})

    stats = anyio_run(sync_portal_index(OneBrokenDiskBitrix(), index, settings=get_settings()))

    assert stats.disk_items == 4
    assert stats.errors == [
        "disk: storage 20 root folder 600: Bitrix REST error in disk.folder.getchildren: HTTP_502 Bad Gateway"
    ]
    assert stats.prune_skipped == ["disk: incomplete after API errors"]
    assert index.get_item(entity_type="disk_storage", entity_id=20) is not None
    assert index.get_item(entity_type="disk_storage", entity_id=10) is not None
    assert index.get_item(entity_type="disk_file", entity_id=501) is not None
    assert index.get_item(entity_type="disk_file", entity_id=999) is not None


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


def test_search_webhook_indexer_refreshes_task_and_comments_immediately(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("SEARCH_WEBHOOK_INDEXER_ENABLED", "true")
    settings = get_settings()
    index = FakePortalSearchIndex()

    class EventBitrix(FakePortalBitrix):
        async def get_task(self, task_id: int, *, select=None):
            return {
                "task": {
                    "id": str(task_id),
                    "title": "Проверить договор",
                    "description": "Новая редакция",
                    "status": "2",
                    "responsibleId": "9",
                    "createdBy": "1",
                    "changedDate": "2026-07-22T10:00:00+03:00",
                }
            }

        async def result(self, method: str, payload: dict):
            if method == "task.commentitem.getlist":
                return [{"POST_MESSAGE": "Добавлен свежий комментарий"}]
            if method == "tasks.task.result.list":
                return []
            return await super().result(method, payload)

    job, prepared = prepare_search_webhook_job(
        {"event": "ONTASKCOMMENTADD", "data": {"FIELDS_AFTER": {"TASK_ID": "202"}}},
        settings=settings,
    )
    assert job is not None
    assert prepared["entity_type"] == "task"

    result = anyio_run(process_search_webhook_job(EventBitrix(), index, job, status={}, settings=settings))

    item = index.get_item(entity_type="task", entity_id=202)
    assert result["reason"] == "task_indexed"
    assert item is not None
    assert "свежий комментарий" in item.body.casefold()
    assert item.metadata["event_synced_at"]

    missing_task_job, rejected = prepare_search_webhook_job(
        {"event": "ONTASKCOMMENTADD", "data": {"FIELDS_AFTER": {"ID": "999"}}},
        settings=settings,
    )
    assert missing_task_job is None
    assert rejected["reason"] == "task_id_not_found"


def test_search_webhook_indexer_refreshes_catalog_product_metadata(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("SEARCH_WEBHOOK_INDEXER_ENABLED", "true")
    settings = get_settings()
    index = FakePortalSearchIndex()

    class ProductEventBitrix(FakePortalBitrix):
        async def result(self, method: str, payload: dict):
            if method == "catalog.product.get":
                return {"product": {"id": 1001, "iblockId": 7, "name": "Амортизатор", "previewText": "Газовый"}}
            return await super().result(method, payload)

    job, prepared = prepare_search_webhook_job(
        {"event": "CATALOG.PRODUCT.ON.UPDATE", "data": {"FIELDS_AFTER": {"ID": "1001"}}},
        settings=settings,
    )
    assert job is not None
    assert prepared["entity_type"] == "catalog_product"

    result = anyio_run(process_search_webhook_job(ProductEventBitrix(), index, job, status={}, settings=settings))

    item = index.get_item(entity_type="catalog_product", entity_id=1001)
    assert result["reason"] == "catalog_product_indexed"
    assert item is not None
    assert item.title == "Амортизатор"
    assert item.metadata["event_synced_at"]


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
            if "task_result_0" in payload["cmd"]:
                assert "taskId=202" in payload["cmd"]["task_result_0"]
                return {
                    "result": {
                        "task_result_0": [
                            {
                                "text": (
                                    "Метка: AI_SERVER_TASK_CLOSE_INCOMPLETE\n"
                                    "Невыполненные пункты:\n"
                                    "- не проверен архив\n"
                                    "Неподтверждённые пункты:\n"
                                    "- нет фото результата"
                                )
                            }
                        ]
                    },
                    "result_error": [],
                }
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
        if method == "tasks.task.result.list":
            raise AssertionError("task result sync should use batch")
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
