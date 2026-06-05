import json
from pathlib import Path

from ai_server.agents.pto import PtoSpecialist
from ai_server.agents.pto_llm import PtoLLMToolCall
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.document_access import DocumentToolset
from tests.fakes import FakeEmbeddingProvider, FakePtoLLM


def test_document_toolset_compares_spreadsheets(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    index = _document_index(tmp_path / "search_index.sqlite")
    toolset = DocumentToolset(client=FakeDocumentBitrix(), portal_search=index, user_id=9)

    result = anyio_run(
        toolset.spreadsheet_compare(
            {
                "first_query": "смета январь",
                "second_query": "смета февраль",
                "limit": 10,
            }
        )
    )

    assert result.status == "ok"
    report = result.data["report"]
    assert report["common_rows"] == 2
    assert report["changed"][0]["key"] == "Кабель UTP"
    assert report["changed"][0]["fields"][0]["field"] == "стоимость"
    assert "Отличий по значениям: 1" in result.data["summary"]


def test_pto_specialist_uses_spreadsheet_compare_tool(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    manifest = get_agent_manifest("pto")
    assert manifest is not None
    index = _document_index(tmp_path / "search_index.sqlite")
    specialist = PtoSpecialist(
        manifest,
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        tools=DocumentToolset(client=FakeDocumentBitrix(), portal_search=index, user_id=9),
        llm=FakePtoLLM(
            tool_calls=[
                PtoLLMToolCall(
                    name="spreadsheet_compare",
                    args={
                        "first_query": "смета январь",
                        "second_query": "смета февраль",
                        "limit": 10,
                    },
                )
            ],
            final_answer="В сметах изменился кабель.",
        ),
    )

    result = anyio_run(
        specialist.handle(AgentTask(task_id="pto-1", request="Сравни сметы за январь и февраль"))
    )

    action = next(item for item in result.actions_taken if item.name == "pto_spreadsheet_compare")
    assert action.status == "ok"
    assert result.answer == "В сметах изменился кабель."
    assert result.handoff_to == []


class FakeDocumentBitrix:
    async def get_disk_file_download_url(self, file_id: int):
        return f"fake://disk/{file_id}"

    async def download_file_from_url(self, url: str, destination: Path, *, max_bytes: int):
        file_id = int(url.rsplit("/", 1)[-1])
        if file_id == 1001:
            data = "Наименование;Количество;Стоимость\nКабель UTP;10;1000\nКамера;2;5000\n"
        else:
            data = "Наименование;Количество;Стоимость\nКабель UTP;10;1200\nКамера;2;5000\n"
        encoded = data.encode("utf-8")
        assert len(encoded) <= max_bytes
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
        return len(encoded)


def _document_index(path: Path) -> PortalSearchIndex:
    index = PortalSearchIndex(path)
    index.ensure_schema()
    _insert_document(index, entity_id="1001", title="Смета январь.csv", body="смета январь кабель камера")
    _insert_document(index, entity_id="1002", title="Смета февраль.csv", body="смета февраль кабель камера")
    return index


def _insert_document(index: PortalSearchIndex, *, entity_id: str, title: str, body: str) -> None:
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
                entity_id,
                title,
                body,
                f"https://example.test/docs/{entity_id}",
                f"disk_file {entity_id} {title} {body}",
                json.dumps({"disk_object_id": int(entity_id), "path": "/ПТО/Сметы"}, ensure_ascii=False),
                "2026-06-01T10:00:00+03:00",
                "2026-06-01T10:00:00+03:00",
                "2026-06-01T10:00:00+03:00",
            ),
        )


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)
