from pathlib import Path

from ai_server.agents.pto import PtoLLMToolCall, PtoSpecialist
from ai_server.agents.pto.tools import (
    DocumentDraftCreateTool,
    DocumentDraftListTool,
    DocumentReadTool,
    SpreadsheetCompareTool,
    SpreadsheetPreviewTool,
)
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
from tests.fakes import FakeEmbeddingProvider, FakePtoLLM


def test_spreadsheet_compare_requires_llm_selected_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    compare_tool = SpreadsheetCompareTool(FakeDocumentBitrix(), settings=get_settings())

    result = anyio_run(
        compare_tool.execute(
            {
                "first_query": "смета январь",
                "second_query": "смета февраль",
                "limit": 10,
            },
            user_id=9,
        )
    )

    assert result.status == "contract_violation"
    assert "spreadsheet_preview" in (result.error or "")


def test_document_toolset_creates_local_draft(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    draft_tool = DocumentDraftCreateTool(get_settings())

    result = anyio_run(
        draft_tool.execute(
            {
                "title": "../Акт проверки",
                "content": "Замечаний по комплектности нет.",
                "extension": ".md",
            }
        )
    )

    assert result.status == "ok"
    path = Path(result.data["path"])
    assert path.exists()
    assert path.parent == tmp_path / "var" / "document_drafts"
    assert ".." not in path.name
    assert "Замечаний" in path.read_text(encoding="utf-8")


def test_pto_specialist_creates_document_draft(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    manifest = get_agent_manifest("pto")
    assert manifest is not None
    settings = get_settings()
    client = FakeDocumentBitrix()
    specialist = PtoSpecialist(
        manifest,
        retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        agent_tools=[
            SpreadsheetPreviewTool(client, settings=settings),
            SpreadsheetCompareTool(client, settings=settings),
            DocumentDraftCreateTool(settings),
            DocumentDraftListTool(settings),
            DocumentReadTool(client, settings=settings),
        ],
        llm=FakePtoLLM(
            tool_call_steps=[
                [
                    PtoLLMToolCall(
                        name="document_draft_create",
                        args={
                            "title": "Акт проверки",
                            "content": "Черновик акта подготовлен ПТО-специалистом.",
                            "extension": ".md",
                        },
                    )
                ],
                [PtoLLMToolCall(name="none")],
            ],
            final_answer="Черновик подготовлен.",
        ),
    )

    result = anyio_run(specialist.handle(AgentTask(task_id="pto-2", request="Подготовь акт проверки")))

    action = next(item for item in result.actions_taken if item.name == "document_draft_create")
    assert action.status == "ok"
    assert Path(action.details["data"]["path"]).exists()
    assert result.answer == "Черновик подготовлен."


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


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)
