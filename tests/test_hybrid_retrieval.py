import asyncio

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from tests.fakes import FakeEmbeddingProvider


def _retriever() -> HybridKnowledgeRetriever:
    return HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())


def test_bitrix_executor_has_no_semantic_knowledge_retrieval():
    manifest = get_agent_manifest("bitrix24")

    hits = _retriever().search(
        manifest,
        "tasks.task.add TITLE RESPONSIBLE_ID создание задачи",
        limit=5,
    )

    assert hits == []


def test_bitrix_specialist_rejects_free_text_without_retrieval():
    result = asyncio.run(
        Bitrix24Specialist(get_agent_manifest("bitrix24")).handle(
            AgentTask(task_id="t1", request="free text must not reach Bitrix reasoning")
        )
    )
    assert result.status == "failed"
    assert result.metadata["reason"] == "ORCHESTRATOR_STRUCTURED_COMMAND_REQUIRED"
