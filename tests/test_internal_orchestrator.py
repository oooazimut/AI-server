import asyncio

from ai_server.models import AgentTask
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import load_agent_manifests
from ai_server.retrieval import HybridKnowledgeRetriever
from tests.fakes import FakeEmbeddingProvider


def test_internal_orchestrator_delegates_bitrix_request():
    result = asyncio.run(
        InternalOrchestrator(
            load_agent_manifests(),
            bitrix_retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
        ).handle(AgentTask(task_id="t1", request="Покажи задачи в Битриксе"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["bitrix24"]
    assert result.actions_taken[0].name == "delegate_to_specialist"

