import asyncio

from ai_server.models import AgentTask
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import load_agent_manifests
from ai_server.retrieval import HybridKnowledgeRetriever
from tests.fakes import FakeBitrixLLM, FakeEmbeddingProvider, FakeInternalOrchestratorLLM, FakePtoLLM


def test_internal_orchestrator_delegates_bitrix_request():
    result = asyncio.run(
        InternalOrchestrator(
            load_agent_manifests(),
            bitrix_retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            bitrix_llm=FakeBitrixLLM(),
            orchestrator_llm=FakeInternalOrchestratorLLM(handoff_to=["bitrix24"]),
        ).handle(AgentTask(task_id="t1", request="Покажи задачи в Битриксе"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["bitrix24"]
    assert result.actions_taken[0].name == "orchestrator_llm_route"
    assert result.actions_taken[1].name == "delegate_to_specialist"


def test_internal_orchestrator_delegates_pto_document_request():
    result = asyncio.run(
        InternalOrchestrator(
            load_agent_manifests(),
            pto_retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            pto_llm=FakePtoLLM(final_answer="ПТО проверил документы."),
            orchestrator_llm=FakeInternalOrchestratorLLM(handoff_to=["pto"]),
        ).handle(AgentTask(task_id="t1", request="Сравни две сметы по объекту"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["pto"]
    assert result.answer == "ПТО проверил документы."
    assert result.actions_taken[1].details["specialist"] == "pto"


def test_internal_orchestrator_reports_configured_model(monkeypatch):
    result = asyncio.run(
        InternalOrchestrator(
            load_agent_manifests(),
            orchestrator_llm=FakeInternalOrchestratorLLM(
                answer="LLM-контур: provider deepseek, model deepseek-v4-flash."
            ),
        ).handle(
            AgentTask(task_id="t1", request="Какая ты модель?")
        )
    )

    assert result.agent_id == "internal_orchestrator"
    assert "deepseek-v4-flash" in result.answer
    assert result.actions_taken[0].name == "orchestrator_llm_route"

