import asyncio

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.logistics import LogisticsSpecialist
from ai_server.agents.pto import PtoSpecialist
from ai_server.models import AgentTask
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import load_agent_manifests
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
from ai_server.specialists import manifest_by_id
from tests.fakes import FakeBitrixLLM, FakeEmbeddingProvider, FakeInternalOrchestratorLLM, FakeLogisticsLLM, FakePtoLLM


def test_internal_orchestrator_delegates_bitrix_request():
    manifests = load_agent_manifests()
    result = asyncio.run(
        InternalOrchestrator(
            manifests,
            specialists={
                "bitrix24": Bitrix24Specialist(
                    manifest_by_id(manifests, "bitrix24"),
                    retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
                    llm=FakeBitrixLLM(),
                ),
            },
            orchestrator_llm=FakeInternalOrchestratorLLM(handoff_to=["bitrix24"]),
        ).handle(AgentTask(task_id="t1", request="Покажи задачи в Битриксе"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["bitrix24"]
    assert result.actions_taken[0].name == "orchestrator_llm_route"
    assert result.actions_taken[1].name == "delegate_to_specialist"


def test_internal_orchestrator_delegates_pto_document_request():
    manifests = load_agent_manifests()
    result = asyncio.run(
        InternalOrchestrator(
            manifests,
            specialists={
                "pto": PtoSpecialist(
                    manifest_by_id(manifests, "pto"),
                    retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
                    llm=FakePtoLLM(final_answer="ПТО проверил документы."),
                ),
            },
            orchestrator_llm=FakeInternalOrchestratorLLM(handoff_to=["pto"]),
        ).handle(AgentTask(task_id="t1", request="Сравни две сметы по объекту"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["pto"]
    assert result.answer == "ПТО проверил документы."
    assert result.actions_taken[1].details["specialist"] == "pto"


def test_internal_orchestrator_reports_configured_model(monkeypatch):
    manifests = load_agent_manifests()
    result = asyncio.run(
        InternalOrchestrator(
            manifests,
            specialists={
                "bitrix24": Bitrix24Specialist(
                    manifest_by_id(manifests, "bitrix24"),
                    retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
                    llm=FakeBitrixLLM(),
                ),
            },
            orchestrator_llm=FakeInternalOrchestratorLLM(
                answer="LLM-контур: provider deepseek, model deepseek-v4-flash."
            ),
        ).handle(AgentTask(task_id="t1", request="Какая ты модель?"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert "deepseek-v4-flash" in result.answer
    assert result.actions_taken[0].name == "orchestrator_llm_route"


def test_internal_orchestrator_delegates_logistics_request(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    manifests = load_agent_manifests()
    result = asyncio.run(
        InternalOrchestrator(
            manifests,
            specialists={
                "logistics": LogisticsSpecialist(
                    manifest_by_id(manifests, "logistics"),
                    retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
                    llm=FakeLogisticsLLM(final_answer="Логист обработал отчет."),
                    settings=get_settings(),
                ),
            },
            orchestrator_llm=FakeInternalOrchestratorLLM(handoff_to=["logistics"]),
        ).handle(AgentTask(task_id="t1", request="Утренний отчет по машинам"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["logistics"]
    assert result.answer == "Логист обработал отчет."
    assert result.actions_taken[1].details["specialist"] == "logistics"
