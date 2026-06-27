import asyncio
import json

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.logistics import LogisticsSpecialist
from ai_server.agents.pto import PtoSpecialist
from ai_server.agents.secure_org_data import (
    SecureOrgDataAgent,
    SecureOrgDataLLMDecision,
    SecureOrgDataLLMDecisionResult,
    SecureOrgDataLLMFinalResult,
    SecureOrgDataLLMToolCall,
    SecureOrgDataStore,
)
from ai_server.models import AgentTask, ModelUsageRecord
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import load_agent_manifests
from ai_server.retrieval import HybridKnowledgeRetriever
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
            orchestrator_llm=FakeInternalOrchestratorLLM(call_specialists=["bitrix24"]),
        ).handle(AgentTask(task_id="t1", request="Покажи задачи в Битриксе"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["bitrix24"]
    assert result.actions_taken[0].name == "orchestrator_llm_decision"
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
            orchestrator_llm=FakeInternalOrchestratorLLM(call_specialists=["pto"]),
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
    assert result.actions_taken[0].name == "orchestrator_llm_decision"


def test_internal_orchestrator_delegates_logistics_request():
    manifests = load_agent_manifests()
    result = asyncio.run(
        InternalOrchestrator(
            manifests,
            specialists={
                "logistics": LogisticsSpecialist(
                    manifest_by_id(manifests, "logistics"),
                    retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
                    llm=FakeLogisticsLLM(final_answer="Логист обработал отчет."),
                ),
            },
            orchestrator_llm=FakeInternalOrchestratorLLM(call_specialists=["logistics"]),
        ).handle(AgentTask(task_id="t1", request="Утренний отчет по машинам"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["logistics"]
    assert result.answer == "Логист обработал отчет."
    assert result.actions_taken[1].details["specialist"] == "logistics"


def test_internal_orchestrator_delegates_secure_org_data_open_search(tmp_path):
    metadata_dir = tmp_path / "kb_data"
    index_dir = metadata_dir / "content_index"
    index_dir.mkdir(parents=True)
    (index_dir / "stage1_open_chunks.jsonl").write_text(
        json.dumps(
            {
                "relativePath": "office/router.docx",
                "name": "Инструкция TL-WR820N по настройке роутеров wi-fi.docx",
                "access": "internal",
                "text": "Настройка wi-fi роутеров TL-WR820N. IP меняется на 192.168.0.254.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (index_dir / "stage1_protected_chunks.jsonl").write_text("", encoding="utf-8")
    manifests = load_agent_manifests()
    result = asyncio.run(
        InternalOrchestrator(
            manifests,
            specialists={
                "secure_org_data": SecureOrgDataAgent(
                    manifest_by_id(manifests, "secure_org_data"),
                    store=SecureOrgDataStore(metadata_dir=metadata_dir),
                    llm=_FakeSecureOrgDataLLM(),
                ),
            },
            orchestrator_llm=FakeInternalOrchestratorLLM(call_specialists=["secure_org_data"]),
        ).handle(AgentTask(task_id="t1", request="Найди инструкцию TL-WR820N"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["secure_org_data"]
    assert result.answer == "Нашел: Инструкция TL-WR820N по настройке роутеров wi-fi.docx"
    assert result.actions_taken[1].name == "delegate_to_specialist"
    assert result.actions_taken[1].details["specialist"] == "secure_org_data"


class _FakeSecureOrgDataLLM:
    def __init__(self) -> None:
        self.decide_calls = []
        self.compose_calls = []

    async def decide(self, **kwargs):
        self.decide_calls.append(kwargs)
        if len(self.decide_calls) == 1:
            tool_calls = [
                SecureOrgDataLLMToolCall(
                    name="search_org_data",
                    args={"query": "TL-WR820N", "limit": 3, "include_paths": True},
                    summary="Поиск открытой инструкции",
                )
            ]
        else:
            tool_calls = [SecureOrgDataLLMToolCall(name="none")]
        return SecureOrgDataLLMDecisionResult(
            decision=SecureOrgDataLLMDecision(
                status="completed",
                answer="",
                confidence=0.9,
                tool_calls=tool_calls,
            ),
            model_usage=_fake_secure_usage(),
        )

    async def compose(self, **kwargs):
        self.compose_calls.append(kwargs)
        results = []
        for tool_result in kwargs.get("tool_results") or []:
            results.extend((tool_result.data or {}).get("results") or [])
        title = results[0]["title"] if results else "ничего не найдено"
        return SecureOrgDataLLMFinalResult(
            status="completed",
            answer=f"Нашел: {title}",
            model_usage=_fake_secure_usage(),
        )


def _fake_secure_usage() -> ModelUsageRecord:
    return ModelUsageRecord(
        agent_id="secure_org_data",
        provider="fake",
        model="fake-secure-org-data",
        status="used",
    )
