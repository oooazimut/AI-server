import asyncio

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.logistics import LogisticsSpecialist
from ai_server.agents.pto import PtoSpecialist
from ai_server.models import AgentManifest, AgentResult, AgentTask, ModelUsageRecord
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.orchestrators.tools import CallSpecialistTool
from ai_server.registry import load_agent_manifests
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.specialists import manifest_by_id
from tests.fakes import (
    FakeBitrixLLM,
    FakeEmbeddingProvider,
    FakeInternalOrchestratorLLM,
    FakeLogisticsLLM,
    FakeOrchestratorStore,
    FakePtoLLM,
)


def _orch_manifest() -> AgentManifest:
    return AgentManifest(
        id="internal_orchestrator",
        name="Переговорщик",
        kind="orchestrator",
        description="test",
    )


def _make_orch(specialists: dict, llm, *, store=None) -> InternalOrchestrator:
    manifests = load_agent_manifests()
    call_tool = CallSpecialistTool(specialists, manifests, store=store)
    orch = InternalOrchestrator(
        _orch_manifest(),
        agent_tools=[call_tool],
        llm=llm,
        store=store,
    )
    call_tool.schedule_fn = orch._apply_scheduled_tasks_from_specialist
    return orch


def test_internal_orchestrator_delegates_bitrix_request():
    manifests = load_agent_manifests()
    specialists = {
        "bitrix24": Bitrix24Specialist(
            manifest_by_id(manifests, "bitrix24"),
            retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            llm=FakeBitrixLLM(),
        )
    }
    result = asyncio.run(
        _make_orch(specialists, FakeInternalOrchestratorLLM(call_specialists=["bitrix24"])).handle(
            AgentTask(task_id="t1", request="Покажи задачи в Битриксе")
        )
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["bitrix24"]
    # actions: load_context, llm_decision, call_specialist, llm_final_answer
    assert result.actions_taken[1].name == "orchestrator_llm_decision"
    assert result.actions_taken[2].name == "call_specialist"


def test_internal_orchestrator_fast_returns_terminal_specialist_answer():
    class _TerminalSpecialist:
        async def handle(self, task):
            return AgentResult(
                status="completed",
                agent_id="bitrix24",
                answer="terminal answer",
                confidence=0.9,
                model_usage=[
                    ModelUsageRecord(
                        agent_id="bitrix24",
                        provider="deepseek",
                        model="deepseek-v4-flash",
                        input_tokens=7154,
                        output_tokens=195,
                    )
                ],
                metadata={
                    "terminal": True,
                    "answer_is_final": True,
                    "safe_to_send": True,
                    "fast_return": True,
                    "fast_return_reason": "read_only_tool_success",
                    "terminal_tool": "bitrix_my_tasks",
                },
            )

    fake_llm = FakeInternalOrchestratorLLM(call_specialists=["bitrix24"], status="needs_clarification")
    result = asyncio.run(
        _make_orch({"bitrix24": _TerminalSpecialist()}, fake_llm).handle(
            AgentTask(task_id="t1", request="Bitrix show my tasks")
        )
    )

    assert result.status == "completed"
    assert result.answer == "terminal answer"
    assert len(fake_llm.decide_calls) == 1
    assert result.metadata["fast_return"] is True
    assert result.metadata["terminal_tool"] == "call_specialist"
    assert result.metadata["specialist_terminal_tool"] == "bitrix_my_tasks"
    assert any(action.name == "orchestrator_fast_return" for action in result.actions_taken)
    assert any(
        usage.agent_id == "bitrix24" and usage.input_tokens == 7154 and usage.output_tokens == 195
        for usage in result.model_usage
    )


def test_internal_orchestrator_stops_after_plain_specialist_clarification():
    class _ClarifyingSpecialist:
        async def handle(self, task):
            return AgentResult(
                status="needs_clarification",
                agent_id="bitrix24",
                answer="Уточните запрос.",
                confidence=0.8,
            )

    fake_llm = FakeInternalOrchestratorLLM(call_specialists=["bitrix24"])
    result = asyncio.run(
        _make_orch({"bitrix24": _ClarifyingSpecialist()}, fake_llm).handle(
            AgentTask(task_id="t1", request="Битрикс покажи договоры")
        )
    )

    assert result.status == "needs_clarification"
    assert result.answer == "Уточните запрос."
    assert len(fake_llm.decide_calls) == 1
    assert result.metadata["fast_return_reason"] == "specialist_answer_terminal"
    assert result.metadata["terminal_status"] == "needs_clarification"


def test_internal_orchestrator_delegates_pto_document_request():
    manifests = load_agent_manifests()
    specialists = {
        "pto": PtoSpecialist(
            manifest_by_id(manifests, "pto"),
            retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            llm=FakePtoLLM(final_answer="ПТО проверил документы."),
        )
    }
    result = asyncio.run(
        _make_orch(specialists, FakeInternalOrchestratorLLM(call_specialists=["pto"])).handle(
            AgentTask(task_id="t1", request="Сравни две сметы по объекту")
        )
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["pto"]
    assert result.answer == "ПТО проверил документы."
    assert result.actions_taken[2].details["data"]["specialist"] == "pto"


def test_internal_orchestrator_reports_configured_model():
    manifests = load_agent_manifests()
    specialists = {
        "bitrix24": Bitrix24Specialist(
            manifest_by_id(manifests, "bitrix24"),
            retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            llm=FakeBitrixLLM(),
        )
    }
    result = asyncio.run(
        _make_orch(
            specialists,
            FakeInternalOrchestratorLLM(answer="LLM-контур: provider deepseek, model deepseek-v4-flash."),
        ).handle(AgentTask(task_id="t1", request="Какая ты модель?"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert "deepseek-v4-flash" in result.answer
    assert result.actions_taken[1].name == "orchestrator_llm_decision"


def test_pending_specialist_set_when_needs_clarification():
    """После вызова специалиста: если вернул needs_clarification, pending_specialist записывается в KV."""
    manifests = load_agent_manifests()
    store = FakeOrchestratorStore()
    specialists = {
        "bitrix24": Bitrix24Specialist(
            manifest_by_id(manifests, "bitrix24"),
            retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            llm=FakeBitrixLLM(final_status="needs_clarification", final_answer="Укажите срок"),
        )
    }
    asyncio.run(
        _make_orch(
            specialists,
            FakeInternalOrchestratorLLM(call_specialists=["bitrix24"]),
            store=store,
        ).handle(AgentTask(task_id="t1", request="Создай задачу", context={"dialog_key": "dlg1"}))
    )

    assert store._kv.get(("dlg1", "pending_specialist")) == "bitrix24"


def test_pending_specialist_injected_into_decide_context():
    """Если pending_specialist выставлен в KV — он инжектируется в task.context для LLM decide."""
    manifests = load_agent_manifests()
    store = FakeOrchestratorStore()
    store.set_pending("dlg1", "bitrix24")
    fake_llm = FakeInternalOrchestratorLLM(call_specialists=["bitrix24"])

    specialists = {
        "bitrix24": Bitrix24Specialist(
            manifest_by_id(manifests, "bitrix24"),
            retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            llm=FakeBitrixLLM(final_status="completed", final_answer="Задача создана."),
        )
    }
    asyncio.run(
        _make_orch(specialists, fake_llm, store=store).handle(
            AgentTask(task_id="t2", request="три дня", context={"dialog_key": "dlg1"})
        )
    )

    assert len(fake_llm.decide_calls) >= 1
    assert fake_llm.decide_calls[0]["task"].context.get("pending_specialist") == "bitrix24"


def test_pending_specialist_cleared_after_completed():
    """После вызова специалиста: если вернул completed — pending_specialist удаляется."""
    manifests = load_agent_manifests()
    store = FakeOrchestratorStore()
    store.set_pending("dlg1", "bitrix24")

    specialists = {
        "bitrix24": Bitrix24Specialist(
            manifest_by_id(manifests, "bitrix24"),
            retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            llm=FakeBitrixLLM(final_status="completed", final_answer="Задача создана."),
        )
    }
    result = asyncio.run(
        _make_orch(
            specialists,
            FakeInternalOrchestratorLLM(call_specialists=["bitrix24"]),
            store=store,
        ).handle(AgentTask(task_id="t3", request="три дня", context={"dialog_key": "dlg1"}))
    )

    assert store._kv.get(("dlg1", "pending_specialist")) is None
    assert result.handoff_to == ["bitrix24"]


def test_pending_specialist_stays_if_still_needs_clarification():
    """После вызова специалиста: если снова needs_clarification — pending_specialist остаётся."""
    manifests = load_agent_manifests()
    store = FakeOrchestratorStore()
    store.set_pending("dlg1", "bitrix24")

    specialists = {
        "bitrix24": Bitrix24Specialist(
            manifest_by_id(manifests, "bitrix24"),
            retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            llm=FakeBitrixLLM(final_status="needs_clarification", final_answer="Укажите исполнителя"),
        )
    }
    asyncio.run(
        _make_orch(
            specialists,
            FakeInternalOrchestratorLLM(call_specialists=["bitrix24"]),
            store=store,
        ).handle(AgentTask(task_id="t4", request="три дня", context={"dialog_key": "dlg1"}))
    )

    assert store._kv.get(("dlg1", "pending_specialist")) == "bitrix24"


def test_internal_orchestrator_delegates_logistics_request():
    manifests = load_agent_manifests()
    specialists = {
        "logistics": LogisticsSpecialist(
            manifest_by_id(manifests, "logistics"),
            retriever=HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider()),
            llm=FakeLogisticsLLM(final_answer="Логист обработал отчет."),
        )
    }
    result = asyncio.run(
        _make_orch(specialists, FakeInternalOrchestratorLLM(call_specialists=["logistics"])).handle(
            AgentTask(task_id="t1", request="Утренний отчет по машинам")
        )
    )

    assert result.agent_id == "internal_orchestrator"
    assert result.handoff_to == ["logistics"]
    assert result.answer == "Логист обработал отчет."
    assert result.actions_taken[2].details["data"]["specialist"] == "logistics"
