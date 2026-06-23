import asyncio

from ai_server.agents.pto import PtoLLMToolCall, PtoSpecialist
from ai_server.models import AgentTask
from ai_server.registry import get_agent_manifest
from ai_server.retrieval import HybridKnowledgeRetriever
from tests.fakes import FakeEmbeddingProvider, FakePtoLLM


def _pto_specialist(*, llm=None) -> PtoSpecialist:
    manifest = get_agent_manifest("pto")
    retriever = HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())
    return PtoSpecialist(
        manifest,
        retriever=retriever,
        agent_tools=[],
        llm=llm or FakePtoLLM(),
    )


def test_pto_specialist_loads_skills_and_completes_with_no_tool_calls():
    result = asyncio.run(
        _pto_specialist().handle(AgentTask(task_id="t1", request="Найди техническую документацию по объекту"))
    )

    assert result.status == "completed"
    assert result.actions_taken[0].name == "load_pto_specialist_context"
    assert result.actions_taken[1].name == "pto_llm_decision"
    assert result.actions_taken[-1].name == "pto_llm_final_answer"


def test_pto_specialist_forwards_dialog_history_to_decide():
    llm = FakePtoLLM()
    history = [
        {"role": "user", "content": "сравни сметы по объекту"},
        {"role": "assistant", "content": "Уточните, по какому объекту?"},
    ]

    asyncio.run(
        _pto_specialist(llm=llm).handle(
            AgentTask(task_id="t1", request="по Транзит-Экспресс", context={"dialog_history": history})
        )
    )

    assert llm.decide_calls[0]["dialog_history"] == history


def test_pto_specialist_defaults_dialog_history_to_none_when_absent():
    llm = FakePtoLLM()

    asyncio.run(_pto_specialist(llm=llm).handle(AgentTask(task_id="t1", request="привет")))

    assert llm.decide_calls[0]["dialog_history"] is None


def test_pto_specialist_emits_guardrail_when_loop_hits_max_steps():
    llm = FakePtoLLM(
        tool_call_steps=[
            [PtoLLMToolCall(name="portal_document_search", args={"query": f"step{i}"})] for i in range(1, 6)
        ]
    )

    result = asyncio.run(_pto_specialist(llm=llm).handle(AgentTask(task_id="t1", request="найди все версии акта")))

    assert len(llm.decide_calls) == 5
    assert "pto_tool_loop_guardrail" in {action.name for action in result.actions_taken}
