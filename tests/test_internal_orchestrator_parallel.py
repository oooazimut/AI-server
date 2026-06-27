import asyncio

import ai_server.specialists as _specialists_module
from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.logistics import LogisticsSpecialist
from ai_server.agents.pto import PtoSpecialist
from ai_server.models import AgentManifest, AgentResult, AgentTask
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
    FakePtoLLM,
)


def _make_fake_specialist_cls():
    class _FakeCls:
        @classmethod
        def build(cls, manifest, **deps):
            inst = cls()
            inst.manifest = manifest
            return inst

        async def handle(self, task):
            return AgentResult(
                status="completed",
                agent_id="fake",
                answer="fake answer",
                confidence=0.5,
            )

    return _FakeCls


def _make_retriever():
    return HybridKnowledgeRetriever(embedding_provider=FakeEmbeddingProvider())


def _bitrix_specialist(manifests, **kwargs):
    return Bitrix24Specialist(
        manifest_by_id(manifests, "bitrix24"),
        retriever=_make_retriever(),
        llm=FakeBitrixLLM(**kwargs),
    )


def _pto_specialist(manifests, **kwargs):
    return PtoSpecialist(
        manifest_by_id(manifests, "pto"),
        retriever=_make_retriever(),
        llm=FakePtoLLM(**kwargs),
    )


def _logistics_specialist(manifests, **kwargs):
    return LogisticsSpecialist(
        manifest_by_id(manifests, "logistics"),
        retriever=_make_retriever(),
        llm=FakeLogisticsLLM(**kwargs),
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


# Вызов двух специалистов за один turn
def test_orchestrator_executes_two_specialists_in_parallel():
    manifests = load_agent_manifests()
    fake_llm = FakeInternalOrchestratorLLM(
        call_specialists=["bitrix24", "pto"],
        synthesized_answer="Оба специалиста ответили.",
    )
    result = asyncio.run(
        _make_orch(
            {
                "bitrix24": _bitrix_specialist(manifests, final_answer="Битрикс готово."),
                "pto": _pto_specialist(manifests, final_answer="ПТО готово."),
            },
            fake_llm,
        ).handle(AgentTask(task_id="t1", request="Нужно и задачу создать, и документ проверить"))
    )

    assert result.agent_id == "internal_orchestrator"
    assert set(result.handoff_to) == {"bitrix24", "pto"}
    assert result.answer == "Оба специалиста ответили."
    assert len(fake_llm.compose_calls) == 1
    call_actions = [a for a in result.actions_taken if a.name == "call_specialist"]
    assert len(call_actions) == 2


def test_orchestrator_compose_action_recorded():
    manifests = load_agent_manifests()
    result = asyncio.run(
        _make_orch(
            {
                "bitrix24": _bitrix_specialist(manifests),
                "pto": _pto_specialist(manifests),
            },
            FakeInternalOrchestratorLLM(call_specialists=["bitrix24", "pto"]),
        ).handle(AgentTask(task_id="t1", request="Комбинированный запрос"))
    )

    compose_actions = [a for a in result.actions_taken if a.name == "orchestrator_llm_final_answer"]
    assert len(compose_actions) == 1
    assert compose_actions[0].status == "completed"
    assert set(result.handoff_to) == {"bitrix24", "pto"}


def test_orchestrator_single_specialist_no_extra_compose():
    manifests = load_agent_manifests()
    fake_llm = FakeInternalOrchestratorLLM(call_specialists=["bitrix24"])
    result = asyncio.run(
        _make_orch(
            {"bitrix24": _bitrix_specialist(manifests, final_answer="Готово.")},
            fake_llm,
        ).handle(AgentTask(task_id="t1", request="Задача в Битриксе"))
    )

    assert result.answer == "Готово."
    assert len(fake_llm.compose_calls) == 1
    compose_actions = [a for a in result.actions_taken if a.name == "orchestrator_llm_final_answer"]
    assert len(compose_actions) == 1


# Обработка исключений специалиста
def test_orchestrator_handles_specialist_exception():
    class BrokenSpecialist:
        async def handle(self, task):
            raise RuntimeError("Specialist exploded")

    manifests = load_agent_manifests()
    result = asyncio.run(
        _make_orch(
            {
                "bitrix24": BrokenSpecialist(),
                "pto": _pto_specialist(manifests, final_answer="ПТО в порядке."),
            },
            FakeInternalOrchestratorLLM(call_specialists=["bitrix24", "pto"]),
        ).handle(AgentTask(task_id="t1", request="Запрос"))
    )

    error_actions = [a for a in result.actions_taken if a.status == "error" and a.name == "call_specialist"]
    assert len(error_actions) == 1
    assert "RuntimeError" in error_actions[0].details.get("error", "") or "RuntimeError" in str(
        error_actions[0].details
    )


def test_orchestrator_all_specialists_fail_returns_failed():
    class BrokenSpecialist:
        async def handle(self, task):
            raise ValueError("always fails")

    result = asyncio.run(
        _make_orch(
            {"bitrix24": BrokenSpecialist(), "pto": BrokenSpecialist()},
            FakeInternalOrchestratorLLM(call_specialists=["bitrix24", "pto"]),
        ).handle(AgentTask(task_id="t1", request="Запрос"))
    )

    assert result.status == "failed"


def test_orchestrator_no_matching_specialists_returns_direct_answer():
    manifests = load_agent_manifests()
    result = asyncio.run(
        _make_orch(
            {"unrelated": _make_fake_specialist_cls().build(manifests[0])},
            FakeInternalOrchestratorLLM(
                call_specialists=["nonexistent_specialist"],
                answer="Специалист недоступен.",
            ),
        ).handle(AgentTask(task_id="t1", request="Вопрос"))
    )

    assert result.answer == "Специалист недоступен."
    assert result.handoff_to == []


def test_orchestrator_llm_failure_returns_failed(monkeypatch):
    class AlwaysFailingLLM:
        async def decide(self, **kwargs):
            raise ConnectionError("LLM is down")

        async def compose(self, **kwargs):
            raise ConnectionError("LLM is down")

    monkeypatch.setattr(_specialists_module, "_load_entrypoint", lambda ep: _make_fake_specialist_cls())
    result = asyncio.run(_make_orch({}, AlwaysFailingLLM()).handle(AgentTask(task_id="t1", request="Запрос")))

    assert result.status == "failed"
    assert "ConnectionError" in result.answer


# статус теперь из compose()
def test_orchestrator_direct_answer_no_specialists():
    manifests = load_agent_manifests()
    result = asyncio.run(
        _make_orch(
            {"bitrix24": _bitrix_specialist(manifests)},
            FakeInternalOrchestratorLLM(answer="Я отвечаю сам."),
        ).handle(AgentTask(task_id="t1", request="Просто вопрос"))
    )

    assert result.answer == "Я отвечаю сам."
    assert result.handoff_to == []
    assert result.status == "completed"
