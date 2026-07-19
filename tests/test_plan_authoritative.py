import asyncio
import json

import pytest

from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ModelUsageRecord
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.orchestrators.plan_authoritative import (
    FINAL_SCHEMA,
    PLAN_SCHEMA,
    PlanAuthoritativeOrchestrator,
    PlanRejected,
    _constraints,
    _decode_plan,
    _hash,
)
from ai_server.orchestrators.tools.call_specialist import CallSpecialistTool


def _plan(request: str, **changes):
    value = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "p1",
        "request_hash": _hash(request),
        "state": "EXECUTE",
        "clarification": None,
        "max_rounds": 3,
        "subtasks": [{"subtask_id": "s1", "specialist_id": "bitrix24", "request": request}],
    }
    value.update(changes)
    return json.dumps(value)


def test_rejected_plan_never_becomes_legacy_route():
    request = "только Битрикс покажи склад"
    constraints = _constraints(request, {"bitrix24": {}, "logistics": {}})
    raw = _plan(request, subtasks=[{"subtask_id": "s1", "specialist_id": "logistics", "request": request}])
    with pytest.raises(PlanRejected, match="SOURCE_RESTRICTION_VIOLATION"):
        _decode_plan(raw, plan_id="p1", request=request, constraints=constraints)


def test_final_can_only_order_executor_facts():
    facts = [
        {"subtask_id": "s1", "answer": "first"},
        {"subtask_id": "s2", "answer": "second"},
    ]
    raw = json.dumps({"schema_version": FINAL_SCHEMA, "plan_id": "p1", "response_hash": "h", "ordered_subtask_ids": ["s2", "s1"]})
    assert PlanAuthoritativeOrchestrator._decode_final(raw, "p1", "h", facts) == "second; first"
    invalid = json.dumps({"schema_version": FINAL_SCHEMA, "plan_id": "p1", "response_hash": "h", "ordered_subtask_ids": ["s1"]})
    with pytest.raises(PlanRejected, match="FINAL_COMPLETENESS_FAILED"):
        PlanAuthoritativeOrchestrator._decode_final(invalid, "p1", "h", facts)


def test_live_factory_selects_plan_authoritative_runtime():
    class Planner:
        async def plan(self, **kwargs):  # pragma: no cover - only factory contract matters here
            raise AssertionError

        async def finalize(self, **kwargs):  # pragma: no cover - only factory contract matters here
            raise AssertionError

    subject = InternalOrchestrator.build(
        AgentManifest(id="internal_orchestrator", name="Оркестр", kind="orchestrator", description="test"),
        manifests=[],
        orchestrator_llm=Planner(),
    )
    assert isinstance(subject, PlanAuthoritativeOrchestrator)


class _Planner:
    async def plan(self, *, task, constraints, **kwargs):
        return _plan(task.request, plan_id=constraints["plan_id"]), ModelUsageRecord(agent_id="test", provider="test", model="test")

    async def finalize(self, *, plan_id, response_hash, results, **kwargs):
        return json.dumps({"schema_version": FINAL_SCHEMA, "plan_id": plan_id, "response_hash": response_hash, "ordered_subtask_ids": [item["subtask_id"] for item in results]}), ModelUsageRecord(agent_id="test", provider="test", model="test")


class _Specialist:
    def __init__(self, result):
        self.result = result
        self.tasks = []

    async def handle(self, task):
        self.tasks.append(task)
        return self.result


def _live_subject(result, *, planner=None):
    specialist = _Specialist(result)
    specialist_manifest = AgentManifest(id="bitrix24", name="Битрикс", kind="specialist", description="test")
    call = CallSpecialistTool({"bitrix24": specialist}, [specialist_manifest])
    orchestrator = PlanAuthoritativeOrchestrator(
        AgentManifest(id="internal_orchestrator", name="Оркестр", kind="orchestrator", description="test"),
        agent_tools=[call], planner=planner or _Planner(), llm=planner or _Planner(),
    )
    return orchestrator, specialist


def test_handle_propagates_causal_ids_before_specialist_call():
    result = AgentResult(status="completed", agent_id="bitrix24", answer="ready")
    subject, specialist = _live_subject(result)
    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="склад", context={"dialog_key": "d1"})))
    assert output.status == "completed"
    received = specialist.tasks[0].context
    assert received["t0006_plan_id"].startswith("plan-")
    assert len(received["t0006_response_hash"]) == 64
    assert received["t0006_subtask_id"] == "s1"
    assert received["t0006_attempt_id"].startswith("attempt-")
    assert output.metadata["branches"][0]["attempt_id"] == received["t0006_attempt_id"]


def test_handle_preserves_approval_and_needs_human_state():
    specialist_result = AgentResult(
        status="needs_human", agent_id="bitrix24", answer="approval needed",
        actions_requiring_approval=[ActionRecord(name="write", status="pending")],
    )
    subject, _ = _live_subject(specialist_result)
    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="склад")))
    assert output.status == "needs_human"
    assert [item.name for item in output.actions_requiring_approval] == ["write"]


class _RephrasingPlanner(_Planner):
    async def plan(self, *, task, constraints, **kwargs):
        return _plan(
            task.request,
            plan_id=constraints["plan_id"],
            subtasks=[{"subtask_id": "s1", "specialist_id": "bitrix24", "request": "planner-shortened-input"}],
        ), ModelUsageRecord(agent_id="test", provider="test", model="test")


@pytest.mark.parametrize(
    "original_request",
    [
        "Битрикс найди проект Ларгус-2.",
        "Битрикс найди проект Ларгус 2.",
        "Битрикс создай задачу на меня: подготовить тестовый отчет. Не создавай сразу.",
        "Битрикс отмени черновик задачи.",
        "Битрикс создай задачу в проекте Ларгус-2 на меня: проверить документы. Не создавай сразу.",
    ],
)
def test_handle_preserves_verbatim_request_for_validated_bitrix_specialist(original_request):
    subject, specialist = _live_subject(AgentResult(status="completed", agent_id="bitrix24", answer="ready"), planner=_RephrasingPlanner())

    asyncio.run(subject.handle(AgentTask(task_id="t1", request=original_request, context={"dialog_key": "d1", "dialog_id": "chat1"})))

    received = specialist.tasks[0]
    assert received.request == original_request
    assert received.context["t0006_original_request"] == original_request
    assert received.context["t0006_effective_specialist_request"] == original_request
    assert received.context["t0006_planned_subtask_request"] == "planner-shortened-input"
