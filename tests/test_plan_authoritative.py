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
from tests.fakes import FakeOrchestratorStore


def _plan(request: str, **changes):
    value = {
        "schema_version": PLAN_SCHEMA,
        "plan_id": "p1",
        "request_hash": _hash(request),
        "state": "EXECUTE",
        "clarification": None,
        "max_rounds": 3,
        "subtasks": [
            {
                "subtask_id": "s1",
                "segment_id": None,
                "specialist_id": "bitrix24",
                "capability": "bitrix_warehouse_search",
                "request": request,
            }
        ],
    }
    value.update(changes)
    return json.dumps(value)


def test_rejected_plan_never_becomes_legacy_route():
    request = "только Битрикс покажи склад"
    constraints = _constraints(request, {"bitrix24": {}, "logistics": {}})
    raw = _plan(
        request,
        subtasks=[
            {
                "subtask_id": "s1",
                "segment_id": None,
                "specialist_id": "logistics",
                "capability": "vehicle_usage_context",
                "request": request,
            }
        ],
    )
    with pytest.raises(PlanRejected, match="SOURCE_RESTRICTION_VIOLATION"):
        _decode_plan(raw, plan_id="p1", request=request, constraints=constraints)


def test_final_can_only_order_executor_facts():
    facts = [
        {"subtask_id": "s1", "answer": "first"},
        {"subtask_id": "s2", "answer": "second"},
    ]
    raw = json.dumps(
        {"schema_version": FINAL_SCHEMA, "plan_id": "p1", "response_hash": "h", "ordered_subtask_ids": ["s2", "s1"]}
    )
    assert PlanAuthoritativeOrchestrator._decode_final(raw, "p1", "h", facts) == "second; first"
    invalid = json.dumps(
        {"schema_version": FINAL_SCHEMA, "plan_id": "p1", "response_hash": "h", "ordered_subtask_ids": ["s1"]}
    )
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
        return _plan(task.request, plan_id=constraints["plan_id"]), ModelUsageRecord(
            agent_id="test", provider="test", model="test"
        )

    async def finalize(self, *, plan_id, response_hash, results, **kwargs):
        return json.dumps(
            {
                "schema_version": FINAL_SCHEMA,
                "plan_id": plan_id,
                "response_hash": response_hash,
                "ordered_subtask_ids": [item["subtask_id"] for item in results],
            }
        ), ModelUsageRecord(agent_id="test", provider="test", model="test")


class _Specialist:
    def __init__(self, result):
        self.result = result
        self.tasks = []

    async def handle(self, task):
        self.tasks.append(task)
        return self.result


def _live_subject(result, *, planner=None, store=None):
    specialist = _Specialist(result)
    specialist_manifest = AgentManifest(id="bitrix24", name="Битрикс", kind="specialist", description="test")
    specialist_manifest.capabilities = ["bitrix_warehouse_search"]
    call = CallSpecialistTool({"bitrix24": specialist}, [specialist_manifest], store=store)
    orchestrator = PlanAuthoritativeOrchestrator(
        AgentManifest(id="internal_orchestrator", name="Оркестр", kind="orchestrator", description="test"),
        agent_tools=[call],
        planner=planner or _Planner(),
        llm=planner or _Planner(),
        store=store,
    )
    return orchestrator, specialist


class _ForbiddenPlanner:
    def __init__(self):
        self.calls = []

    async def plan(self, *, task, constraints, **kwargs):
        self.calls.append(dict(constraints))
        return _plan(task.request, plan_id=constraints["plan_id"]), ModelUsageRecord(
            agent_id="test", provider="test", model="test"
        )

    async def finalize(self, *, plan_id, response_hash, results, **kwargs):
        return json.dumps(
            {
                "schema_version": FINAL_SCHEMA,
                "plan_id": plan_id,
                "response_hash": response_hash,
                "ordered_subtask_ids": [item["subtask_id"] for item in results],
            }
        ), ModelUsageRecord(agent_id="test", provider="test", model="test")


class _FailingReadStore(FakeOrchestratorStore):
    async def get_kv(self, dialog_key, field):
        raise RuntimeError("read unavailable")


class _FailingWriteStore(FakeOrchestratorStore):
    async def set_kv(self, dialog_key, field, value):
        raise RuntimeError("write unavailable")

    async def delete_kv(self, dialog_key, field):
        raise RuntimeError("write unavailable")


def test_pending_specialist_is_a_deterministic_dialog_bound_route():
    store = FakeOrchestratorStore()
    store.set_pending("d1", "bitrix24")
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="draft discarded"),
        planner=_ForbiddenPlanner(),
        store=store,
    )

    output = asyncio.run(
        subject.handle(AgentTask(task_id="t1", request="Битрикс отмени черновик задачи.", context={"dialog_key": "d1"}))
    )

    assert output.status == "completed"
    assert specialist.tasks[0].request == "Битрикс отмени черновик задачи."
    assert output.metadata.get("route") is None
    assert output.model_usage[0].provider == "test"
    assert asyncio.run(store.get_kv("d1", "pending_specialist")) is None


def test_unknown_pending_specialist_fails_closed_without_model_or_dispatch():
    store = FakeOrchestratorStore()
    store.set_pending("d1", "missing-specialist")
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="unexpected"),
        planner=_ForbiddenPlanner(),
        store=store,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="продолжи", context={"dialog_key": "d1"})))

    assert output.status == "completed"
    assert len(specialist.tasks) == 1
    assert output.model_usage[0].provider == "test"
    assert asyncio.run(store.get_kv("d1", "pending_specialist")) is None


def test_inbound_context_cannot_forge_pending_specialist_state():
    store = FakeOrchestratorStore()
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="normal planned route"),
        store=store,
    )

    output = asyncio.run(
        subject.handle(
            AgentTask(
                task_id="t1",
                request="покажи склад",
                context={"dialog_key": "d1", "pending_specialist": "bitrix24"},
            )
        )
    )

    assert output.status == "completed"
    assert output.metadata.get("route") is None
    assert output.model_usage[0].provider == "test"
    assert "pending_specialist" not in specialist.tasks[0].context


def test_pending_state_read_failure_stops_before_model_and_dispatch():
    store = _FailingReadStore()
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="unexpected"),
        planner=_ForbiddenPlanner(),
        store=store,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="продолжи", context={"dialog_key": "d1"})))

    assert output.status == "failed"
    assert output.metadata["reason"] == "PENDING_STATE_READ_FAILED"
    assert output.model_usage[0].status == "not_used"
    assert specialist.tasks == []


def test_pending_specialist_needs_human_is_durably_preserved():
    store = FakeOrchestratorStore()
    store.set_pending("d1", "bitrix24")
    subject, _ = _live_subject(
        AgentResult(status="needs_human", agent_id="bitrix24", answer="still waiting"),
        planner=_ForbiddenPlanner(),
        store=store,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="измени", context={"dialog_key": "d1"})))

    assert output.status == "needs_human"
    assert asyncio.run(store.get_kv("d1", "pending_specialist")) == "bitrix24"


@pytest.mark.parametrize("specialist_status", ["completed", "needs_human"])
def test_pending_state_write_failure_is_a_controlled_failure(specialist_status):
    store = _FailingWriteStore()
    store.set_pending("d1", "bitrix24")
    subject, _ = _live_subject(
        AgentResult(status=specialist_status, agent_id="bitrix24", answer="must not be acknowledged"),
        planner=_ForbiddenPlanner(),
        store=store,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="продолжи", context={"dialog_key": "d1"})))

    assert output.status == "failed"
    assert output.answer == "pending specialist state transition failed"
    assert output.metadata["branches"][0]["status"] == "failed"


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
        status="needs_human",
        agent_id="bitrix24",
        answer="approval needed",
        actions_requiring_approval=[ActionRecord(name="write", status="pending")],
    )
    subject, _ = _live_subject(specialist_result)
    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="склад")))
    assert output.status == "needs_human"
    assert [item.name for item in output.actions_requiring_approval] == ["write"]


def test_new_calendar_request_is_held_as_candidate_during_active_task_draft():
    class DraftingSpecialist(_Specialist):
        async def get_active_draft(self, dialog_key):
            return {"_draft_id": "draft-1", "_draft_type": "task_create", "_draft_version": 1}

    store = FakeOrchestratorStore()
    specialist = DraftingSpecialist(AgentResult(status="completed", agent_id="bitrix24", answer="unexpected"))
    manifest = AgentManifest(
        id="bitrix24", name="Bitrix", kind="specialist", description="test", capabilities=["bitrix_warehouse_search"]
    )
    call = CallSpecialistTool({"bitrix24": specialist}, [manifest], store=store)
    subject = PlanAuthoritativeOrchestrator(
        AgentManifest(id="internal_orchestrator", name="Orchestrator", kind="orchestrator", description="test"),
        agent_tools=[call],
        planner=_Planner(),
        llm=_Planner(),
        store=store,
    )

    output = asyncio.run(
        subject.handle(
            AgentTask(
                task_id="t1",
                request="Создай напоминание на завтра на 12 0 0 проверить отчет",
                context={"dialog_key": "d1"},
            )
        )
    )

    assert output.status == "needs_clarification"
    assert len(specialist.tasks) == 0
    assert output.metadata["reason"] == "REPLACEMENT_CANDIDATE_SAVED"
    candidate = asyncio.run(store.get_replacement_candidate("d1"))
    assert candidate["request_text"] == "Создай напоминание на завтра на 12 0 0 проверить отчет"
    assert candidate["draft_id"] == "draft-1"


def test_second_task_draft_reaches_pro_plan_as_same_type_revision():
    class DraftingSpecialist(_Specialist):
        async def get_active_draft(self, dialog_key):
            return {"_draft_id": "draft-1", "_draft_type": "task_create", "_draft_version": 1}

    store = FakeOrchestratorStore()
    specialist = DraftingSpecialist(AgentResult(status="completed", agent_id="bitrix24", answer="unexpected"))
    manifest = AgentManifest(
        id="bitrix24", name="Bitrix", kind="specialist", description="test", capabilities=["bitrix_warehouse_search"]
    )
    call = CallSpecialistTool({"bitrix24": specialist}, [manifest], store=store)
    subject = PlanAuthoritativeOrchestrator(
        AgentManifest(id="internal_orchestrator", name="Orchestrator", kind="orchestrator", description="test"),
        agent_tools=[call],
        planner=_Planner(),
        llm=_Planner(),
        store=store,
    )

    output = asyncio.run(
        subject.handle(AgentTask(task_id="t1", request="создай задачу проверить договор", context={"dialog_key": "d1"}))
    )

    assert output.status == "completed"
    assert len(specialist.tasks) == 1


def test_read_only_request_remains_available_while_replacement_candidate_waits():
    class DraftingSpecialist(_Specialist):
        async def get_active_draft(self, dialog_key):
            return {"_draft_id": "draft-1", "_draft_type": "task_create", "_draft_version": 1}

    store = FakeOrchestratorStore()
    asyncio.run(
        store.save_replacement_candidate(
            "d1",
            request_text="Создай напоминание на завтра проверить отчёт",
            draft_id="draft-1",
            draft_type="task_create",
        )
    )
    specialist = DraftingSpecialist(AgentResult(status="completed", agent_id="bitrix24", answer="Задачи найдены"))
    manifest = AgentManifest(
        id="bitrix24", name="Bitrix", kind="specialist", description="test", capabilities=["bitrix_warehouse_search"]
    )
    call = CallSpecialistTool({"bitrix24": specialist}, [manifest], store=store)
    subject = PlanAuthoritativeOrchestrator(
        AgentManifest(id="internal_orchestrator", name="Orchestrator", kind="orchestrator", description="test"),
        agent_tools=[call],
        planner=_Planner(),
        llm=_Planner(),
        store=store,
    )

    output = asyncio.run(
        subject.handle(AgentTask(task_id="t1", request="Покажи мои задачи", context={"dialog_key": "d1"}))
    )

    assert output.status == "completed"
    assert len(specialist.tasks) == 1
    assert asyncio.run(store.get_replacement_candidate("d1")) is not None


def test_duplicate_branch_is_rejected_before_any_specialist_can_run():
    request = "покажи склад Борисова"
    constraints = _constraints(request, {"bitrix24": {"capabilities": ["bitrix_warehouse_search"]}})
    raw = _plan(
        request,
        subtasks=[
            {
                "subtask_id": "s1",
                "segment_id": None,
                "specialist_id": "bitrix24",
                "capability": "bitrix_warehouse_search",
                "request": request,
            },
            {
                "subtask_id": "s2",
                "segment_id": None,
                "specialist_id": "bitrix24",
                "capability": "bitrix_warehouse_search",
                "request": request,
            },
        ],
    )

    with pytest.raises(PlanRejected, match="DUPLICATE_SUBTASK"):
        _decode_plan(raw, plan_id="p1", request=request, constraints=constraints)


def test_explicit_three_warehouse_request_requires_one_branch_per_label():
    request = "Покажи склад Борисова Карасева и Ивашина"
    constraints = _constraints(request, {"bitrix24": {"capabilities": ["bitrix_warehouse_search"]}})
    raw = _plan(
        request,
        subtasks=[
            {
                "subtask_id": "s1",
                "segment_id": None,
                "specialist_id": "bitrix24",
                "capability": "bitrix_warehouse_search",
                "request": "покажи склад Ивашина",
            }
        ],
    )

    assert constraints["required_warehouse_labels"] == ["борисова", "карасева", "ивашина"]
    with pytest.raises(PlanRejected, match="WAREHOUSE_SEGMENT_INCOMPLETE"):
        _decode_plan(raw, plan_id="p1", request=request, constraints=constraints)


@pytest.mark.parametrize(
    ("user_text", "labels"),
    [
        ("Покажи склад Борисова и Карасева", ["борисова", "карасева"]),
        ("Покажи остатки склад Карасева и Борисова", ["карасева", "борисова"]),
        ("Покажи остатки по складам Борисова, Карасева и Ивашина", ["борисова", "карасева", "ивашина"]),
    ],
)
def test_warehouse_wording_variants_require_one_validated_branch_per_label(user_text, labels):
    constraints = _constraints(user_text, {"bitrix24": {"capabilities": ["bitrix_warehouse_search"]}})
    assert constraints["required_warehouse_labels"] == labels

    incomplete = _plan(
        user_text,
        subtasks=[
            {
                "subtask_id": "s1",
                "segment_id": None,
                "specialist_id": "bitrix24",
                "capability": "bitrix_warehouse_search",
                "request": f"Найти склад {labels[0]}",
            }
        ],
    )
    with pytest.raises(PlanRejected, match="WAREHOUSE_SEGMENT_INCOMPLETE"):
        _decode_plan(incomplete, plan_id="p1", request=user_text, constraints=constraints)

    complete = _plan(
        user_text,
        subtasks=[
            {
                "subtask_id": f"s{index}",
                "segment_id": None,
                "specialist_id": "bitrix24",
                "capability": "bitrix_warehouse_search",
                "request": f"Найти склад {label}",
            }
            for index, label in enumerate(labels, start=1)
        ],
    )
    plan = _decode_plan(complete, plan_id="p1", request=user_text, constraints=constraints)
    assert len(plan.subtasks) == len(labels)


def test_composite_plan_dispatches_each_validated_branch_not_the_whole_dialog_request():
    class CompositePlanner(_Planner):
        async def plan(self, *, task, constraints, **kwargs):
            return _plan(
                task.request,
                plan_id=constraints["plan_id"],
                subtasks=[
                    {
                        "subtask_id": "s1",
                        "segment_id": None,
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": "покажи склад Борисова",
                    },
                    {
                        "subtask_id": "s2",
                        "segment_id": None,
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": "покажи склад Карасева",
                    },
                ],
            ), ModelUsageRecord(agent_id="test", provider="test", model="test")

    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="ready"), planner=CompositePlanner()
    )

    asyncio.run(subject.handle(AgentTask(task_id="t1", request="склады Борисова и Карасева")))

    assert [item.request for item in specialist.tasks] == ["покажи склад Борисова", "покажи склад Карасева"]


def test_three_warehouse_plan_dispatches_three_independent_virtual_branches():
    labels = ["Борисова", "Карасева", "Ивашина"]

    class ThreeWarehousePlanner(_Planner):
        async def plan(self, *, task, constraints, **kwargs):
            return _plan(
                task.request,
                plan_id=constraints["plan_id"],
                subtasks=[
                    {
                        "subtask_id": f"warehouse-{index}",
                        "segment_id": None,
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": f"Найти склад {label}",
                    }
                    for index, label in enumerate(labels, start=1)
                ],
            ), ModelUsageRecord(agent_id="test", provider="test", model="test")

    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="warehouse page"),
        planner=ThreeWarehousePlanner(),
    )
    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="Покажи склады Борисова, Карасева и Ивашина")))

    assert len(specialist.tasks) == 3
    assert {item.request for item in specialist.tasks} == {f"Найти склад {label}" for label in labels}
    assert len(output.metadata["branches"]) == 3


@pytest.mark.parametrize(
    "labels",
    [
        ["Борисова", "Карасева"],
        ["Борисова", "Карасева", "Ивашина"],
    ],
)
def test_multi_warehouse_virtual_flow_keeps_every_result_in_one_answer(labels):
    class WarehousePlanner(_Planner):
        async def plan(self, *, task, constraints, **kwargs):
            return _plan(
                task.request,
                plan_id=constraints["plan_id"],
                subtasks=[
                    {
                        "subtask_id": f"warehouse-{index}",
                        "segment_id": None,
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": f"Найти склад {label}",
                    }
                    for index, label in enumerate(labels, start=1)
                ],
            ), ModelUsageRecord(agent_id="test", provider="test", model="test")

    class WarehouseSpecialist:
        def __init__(self):
            self.tasks = []

        async def handle(self, task):
            self.tasks.append(task)
            label = task.request.removeprefix("Найти склад ")
            return AgentResult(status="completed", agent_id="bitrix24", answer=f"Остатки склада {label}")

    specialist = WarehouseSpecialist()
    manifest = AgentManifest(id="bitrix24", name="Битрикс", kind="specialist", description="test")
    manifest.capabilities = ["bitrix_warehouse_search"]
    call = CallSpecialistTool({"bitrix24": specialist}, [manifest])
    planner = WarehousePlanner()
    subject = PlanAuthoritativeOrchestrator(
        AgentManifest(id="internal_orchestrator", name="Оркестр", kind="orchestrator", description="test"),
        agent_tools=[call],
        planner=planner,
        llm=planner,
    )

    output = asyncio.run(
        subject.handle(AgentTask(task_id="t1", request=f"Покажи склады {', '.join(labels[:-1])} и {labels[-1]}"))
    )

    assert output.status == "completed"
    assert [task.request for task in specialist.tasks] == [f"Найти склад {label}" for label in labels]
    for label in labels:
        assert f"Остатки склада {label}" in output.answer
    assert len(output.metadata["branches"]) == len(labels)


def test_composite_result_is_assembled_in_validated_plan_order_without_a_final_model_call():
    class CompositePlanner(_Planner):
        def __init__(self):
            self.final_calls = 0

        async def plan(self, *, task, constraints, **kwargs):
            return _plan(
                task.request,
                plan_id=constraints["plan_id"],
                subtasks=[
                    {
                        "subtask_id": "s1",
                        "segment_id": None,
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": "first",
                    },
                    {
                        "subtask_id": "s2",
                        "segment_id": None,
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": "second",
                    },
                ],
            ), ModelUsageRecord(agent_id="test", provider="test", model="test")

        async def finalize(self, **kwargs):
            self.final_calls += 1
            raise AssertionError("deterministic renderer must not call a final model")

    planner = CompositePlanner()
    subject, _ = _live_subject(AgentResult(status="completed", agent_id="bitrix24", answer="fact"), planner=planner)

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="two branches")))

    assert output.status == "completed"
    assert planner.final_calls == 0
    assert output.model_usage == [ModelUsageRecord(agent_id="test", provider="test", model="test")]
    assert (
        next(action for action in output.actions_taken if action.name == "final_validation").details["reason"]
        == "VALIDATED_PLAN_ORDER"
    )


def test_unknown_number_still_calls_pro_but_cannot_dispatch_a_specialist():
    class ReferencePlanner(_Planner):
        def __init__(self):
            self.calls = 0

        async def plan(self, *, task, constraints, **kwargs):
            self.calls += 1
            return _plan(
                task.request,
                plan_id=constraints["plan_id"],
                state="CLARIFICATION_REQUIRED",
                clarification="Укажите актуальный диалог.",
                subtasks=[],
            ), ModelUsageRecord(agent_id="test", provider="test", model="test")

    store = FakeOrchestratorStore()
    planner = ReferencePlanner()
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="must not run"), planner=planner, store=store
    )
    task = AgentTask(
        task_id="t1",
        request="Покажи следующую страницу 999",
        context={"dialog_key": "chat:4321:user:1", "base_dialog_key": "chat:4321:user:1"},
    )

    output = asyncio.run(subject.handle(task))

    assert output.status == "needs_clarification"
    assert planner.calls == 1
    assert specialist.tasks == []
    assert output.model_usage[0].provider == "test"


def test_numbered_warehouse_continuation_reaches_the_exact_original_branch():
    store = FakeOrchestratorStore()
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="warehouse page"),
        store=store,
    )
    base_context = {"dialog_key": "chat:4321:user:1", "base_dialog_key": "chat:4321:user:1"}

    asyncio.run(subject.handle(AgentTask(task_id="t1", request="Покажи склад Борисова", context=base_context)))
    asyncio.run(subject.handle(AgentTask(task_id="t2", request="Покажи склад Карасева", context=base_context)))
    asyncio.run(subject.handle(AgentTask(task_id="t3", request="101 следующая страница", context=base_context)))

    first_branch = specialist.tasks[0].context["dialog_key"]
    second_branch = specialist.tasks[1].context["dialog_key"]
    continuation = specialist.tasks[2]
    assert first_branch != second_branch
    assert continuation.request == "следующая страница"
    assert continuation.context["conversation_number"] == 101
    assert continuation.context["dialog_key"] == first_branch
    assert continuation.context["dialog_key"] != second_branch


def test_trace_captures_pro_plan_parallel_specialists_and_deterministic_render():
    class TimingTrace:
        def __init__(self):
            self.events = []

        async def record_timing(self, **kwargs):
            self.events.append(kwargs)

    class CompositePlanner(_Planner):
        async def plan(self, *, task, constraints, **kwargs):
            return _plan(
                task.request,
                plan_id=constraints["plan_id"],
                subtasks=[
                    {
                        "subtask_id": "s1",
                        "segment_id": None,
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": "first",
                    },
                    {
                        "subtask_id": "s2",
                        "segment_id": None,
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": "second",
                    },
                ],
            ), ModelUsageRecord(agent_id="test", provider="test", model="test")

    subject, _ = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="fact"), planner=CompositePlanner()
    )
    trace = TimingTrace()
    subject._conversation_trace = trace

    asyncio.run(subject.handle(AgentTask(task_id="t1", request="two branches")))

    stages = [event["stage"] for event in trace.events]
    assert {
        "orchestrator_entry",
        "conversation_reference",
        "orchestrator_state_load",
        "pro_plan",
        "plan_validation",
        "deterministic_render",
    } <= set(stages)
    dispatches = [event for event in trace.events if event["stage"] == "specialist_dispatch"]
    assert len(dispatches) == 2
    assert {event["details"]["subtask_id"] for event in dispatches} == {"s1", "s2"}
    assert len({event["details"]["parallel_group"] for event in dispatches}) == 1


def test_multi_source_partial_failure_keeps_success_and_names_failed_source():
    class _MultiPlanner(_Planner):
        async def plan(self, *, task, constraints, **kwargs):
            return _plan(
                task.request,
                plan_id=constraints["plan_id"],
                subtasks=[
                    {
                        "subtask_id": "s1",
                        "segment_id": None,
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": task.request,
                    },
                    {
                        "subtask_id": "s2",
                        "segment_id": None,
                        "specialist_id": "kartoteka",
                        "capability": "search",
                        "request": task.request,
                    },
                ],
            ), ModelUsageRecord(agent_id="test", provider="test", model="test")

    class _RaisingSpecialist:
        async def handle(self, task):
            raise RuntimeError("injected source failure")

    planner = _MultiPlanner()
    bitrix = _Specialist(AgentResult(status="completed", agent_id="bitrix24", answer="verified result"))
    kartoteka = _RaisingSpecialist()
    manifests = [
        AgentManifest(
            id="bitrix24",
            name="Bitrix",
            kind="specialist",
            description="test",
            capabilities=["bitrix_warehouse_search"],
        ),
        AgentManifest(id="kartoteka", name="Kartoteka", kind="specialist", description="test", capabilities=["search"]),
    ]
    call = CallSpecialistTool({"bitrix24": bitrix, "kartoteka": kartoteka}, manifests)
    subject = PlanAuthoritativeOrchestrator(
        AgentManifest(id="internal_orchestrator", name="Orchestrator", kind="orchestrator", description="test"),
        agent_tools=[call],
        planner=planner,
        llm=planner,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="find across sources")))

    assert output.status == "completed"
    assert "Источник bitrix24: verified result" in output.answer
    assert "Источник kartoteka: не завершил обработку" in output.answer
    assert [item["specialist_id"] for item in output.metadata["branches"]] == ["bitrix24", "kartoteka"]
    assert [item["status"] for item in output.metadata["branches"]] == ["completed", "error"]


class _RephrasingPlanner(_Planner):
    async def plan(self, *, task, constraints, **kwargs):
        return _plan(
            task.request,
            plan_id=constraints["plan_id"],
            subtasks=[
                {
                    "subtask_id": "s1",
                    "segment_id": None,
                    "specialist_id": "bitrix24",
                    "capability": "bitrix_warehouse_search",
                    "request": "planner-shortened-input",
                }
            ],
        ), ModelUsageRecord(agent_id="test", provider="test", model="test")


class _RepairingPlanner(_Planner):
    def __init__(self, *, second_valid=True, first_reason="schema", second_raises=False):
        self.calls = []
        self.second_valid = second_valid
        self.first_reason = first_reason
        self.second_raises = second_raises

    async def plan(self, *, task, constraints, **kwargs):
        self.calls.append(dict(constraints))
        usage = ModelUsageRecord(agent_id="test", provider="test", model="test")
        if len(self.calls) == 1:
            if self.first_reason == "semantic":
                return _plan(
                    task.request,
                    plan_id=constraints["plan_id"],
                    subtasks=[
                        {
                            "subtask_id": "s1",
                            "segment_id": None,
                            "specialist_id": "missing",
                            "capability": "missing",
                            "request": task.request,
                        }
                    ],
                ), usage
            return json.dumps({"unexpected": True}), usage
        if self.second_raises:
            raise RuntimeError("provider unavailable during repair")
        if not self.second_valid:
            return json.dumps({"still_unexpected": True}), usage
        return _plan(task.request, plan_id=constraints["plan_id"]), usage


def test_repairable_plan_contract_failure_gets_one_changed_method_retry():
    planner = _RepairingPlanner()
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="ready"),
        planner=planner,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="warehouse")))

    assert output.status == "completed"
    assert len(planner.calls) == 2
    assert "repair_reason" not in planner.calls[0]
    assert planner.calls[1]["repair_reason"] == "PLAN_SCHEMA_MISMATCH"
    assert planner.calls[1]["repair_attempt"] == 2
    assert output.metadata["planner_attempts"] == 2
    assert output.metadata["planner_rejections"] == ["PLAN_SCHEMA_MISMATCH"]
    assert output.metadata["planner_attempt_audit"] == [
        {
            "attempt": 1,
            "response_hash": _hash(json.dumps({"unexpected": True})),
            "status": "rejected",
            "rejection": "PLAN_SCHEMA_MISMATCH",
        },
        {
            "attempt": 2,
            "response_hash": output.metadata["response_hash"],
            "status": "accepted",
        },
    ]
    assert len(output.model_usage) == 2
    assert len(specialist.tasks) == 1


def test_repairable_plan_contract_failure_fails_closed_after_one_retry():
    planner = _RepairingPlanner(second_valid=False)
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="unexpected"),
        planner=planner,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="warehouse")))

    assert output.status == "failed"
    assert len(planner.calls) == 2
    assert output.metadata["reason"] == "PLAN_SCHEMA_MISMATCH"
    assert output.metadata["planner_attempts"] == 2
    assert len(output.model_usage) == 2
    assert specialist.tasks == []


def test_repair_provider_error_fails_closed_and_preserves_first_attempt_audit():
    planner = _RepairingPlanner(second_raises=True)
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="unexpected"),
        planner=planner,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="warehouse")))

    assert output.status == "failed"
    assert output.metadata["reason"] == "MODEL_REPAIR_UNAVAILABLE"
    assert output.metadata["planner_attempts"] == 2
    assert output.metadata["planner_rejections"] == [
        "PLAN_SCHEMA_MISMATCH",
        "MODEL_REPAIR_UNAVAILABLE",
    ]
    assert output.metadata["planner_attempt_audit"][0]["response_hash"] == _hash(json.dumps({"unexpected": True}))
    assert output.metadata["planner_attempt_audit"][1] == {
        "attempt": 2,
        "status": "error",
        "rejection": "MODEL_REPAIR_UNAVAILABLE",
    }
    assert len(output.model_usage) == 1
    assert specialist.tasks == []


class _MalformedFinalPlanner(_Planner):
    async def finalize(self, **kwargs):
        return json.dumps({"unexpected": True}), ModelUsageRecord(
            agent_id="test",
            provider="test",
            model="test-final",
        )


def test_single_verified_specialist_skips_final_model_call():
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="executor fact"),
        planner=_MalformedFinalPlanner(),
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="warehouse")))

    assert output.status == "completed"
    assert output.answer == "executor fact"
    assert [usage.model for usage in output.model_usage] == ["test"]
    final_action = next(action for action in output.actions_taken if action.name == "final_validation")
    assert final_action.status == "deterministic"
    assert final_action.details["reason"] == "VALIDATED_PLAN_ORDER"
    assert len(specialist.tasks) == 1


def test_semantic_plan_rejection_is_not_retried():
    planner = _RepairingPlanner(first_reason="semantic")
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="unexpected"),
        planner=planner,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request="warehouse")))

    assert output.status == "failed"
    assert len(planner.calls) == 1
    assert output.metadata["reason"] == "FORBIDDEN_SPECIALIST"
    assert output.metadata["planner_attempts"] == 1
    assert len(output.model_usage) == 1
    assert specialist.tasks == []


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
    subject, specialist = _live_subject(
        AgentResult(status="completed", agent_id="bitrix24", answer="ready"), planner=_RephrasingPlanner()
    )

    asyncio.run(
        subject.handle(
            AgentTask(task_id="t1", request=original_request, context={"dialog_key": "d1", "dialog_id": "chat1"})
        )
    )

    received = specialist.tasks[0]
    assert received.request == original_request
    assert received.context["t0006_original_request"] == original_request
    assert received.context["t0006_effective_specialist_request"] == original_request
    assert received.context["t0006_planned_subtask_request"] == "planner-shortened-input"
    assert received.context["t0006_planned_capability"] == "bitrix_warehouse_search"


def test_explicit_segments_bind_each_part_to_its_named_specialist():
    request = "Логист: покажи автомобили; Bitrix: покажи склад Борисова"
    catalog = {
        "logistics": {"capabilities": ["vehicle_usage_context"]},
        "bitrix24": {"capabilities": ["bitrix_warehouse_search"]},
    }
    constraints = _constraints(request, catalog)
    raw = _plan(
        request,
        subtasks=[
            {
                "subtask_id": "s1",
                "segment_id": "segment-1",
                "specialist_id": "logistics",
                "capability": "vehicle_usage_context",
                "request": "покажи автомобили",
            },
            {
                "subtask_id": "s2",
                "segment_id": "segment-2",
                "specialist_id": "bitrix24",
                "capability": "bitrix_warehouse_search",
                "request": "покажи склад Борисова",
            },
        ],
    )
    plan = _decode_plan(raw, plan_id="p1", request=request, constraints=constraints)
    assert [item.specialist_id for item in plan.subtasks] == ["logistics", "bitrix24"]


def test_voice_style_explicit_segments_bind_each_named_specialist_without_punctuation():
    request = "Логист покажи машины Битрикс покажи склад Борисова"
    catalog = {
        "logistics": {"capabilities": ["vehicle_usage_context"]},
        "bitrix24": {"capabilities": ["bitrix_warehouse_search"]},
    }
    segments = _constraints(request, catalog)["explicit_segments"]
    assert segments == [
        {"segment_id": "segment-1", "specialist_id": "logistics", "request": "покажи машины"},
        {"segment_id": "segment-2", "specialist_id": "bitrix24", "request": "покажи склад Борисова"},
    ]


def test_voice_style_logistics_and_bitrix_flow_dispatches_exact_parts_and_keeps_both_results():
    request = "Логист покажи машины Битрикс покажи склад Борисова"

    class VoiceCompositePlanner(_Planner):
        async def plan(self, *, task, constraints, **kwargs):
            return _plan(
                task.request,
                plan_id=constraints["plan_id"],
                subtasks=[
                    {
                        "subtask_id": "logistics",
                        "segment_id": "segment-1",
                        "specialist_id": "logistics",
                        "capability": "vehicle_usage_context",
                        "request": "покажи машины",
                    },
                    {
                        "subtask_id": "warehouse",
                        "segment_id": "segment-2",
                        "specialist_id": "bitrix24",
                        "capability": "bitrix_warehouse_search",
                        "request": "покажи склад Борисова",
                    },
                ],
            ), ModelUsageRecord(agent_id="test", provider="test", model="test")

    class RecordingSpecialist:
        def __init__(self, agent_id, answer):
            self.agent_id = agent_id
            self.answer = answer
            self.tasks = []

        async def handle(self, task):
            self.tasks.append(task)
            return AgentResult(status="completed", agent_id=self.agent_id, answer=self.answer)

    logistics = RecordingSpecialist("logistics", "Отчёт по машинам готов")
    bitrix = RecordingSpecialist("bitrix24", "Остатки склада Борисова готовы")
    logistics_manifest = AgentManifest(id="logistics", name="Логист", kind="specialist", description="test")
    logistics_manifest.capabilities = ["vehicle_usage_context"]
    bitrix_manifest = AgentManifest(id="bitrix24", name="Битрикс", kind="specialist", description="test")
    bitrix_manifest.capabilities = ["bitrix_warehouse_search"]
    call = CallSpecialistTool(
        {"logistics": logistics, "bitrix24": bitrix},
        [logistics_manifest, bitrix_manifest],
    )
    planner = VoiceCompositePlanner()
    subject = PlanAuthoritativeOrchestrator(
        AgentManifest(id="internal_orchestrator", name="Оркестр", kind="orchestrator", description="test"),
        agent_tools=[call],
        planner=planner,
        llm=planner,
    )

    output = asyncio.run(subject.handle(AgentTask(task_id="t1", request=request)))

    assert [task.request for task in logistics.tasks] == ["покажи машины"]
    assert [task.request for task in bitrix.tasks] == ["покажи склад Борисова"]
    assert "Отчёт по машинам готов" in output.answer
    assert "Остатки склада Борисова готовы" in output.answer
    assert len(output.metadata["branches"]) == 2


def test_explicit_segment_cannot_be_silently_sent_to_another_specialist():
    request = "Bitrix: покажи склад Борисова"
    catalog = {
        "logistics": {"capabilities": ["vehicle_usage_context"]},
        "bitrix24": {"capabilities": ["bitrix_warehouse_search"]},
    }
    constraints = _constraints(request, catalog)
    raw = _plan(
        request,
        subtasks=[
            {
                "subtask_id": "s1",
                "segment_id": "segment-1",
                "specialist_id": "logistics",
                "capability": "vehicle_usage_context",
                "request": "покажи склад Борисова",
            }
        ],
    )
    with pytest.raises(PlanRejected, match="SEGMENT_BINDING_INVALID"):
        _decode_plan(raw, plan_id="p1", request=request, constraints=constraints)
