import json

import pytest

from ai_server.models import AgentManifest
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
