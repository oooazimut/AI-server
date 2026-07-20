import asyncio
import json

from ai_server.orchestrators.local_harness import PLAN_SCHEMA, LocalOrchestratorHarness, planner_prompt, sha256_text


def test_prompt_contract_contains_context_catalog_and_hard_constraints():
    prompt = planner_prompt(
        plan_id="p", request="Только Bitrix: найди склад", dialog_history=[{"role": "user", "content": "before"}]
    )
    assert prompt["dialog_history"] and prompt["capability_catalog"]
    assert prompt["hard_constraints"]["only_source"] == "bitrix"
    assert prompt["required_response"]["plan_id"] == "p"


def test_extra_model_branch_not_requested_by_text_is_rejected_before_dispatch():
    request = "Найди склад Борисова"
    raw = json.dumps(
        {
            "schema_version": PLAN_SCHEMA,
            "plan_id": "p",
            "request_hash": sha256_text(request),
            "state": "EXECUTE",
            "clarification": None,
            "max_rounds": 1,
            "subtasks": [{"subtask_id": "extra", "capability": "delivery", "input": {"query": "Борисов"}}],
        },
        ensure_ascii=False,
    )
    result = asyncio.run(LocalOrchestratorHarness().run_case("extra", request, raw, plan_id="p"))
    assert result.executor_calls == 0
    assert result.plan_validation["reason"] == "FORBIDDEN_CAPABILITY"


def test_boolean_round_limit_and_free_text_final_are_rejected_before_publish():
    request = "Найди склад Борисова"
    boolean_rounds = json.dumps(
        {
            "schema_version": PLAN_SCHEMA,
            "plan_id": "p",
            "request_hash": sha256_text(request),
            "state": "EXECUTE",
            "clarification": None,
            "max_rounds": True,
            "subtasks": [{"subtask_id": "lookup", "capability": "warehouse_lookup", "input": {"query": "Борисов"}}],
        },
        ensure_ascii=False,
    )
    rejected = asyncio.run(LocalOrchestratorHarness().run_case("bool", request, boolean_rounds, plan_id="p"))
    valid = json.loads(boolean_rounds)
    valid["max_rounds"] = 1
    raw = json.dumps(valid, ensure_ascii=False)
    ungrounded_final = json.dumps(
        {
            "schema_version": "t0006.final.v1",
            "plan_id": "p",
            "response_hash": sha256_text(raw),
            "answer": "Выдуманный ответ",
            "ordered_subtask_ids": ["lookup"],
        },
        ensure_ascii=False,
    )
    fallback = asyncio.run(LocalOrchestratorHarness().run_case("facts", request, raw, ungrounded_final, plan_id="p"))
    assert rejected.executor_calls == 0 and rejected.plan_validation["reason"] == "ROUND_LIMIT_INVALID"
    assert fallback.final_validation["reason"] == "FINAL_SCHEMA_MISMATCH"


def test_general_possible_typo_and_ambiguous_label_require_zero_call_clarification():
    for request in ("Найди склад Карисова", "Покажи гараж"):
        raw = json.dumps(
            {
                "schema_version": PLAN_SCHEMA,
                "plan_id": "p",
                "request_hash": sha256_text(request),
                "state": "EXECUTE",
                "clarification": None,
                "max_rounds": 1,
                "subtasks": [{"subtask_id": "lookup", "capability": "warehouse_lookup", "input": {"query": request}}],
            },
            ensure_ascii=False,
        )
        result = asyncio.run(LocalOrchestratorHarness().run_case("ambiguity", request, raw, plan_id="p"))
        assert result.executor_calls == 0
        assert result.plan_validation["reason"] == "CLARIFICATION_REQUIRED"
