import asyncio
import json

from ai_server.orchestrators.local_harness import FINAL_SCHEMA, PLAN_SCHEMA, LocalOrchestratorHarness, sha256_text


def plan(plan_id, request, state="EXECUTE", subtasks=None, clarification=None):
    return json.dumps({
        "schema_version": PLAN_SCHEMA, "plan_id": plan_id, "request_hash": sha256_text(request),
        "state": state, "clarification": clarification, "max_rounds": 1, "subtasks": subtasks or [],
    }, ensure_ascii=False)


def subtask(subtask_id="lookup", capability="warehouse_lookup", query="Борисов"):
    return {"subtask_id": subtask_id, "capability": capability, "input": {"query": query}}


def final(plan_id, response_hash, result):
    return json.dumps({
        "schema_version": FINAL_SCHEMA, "plan_id": plan_id, "response_hash": response_hash,
        "ordered_subtask_ids": [branch.subtask_id for branch in result.branches],
    }, ensure_ascii=False)


def test_no_model_plan_has_zero_calls_and_never_uses_s02_table():
    result = asyncio.run(LocalOrchestratorHarness().run_case("Q01", "Найди склад Борисова", None, plan_id="p1"))
    assert result.executor_calls == 0
    assert result.plan_validation["reason"] == "MODEL_PLAN_UNAVAILABLE"


def test_invalid_json_unknown_capability_and_forbidden_branch_have_zero_calls():
    request = "Только Bitrix: найди склад Борисова"
    harness = LocalOrchestratorHarness()
    invalid = asyncio.run(harness.run_case("x", request, "not json", plan_id="p1"))
    unknown = asyncio.run(harness.run_case("x", request, plan("p2", request, subtasks=[subtask(capability="other")]), plan_id="p2"))
    forbidden = asyncio.run(harness.run_case("x", request, plan("p3", request, subtasks=[subtask(capability="delivery")]), plan_id="p3"))
    assert [invalid.executor_calls, unknown.executor_calls, forbidden.executor_calls] == [0, 0, 0]
    assert {unknown.plan_validation["reason"], forbidden.plan_validation["reason"]} == {"UNKNOWN_CAPABILITY", "FORBIDDEN_CAPABILITY"}


def test_valid_plan_authoritatively_changes_actual_executor_call_and_correlation():
    request = "Покажи склад Борисова и что внутри"
    harness = LocalOrchestratorHarness()
    raw = plan("p1", request, subtasks=[subtask("contents", "contents_stock", "Борисов")])
    first = asyncio.run(harness.run_case("Q02", request, raw, plan_id="p1"))
    raw2 = plan("p2", request, subtasks=[subtask("lookup", "warehouse_lookup", "Борисов")])
    second = asyncio.run(harness.run_case("Q02", request, raw2, plan_id="p2"))
    assert first.branches[0].executor == second.branches[0].executor == "bitrix"
    assert first.branches[0].answer != second.branches[0].answer
    record = harness.executor_calls[0]
    assert record["plan_id"] == first.correlation_ids["plan_id"]
    assert record["response_hash"] == first.correlation_ids["response_hash"]
    assert record["subtask_id"] == first.branches[0].subtask_id
    assert record["attempt_id"] == first.branches[0].attempt_id


def test_clarification_is_successful_zero_call_then_resumes_same_task():
    request = "Найди склад Карисова"
    harness = LocalOrchestratorHarness()
    initial = asyncio.run(harness.run_case("Q07", request, plan("p1", request, "CLARIFICATION_REQUIRED", clarification="Вы имели в виду Карасева?"), plan_id="p1"))
    user_answer = "Да"
    continuation_request = f"{request}\nОтвет пользователя: {user_answer}"
    resumed = asyncio.run(harness.resume(initial.task_id, user_answer, plan("p2", continuation_request, subtasks=[subtask(query="Карасев")]), plan_id="p2"))
    assert initial.verdict == "CLARIFICATION_REQUIRED" and initial.executor_calls == 0
    assert resumed.task_id == initial.task_id and resumed.executor_calls == 1


def test_final_completeness_guard_uses_truthful_deterministic_fallback():
    request = "Найди склад Борисова"
    raw = plan("p1", request, subtasks=[subtask()])
    malformed_final = json.dumps({"answer": "invented"})
    result = asyncio.run(LocalOrchestratorHarness().run_case("Q01", request, raw, malformed_final, plan_id="p1"))
    assert result.final_validation["status"] == "FALLBACK"
    assert "Борисов А.А." in result.final_response
