"""Authoritative-plan local harness used by T-0006 S03.

DeepSeek supplies text only.  This module hashes, parses and validates its strict
JSON plan before a deterministic fixture executor may be reached.  It deliberately
has no network or persistent-service dependency.
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from ai_server.agent_queue_utils import agent_queue_partition_key
from ai_server.models import AgentTask

PLAN_SCHEMA = "t0006.plan.v1"
FINAL_SCHEMA = "t0006.final.v1"
CATALOG = {
    "warehouse_lookup": {"executor": "bitrix", "description": "read approved warehouse identity fixture"},
    "contents_stock": {"executor": "bitrix", "description": "read scripted contents or stock fixture"},
    "delivery": {"executor": "logistics", "description": "read scripted delivery fixture"},
}
WAREHOUSES = {
    "борисов": {"id": "19", "title": "Борисов А.А.", "active": "Y"},
    "гараж": {"id": "3", "title": "Гараж", "active": "Y"},
    "карасев": {"id": "23", "title": "Карасев А.В.", "active": "Y"},
}
AMBIGUOUS_LABELS = {"гараж": ("Гараж", "Гараж Смородин")}
KNOWN_ENTITY_FORMS = ("борисов", "борисова", "гараж", "карасев", "карасева")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_constraints(request: str, *, clarification_resolved: bool = False) -> dict[str, Any]:
    text = request.casefold()
    only_bitrix = "только bitrix" in text or "только битрикс" in text
    requested: list[str] = []
    if any(token in text for token in ("склад", "гараж", "карисов", "карасев", "борисов", "иванов", "петров")):
        requested.append("warehouse_lookup")
    if any(token in text for token in ("остат", "внутри", "содержим")):
        requested.append("contents_stock")
    if any(token in text for token in ("достав", "логист")):
        requested.append("delivery")
    allowed = [capability for capability in requested if not (only_bitrix and capability == "delivery")]
    tokens = ["".join(char for char in token if char.isalpha()) for token in text.split()]
    known_names = KNOWN_ENTITY_FORMS
    possible_typo = next((token for token in tokens if token and token not in known_names and difflib.get_close_matches(token, known_names, n=1, cutoff=0.70)), None)
    ambiguous = next((token for token in tokens if token in AMBIGUOUS_LABELS), None)
    if ambiguous and (any(char.isdigit() for char in text) or "смородин" in text):
        ambiguous = None
    clarification_reason = None
    if possible_typo and not clarification_resolved:
        clarification_reason = "possible_entity_typo"
    elif ambiguous and not clarification_resolved:
        clarification_reason = "ambiguous_entity_label"
    elif "активн" in text and "задач" in text:
        clarification_reason = "active_task_conflict"
    return {"allowed_capabilities": allowed, "only_source": "bitrix" if only_bitrix else None, "max_subtasks": 8, "max_round_trips": 3, "clarification_required": clarification_reason}


def planner_prompt(*, plan_id: str, request: str, dialog_history: list[dict[str, str]], fixture_mode: str = "normal", clarification_resolved: bool = False) -> dict[str, Any]:
    constraints = source_constraints(request, clarification_resolved=clarification_resolved)
    return {
        "schema_version": PLAN_SCHEMA,
        "plan_id": plan_id,
        "request": request,
        "request_hash": sha256_text(request),
        "dialog_history": dialog_history,
        "capability_catalog": CATALOG,
        "hard_constraints": constraints,
        "fixture_mode": fixture_mode,
        "required_response": {
            "schema_version": PLAN_SCHEMA,
            "plan_id": plan_id,
            "request_hash": sha256_text(request),
            "state": "EXECUTE|CLARIFICATION_REQUIRED|CATALOG|NOT_SUPPORTED",
            "clarification": "string or null",
            "max_rounds": "integer 1..3",
            "subtasks": [{"subtask_id": "unique id", "capability": "catalog id", "input": {"query": "string"}}],
        },
    }


def final_prompt(*, plan_id: str, response_hash: str, request: str, results: list["BranchResult"]) -> dict[str, Any]:
    return {
        "schema_version": FINAL_SCHEMA,
        "plan_id": plan_id,
        "response_hash": response_hash,
        "request": request,
        "executor_results": [asdict(result) for result in results],
        "required_response": {"schema_version": FINAL_SCHEMA, "plan_id": plan_id, "response_hash": response_hash, "ordered_subtask_ids": ["all executed ids exactly once"]},
    }


@dataclass(frozen=True)
class PlanSubtask:
    subtask_id: str
    capability: str
    input: dict[str, str]


@dataclass(frozen=True)
class ExecutionPlan:
    plan_id: str
    request_hash: str
    state: str
    clarification: str | None
    max_rounds: int
    subtasks: list[PlanSubtask]


@dataclass
class BranchResult:
    executor: str
    status: str
    answer: str
    attempt_id: str
    plan_id: str
    response_hash: str
    subtask_id: str


@dataclass
class HarnessResult:
    case_id: str
    task_id: str
    verdict: str
    route: list[str]
    clarification: str | None
    branches: list[BranchResult] = field(default_factory=list)
    final_response: str = ""
    correlation_ids: dict[str, str] = field(default_factory=dict)
    executor_calls: int = 0
    model_calls: int = 0
    model_tokens: int = 0
    plan_validation: dict[str, Any] = field(default_factory=dict)
    final_validation: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    latency_ms: int = 0


class PlanValidationError(ValueError):
    pass


def decode_plan(raw: str, *, plan_id: str, request: str, clarification_resolved: bool = False) -> ExecutionPlan:
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise PlanValidationError("INVALID_JSON") from error
    required = {"schema_version", "plan_id", "request_hash", "state", "clarification", "max_rounds", "subtasks"}
    if not isinstance(data, dict) or set(data) != required:
        raise PlanValidationError("PLAN_SCHEMA_MISMATCH")
    if data["schema_version"] != PLAN_SCHEMA or data["plan_id"] != plan_id or data["request_hash"] != sha256_text(request):
        raise PlanValidationError("PLAN_BINDING_MISMATCH")
    state = data["state"]
    if state not in {"EXECUTE", "CLARIFICATION_REQUIRED", "CATALOG", "NOT_SUPPORTED"}:
        raise PlanValidationError("PLAN_STATE_INVALID")
    clarification = data["clarification"]
    max_rounds = data["max_rounds"]
    if type(max_rounds) is not int or not 1 <= max_rounds <= 3:
        raise PlanValidationError("ROUND_LIMIT_INVALID")
    if clarification is not None and (not isinstance(clarification, str) or not clarification.strip()):
        raise PlanValidationError("CLARIFICATION_INVALID")
    subtasks_raw = data["subtasks"]
    if not isinstance(subtasks_raw, list) or len(subtasks_raw) > 8:
        raise PlanValidationError("SUBTASK_COUNT_INVALID")
    if state == "EXECUTE" and not subtasks_raw:
        raise PlanValidationError("EXECUTE_WITHOUT_SUBTASK")
    if state != "EXECUTE" and (subtasks_raw or (state == "CLARIFICATION_REQUIRED" and not clarification) or (state != "CLARIFICATION_REQUIRED" and clarification is not None)):
        raise PlanValidationError("NON_EXECUTION_PLAN_INVALID")
    constraints = source_constraints(request, clarification_resolved=clarification_resolved)
    if constraints["clarification_required"] and state != "CLARIFICATION_REQUIRED":
        raise PlanValidationError("CLARIFICATION_REQUIRED")
    subtasks: list[PlanSubtask] = []
    seen: set[str] = set()
    for item in subtasks_raw:
        if not isinstance(item, dict) or set(item) != {"subtask_id", "capability", "input"}:
            raise PlanValidationError("SUBTASK_SCHEMA_MISMATCH")
        subtask_id, capability, payload = item["subtask_id"], item["capability"], item["input"]
        if not isinstance(subtask_id, str) or not subtask_id or subtask_id in seen:
            raise PlanValidationError("SUBTASK_ID_INVALID")
        if capability not in CATALOG:
            raise PlanValidationError("UNKNOWN_CAPABILITY")
        if capability not in constraints["allowed_capabilities"]:
            raise PlanValidationError("FORBIDDEN_CAPABILITY")
        if not isinstance(payload, dict) or set(payload) != {"query"} or not isinstance(payload["query"], str) or not payload["query"].strip():
            raise PlanValidationError("SUBTASK_INPUT_INVALID")
        seen.add(subtask_id)
        subtasks.append(PlanSubtask(subtask_id, capability, payload))
    return ExecutionPlan(plan_id, sha256_text(request), state, clarification, max_rounds, subtasks)


def decode_final(raw: str, *, plan: ExecutionPlan, response_hash: str, results: list[BranchResult]) -> str:
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise PlanValidationError("FINAL_INVALID_JSON") from error
    required = {"schema_version", "plan_id", "response_hash", "ordered_subtask_ids"}
    if not isinstance(data, dict) or set(data) != required:
        raise PlanValidationError("FINAL_SCHEMA_MISMATCH")
    if data["schema_version"] != FINAL_SCHEMA or data["plan_id"] != plan.plan_id or data["response_hash"] != response_hash:
        raise PlanValidationError("FINAL_BINDING_MISMATCH")
    executed = {item.subtask_id for item in results}
    ordered = data["ordered_subtask_ids"]
    if not isinstance(ordered, list) or any(not isinstance(item, str) for item in ordered):
        raise PlanValidationError("FINAL_COMPLETENESS_FAILED")
    if len(ordered) != len(set(ordered)) or set(ordered) != executed:
        raise PlanValidationError("FINAL_COMPLETENESS_FAILED")
    by_id = {item.subtask_id: item.answer for item in results}
    return "; ".join(by_id[item] for item in ordered)


class LocalOrchestratorHarness:
    """A deterministic executor reachable only through a validated model plan."""

    def __init__(self) -> None:
        self.executor_calls: list[dict[str, str]] = []
        self.pending: dict[str, tuple[str, str, str]] = {}

    async def run_case(self, case_id: str, request: str, planner_response: str | None, final_response: str | None = None, *, fixture_mode: str = "normal", dialog_history: list[dict[str, str]] | None = None, task_id: str | None = None, plan_id: str | None = None, clarification_resolved: bool = False) -> HarnessResult:
        started = time.monotonic()
        task_id = task_id or f"T0006-S03-{case_id}-{uuid.uuid4().hex[:8]}"
        plan_id = plan_id or f"plan-{uuid.uuid4().hex}"
        response_hash = sha256_text(planner_response or "")
        task = AgentTask(task_id=task_id, request=request, user={"id": "u1"}, context={"dialog_key": f"chat:local:user:{case_id}"})
        correlation = {"task_id": task.task_id, "partition": agent_queue_partition_key({"payload": task.model_dump()}), "plan_id": plan_id, "response_hash": response_hash}
        result = HarnessResult(case_id, task_id, "FAILED", [], None, correlation_ids=correlation)
        if planner_response is None:
            result.plan_validation = {"status": "REJECTED", "reason": "MODEL_PLAN_UNAVAILABLE"}
            result.notes.append("No plan means no dispatch; S02 scenario routing is not consulted.")
            result.latency_ms = round((time.monotonic() - started) * 1000)
            return result
        result.model_calls = 1
        try:
            plan = decode_plan(planner_response, plan_id=plan_id, request=request, clarification_resolved=clarification_resolved)
        except PlanValidationError as error:
            result.plan_validation = {"status": "REJECTED", "reason": str(error)}
            result.notes.append("Rejected plan made zero executor calls.")
            result.latency_ms = round((time.monotonic() - started) * 1000)
            return result
        result.plan_validation = {"status": "ACCEPT", "plan_id": plan.plan_id, "response_hash": response_hash}
        result.route = [CATALOG[item.capability]["executor"] for item in plan.subtasks]
        result.clarification = plan.clarification
        if plan.state == "CLARIFICATION_REQUIRED":
            self.pending[task_id] = (case_id, request, fixture_mode)
            result.verdict, result.final_response = "CLARIFICATION_REQUIRED", plan.clarification or ""
            result.notes.append("Clarification state is successful and has zero executor calls.")
            result.latency_ms = round((time.monotonic() - started) * 1000)
            return result
        if plan.state == "NOT_SUPPORTED":
            result.verdict, result.final_response = "NOT_SUPPORTED", "Запрос не входит в активный каталог возможностей."
            result.latency_ms = round((time.monotonic() - started) * 1000)
            return result
        if plan.state == "CATALOG":
            result.verdict, result.final_response = "PASS", "Доступно: " + ", ".join(CATALOG) + "."
            result.latency_ms = round((time.monotonic() - started) * 1000)
            return result
        for round_no in range(1, plan.max_rounds + 1):
            branch_offset = (round_no - 1) * len(plan.subtasks)
            current = list(await asyncio.gather(*(self._execute(plan, response_hash, item, fixture_mode, branch_offset + index + 1) for index, item in enumerate(plan.subtasks))))
            result.branches.extend(current)
            if not all(item.status == "not_mine" for item in current):
                break
        result.executor_calls = len(result.branches)
        if final_response is None:
            result.verdict = "PARTIAL"
            result.final_response = self._truthful_fallback(result.branches)
            result.final_validation = {"status": "FALLBACK", "reason": "FINAL_MODEL_UNAVAILABLE"}
        else:
            result.model_calls += 1
            try:
                result.final_response = decode_final(final_response, plan=plan, response_hash=response_hash, results=result.branches)
                result.final_validation = {"status": "ACCEPT"}
                result.verdict = "PASS" if all(item.status == "ok" for item in result.branches) else "PARTIAL"
            except PlanValidationError as error:
                result.verdict, result.final_response = "PARTIAL", self._truthful_fallback(result.branches)
                result.final_validation = {"status": "FALLBACK", "reason": str(error)}
        result.latency_ms = round((time.monotonic() - started) * 1000)
        return result

    async def resume(self, task_id: str, user_answer: str, planner_response: str, final_response: str | None = None, *, plan_id: str | None = None) -> HarnessResult:
        case_id, request, fixture_mode = self.pending.pop(task_id)
        continuation_request = f"{request}\nОтвет пользователя: {user_answer}"
        return await self.run_case(case_id, continuation_request, planner_response, final_response, fixture_mode=fixture_mode, dialog_history=[{"role": "user", "content": request}, {"role": "user", "content": user_answer}], task_id=task_id, plan_id=plan_id, clarification_resolved=True)

    async def _execute(self, plan: ExecutionPlan, response_hash: str, subtask: PlanSubtask, fixture_mode: str, index: int) -> BranchResult:
        await asyncio.sleep(0)
        executor = CATALOG[subtask.capability]["executor"]
        attempt_id = f"{plan.plan_id}:attempt:{index}"
        self.executor_calls.append({"plan_id": plan.plan_id, "response_hash": response_hash, "subtask_id": subtask.subtask_id, "attempt_id": attempt_id, "executor": executor})
        query = subtask.input["query"].casefold()
        if fixture_mode == "timeout":
            return BranchResult(executor, "timeout", "scripted local adapter timeout", attempt_id, plan.plan_id, response_hash, subtask.subtask_id)
        if fixture_mode == "not_mine":
            return BranchResult(executor, "not_mine", "scripted executor says not mine", attempt_id, plan.plan_id, response_hash, subtask.subtask_id)
        if fixture_mode == "one_error" and index == 2:
            return BranchResult(executor, "error", "scripted local adapter failure", attempt_id, plan.plan_id, response_hash, subtask.subtask_id)
        if executor == "logistics":
            answer = "scripted delivery: route requires confirmation"
        elif subtask.capability == "contents_stock":
            answer = "scripted contents (not Bitrix facts): cable=12, fasteners=20"
        else:
            item = next((record for name, record in WAREHOUSES.items() if name in query), None)
            answer = f"{item['title']} (id {item['id']}, active {item['active']})" if item else "not present in cleaned warehouse fixture"
        status = "ok" if "not present" not in answer else "not_found"
        return BranchResult(executor, status, answer, attempt_id, plan.plan_id, response_hash, subtask.subtask_id)

    @staticmethod
    def _truthful_fallback(results: list[BranchResult]) -> str:
        successful = [item.answer for item in results if item.status == "ok"]
        failed = [item.subtask_id for item in results if item.status != "ok"]
        answer = "; ".join(successful) if successful else "Нет подтверждённых результатов исполнителей."
        return answer + (f" Не завершены ветки: {', '.join(failed)}." if failed else "")
