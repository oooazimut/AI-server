"""Strict model-plan orchestration for the live worker path.

The model is an untrusted planner: it returns a constrained JSON document, while
this module binds it to the inbound request and deterministically decides whether
any specialist can be called.  There is intentionally no keyword-route fallback.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ModelUsageRecord, ToolResult, ToolStatus
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.orchestrators.tools.call_specialist import CallSpecialistTool
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient

PLAN_SCHEMA = "t0006.plan.v1"
FINAL_SCHEMA = "t0006.final.v1"


class PlanAuthoritativeLLM(Protocol):
    async def plan(self, *, manifest: AgentManifest, task: AgentTask, catalog: dict[str, Any], constraints: dict[str, Any]) -> tuple[str, ModelUsageRecord]: ...
    async def finalize(self, *, manifest: AgentManifest, task: AgentTask, plan_id: str, response_hash: str, results: list[dict[str, Any]]) -> tuple[str, ModelUsageRecord]: ...


class DeepSeekPlanService:
    """The only live model surface used by the S04 orchestrator."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def plan(self, *, manifest: AgentManifest, task: AgentTask, catalog: dict[str, Any], constraints: dict[str, Any]) -> tuple[str, ModelUsageRecord]:
        completion = await self.client.complete(
            agent_id=manifest.id,
            json_mode=True,
            messages=[
                {"role": "system", "content": "Return only strict JSON. You are an untrusted planner; do not call tools or compose a user answer."},
                {"role": "user", "content": json.dumps({
                    "schema_version": PLAN_SCHEMA, "plan_id": constraints["plan_id"], "request": task.request,
                    "request_hash": constraints["request_hash"], "capability_catalog": catalog,
                    "hard_constraints": {k: v for k, v in constraints.items() if k not in {"plan_id", "request_hash"}},
                    "required_response": {"schema_version": PLAN_SCHEMA, "plan_id": constraints["plan_id"], "request_hash": constraints["request_hash"], "state": "EXECUTE|CLARIFICATION_REQUIRED|CATALOG|NOT_SUPPORTED", "clarification": "string or null", "max_rounds": "integer 1..3", "subtasks": [{"subtask_id": "unique", "specialist_id": "catalog id", "request": "bounded request"}]},
                }, ensure_ascii=False)},
            ],
        )
        return completion.content, completion.model_usage

    async def finalize(self, *, manifest: AgentManifest, task: AgentTask, plan_id: str, response_hash: str, results: list[dict[str, Any]]) -> tuple[str, ModelUsageRecord]:
        completion = await self.client.complete(
            agent_id=manifest.id,
            json_mode=True,
            messages=[
                {"role": "system", "content": "Return only strict JSON. You may order, but never alter, executor facts."},
                {"role": "user", "content": json.dumps({"schema_version": FINAL_SCHEMA, "plan_id": plan_id, "response_hash": response_hash, "executor_results": results, "required_response": {"schema_version": FINAL_SCHEMA, "plan_id": plan_id, "response_hash": response_hash, "ordered_subtask_ids": "all IDs exactly once"}}, ensure_ascii=False)},
            ],
        )
        return completion.content, completion.model_usage


class PlanRejected(ValueError):
    pass


@dataclass(frozen=True)
class Subtask:
    subtask_id: str
    specialist_id: str
    request: str


@dataclass(frozen=True)
class Plan:
    plan_id: str
    state: str
    clarification: str | None
    subtasks: list[Subtask]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _constraints(request: str, catalog: dict[str, Any]) -> dict[str, Any]:
    text = request.casefold()
    only_bitrix = "только bitrix" in text or "только битрикс" in text
    return {
        "only_source": "bitrix24" if only_bitrix else None,
        "allowed_specialists": sorted(catalog),
        "max_subtasks": 8,
        "max_round_trips": 3,
    }


def _decode_plan(raw: str, *, plan_id: str, request: str, constraints: dict[str, Any]) -> Plan:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PlanRejected("INVALID_JSON") from exc
    required = {"schema_version", "plan_id", "request_hash", "state", "clarification", "max_rounds", "subtasks"}
    if not isinstance(value, dict) or set(value) != required:
        raise PlanRejected("PLAN_SCHEMA_MISMATCH")
    if value["schema_version"] != PLAN_SCHEMA or value["plan_id"] != plan_id or value["request_hash"] != _hash(request):
        raise PlanRejected("PLAN_BINDING_MISMATCH")
    state = value["state"]
    if state not in {"EXECUTE", "CLARIFICATION_REQUIRED", "CATALOG", "NOT_SUPPORTED"}:
        raise PlanRejected("PLAN_STATE_INVALID")
    if type(value["max_rounds"]) is not int or not 1 <= value["max_rounds"] <= constraints["max_round_trips"]:
        raise PlanRejected("ROUND_LIMIT_INVALID")
    clarification = value["clarification"]
    if clarification is not None and (not isinstance(clarification, str) or not clarification.strip()):
        raise PlanRejected("CLARIFICATION_INVALID")
    items = value["subtasks"]
    if not isinstance(items, list) or len(items) > constraints["max_subtasks"]:
        raise PlanRejected("SUBTASK_COUNT_INVALID")
    if state == "EXECUTE" and not items:
        raise PlanRejected("EXECUTE_WITHOUT_SUBTASK")
    if state != "EXECUTE" and (items or (state == "CLARIFICATION_REQUIRED" and not clarification) or (state != "CLARIFICATION_REQUIRED" and clarification is not None)):
        raise PlanRejected("NON_EXECUTION_PLAN_INVALID")
    seen: set[str] = set()
    subtasks: list[Subtask] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {"subtask_id", "specialist_id", "request"}:
            raise PlanRejected("SUBTASK_SCHEMA_MISMATCH")
        subtask_id = item["subtask_id"]
        specialist_id = item["specialist_id"]
        subrequest = item["request"]
        if not isinstance(subtask_id, str) or not subtask_id or subtask_id in seen:
            raise PlanRejected("SUBTASK_ID_INVALID")
        if not isinstance(specialist_id, str) or specialist_id not in constraints["allowed_specialists"]:
            raise PlanRejected("FORBIDDEN_SPECIALIST")
        if constraints["only_source"] and specialist_id != "bitrix24":
            raise PlanRejected("SOURCE_RESTRICTION_VIOLATION")
        if not isinstance(subrequest, str) or not subrequest.strip():
            raise PlanRejected("SUBTASK_REQUEST_INVALID")
        seen.add(subtask_id)
        subtasks.append(Subtask(subtask_id, specialist_id, subrequest))
    return Plan(plan_id, state, clarification, subtasks)


class PlanAuthoritativeOrchestrator(InternalOrchestrator):
    """Live replacement selected by ``InternalOrchestrator.build`` for S04."""

    def __init__(self, *args: Any, planner: PlanAuthoritativeLLM, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._planner = planner

    @classmethod
    def build(cls, manifest: AgentManifest | None, **kwargs: Any) -> "PlanAuthoritativeOrchestrator":
        from ai_server.specialists import build_specialist_registry
        from ai_server.orchestrators.tools import ManageSuspendedTool, ScheduleTaskTool

        manifests = kwargs.pop("manifests", None) or []
        planner = kwargs.pop("orchestrator_llm")
        store = kwargs.pop("orchestrator_store", None)
        retriever = kwargs.pop("orchestrator_retriever", None)
        channels = kwargs.pop("channels", None)
        footer_service = kwargs.pop("footer_service", None)
        result_publisher = kwargs.pop("result_publisher", None)
        if not kwargs.get("bitrix_bot"):
            kwargs["bitrix_bot"] = kwargs.get("bitrix_client")
        specialists = build_specialist_registry(manifests, audience="employee", **{k: v for k, v in kwargs.items() if v is not None})
        call = CallSpecialistTool(specialists, manifests, scheduler=kwargs.get("scheduler"), store=store)
        from ai_server.orchestrators.internal import _dummy_manifest
        instance = cls(
            manifest or _dummy_manifest(),
            agent_tools=[call, ManageSuspendedTool(store=store), ScheduleTaskTool(scheduler=kwargs.get("scheduler"))],
            llm=planner, store=store, scheduler=kwargs.get("scheduler"), retriever=retriever,
            channels=channels, footer_service=footer_service, result_publisher=result_publisher,
            conversation_trace=kwargs.get("conversation_trace"), dialog_guard=kwargs.get("dialog_guard"), planner=planner,
        )
        call.schedule_fn = instance._apply_scheduled_tasks_from_specialist
        return instance

    def _catalog(self) -> dict[str, dict[str, Any]]:
        call = self._tool_registry.get("call_specialist")
        if not isinstance(call, CallSpecialistTool):
            return {}
        return {
            agent_id: {"description": str(manifest.handoff_description or manifest.name)}
            for agent_id, manifest in call._manifests.items()
            if agent_id in call._specialists
        }

    async def handle(self, task: AgentTask) -> AgentResult:
        started = time.monotonic()
        dialog_key = str(task.context.get("dialog_key") or "")
        active = False
        if self._dialog_guard is not None and dialog_key:
            generation = await self._dialog_guard.mark_active(task, ttl_seconds=3600)
            task = task.model_copy(update={"context": {**task.context, "dialog_cancel_generation": int(generation)}})
            active = True
        try:
            catalog = self._catalog()
            constraints = _constraints(task.request, catalog)
            plan_id = f"plan-{uuid.uuid4().hex}"
            raw, usage = await self._planner.plan(manifest=self.manifest, task=task, catalog=catalog, constraints={**constraints, "plan_id": plan_id, "request_hash": _hash(task.request)})
            response_hash = _hash(raw)
            try:
                plan = _decode_plan(raw, plan_id=plan_id, request=task.request, constraints=constraints)
            except PlanRejected as exc:
                result = self._terminal(task, "failed", "Не удалось безопасно подтвердить план обработки запроса.", usage, {"reason": str(exc), "response_hash": response_hash, "plan_id": plan_id})
            else:
                result = await self._execute(task, plan, response_hash, usage)
            result = result.model_copy(update={"metadata": {**result.metadata, "total_ms": round((time.monotonic() - started) * 1000, 1)}})
            await self._send_to_channel(task, result)
            await self._publish_result(task, result)
            return result
        finally:
            if active and self._dialog_guard is not None:
                await self._dialog_guard.clear_active(task)

    def _terminal(self, task: AgentTask, status: str, answer: str, usage: ModelUsageRecord, metadata: dict[str, Any]) -> AgentResult:
        return AgentResult(status=status, agent_id=self.manifest.id, answer=answer, model_usage=[usage], actions_taken=[ActionRecord(name="plan_validation", status="rejected", details=metadata)], metadata=metadata)

    async def _execute(self, task: AgentTask, plan: Plan, response_hash: str, usage: ModelUsageRecord) -> AgentResult:
        base_meta = {"plan_id": plan.plan_id, "response_hash": response_hash, "plan_state": plan.state}
        if plan.state == "CLARIFICATION_REQUIRED":
            return AgentResult(status="needs_clarification", agent_id=self.manifest.id, answer=plan.clarification or "Уточните запрос.", model_usage=[usage], actions_taken=[ActionRecord(name="plan_validation", status="ok", details=base_meta)], metadata=base_meta)
        if plan.state == "NOT_SUPPORTED":
            return AgentResult(status="completed", agent_id=self.manifest.id, answer="Запрос не входит в активный каталог возможностей.", model_usage=[usage], actions_taken=[ActionRecord(name="plan_validation", status="ok", details=base_meta)], metadata=base_meta)
        if plan.state == "CATALOG":
            return AgentResult(status="completed", agent_id=self.manifest.id, answer="Доступны только подтверждённые возможности активных специалистов.", model_usage=[usage], actions_taken=[ActionRecord(name="plan_validation", status="ok", details=base_meta)], metadata=base_meta)
        call = self._tool_registry.get("call_specialist")
        if not isinstance(call, CallSpecialistTool):
            return self._terminal(task, "failed", "Исполнитель специалистов недоступен.", usage, {**base_meta, "reason": "CALL_TOOL_UNAVAILABLE"})

        async def run(subtask: Subtask) -> tuple[Subtask, ToolResult]:
            value = await call.execute_with_task({"specialist_id": subtask.specialist_id, "request": subtask.request}, task=task)
            return subtask, value

        completed = await asyncio.gather(*(run(item) for item in plan.subtasks), return_exceptions=True)
        facts: list[dict[str, Any]] = []
        actions: list[ActionRecord] = [ActionRecord(name="plan_validation", status="ok", details=base_meta)]
        for item, value in zip(plan.subtasks, completed, strict=True):
            attempt_id = f"attempt-{uuid.uuid4().hex}"
            if isinstance(value, Exception):
                facts.append({"subtask_id": item.subtask_id, "attempt_id": attempt_id, "status": "failed", "answer": "Специалист не завершил обработку."})
                actions.append(ActionRecord(name="call_specialist", status="error", details={**base_meta, "subtask_id": item.subtask_id, "attempt_id": attempt_id, "specialist_id": item.specialist_id}))
                continue
            _, tool = value
            data = tool.data if isinstance(tool.data, dict) else {}
            facts.append({"subtask_id": item.subtask_id, "attempt_id": attempt_id, "status": str(data.get("status") or tool.status), "answer": str(data.get("answer") or tool.error or "")})
            actions.append(ActionRecord(name="call_specialist", status=str(tool.status), details={**base_meta, "subtask_id": item.subtask_id, "attempt_id": attempt_id, "specialist_id": item.specialist_id}))
        try:
            raw_final, final_usage = await self._planner.finalize(manifest=self.manifest, task=task, plan_id=plan.plan_id, response_hash=response_hash, results=facts)
            answer = self._decode_final(raw_final, plan.plan_id, response_hash, facts)
            usages = [usage, final_usage]
        except Exception:
            answer = "; ".join(str(item["answer"]).strip() for item in facts if str(item["answer"]).strip()) or "Не удалось получить подтверждённый результат специалистов."
            usages = [usage]
            actions.append(ActionRecord(name="final_validation", status="fallback", details=base_meta))
        status = "completed" if any(item["status"] == "completed" for item in facts) else "failed"
        return AgentResult(status=status, agent_id=self.manifest.id, answer=answer, model_usage=usages, actions_taken=actions, handoff_to=sorted({item.specialist_id for item in plan.subtasks}), metadata={**base_meta, "branches": facts})

    @staticmethod
    def _decode_final(raw: str, plan_id: str, response_hash: str, facts: list[dict[str, Any]]) -> str:
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != {"schema_version", "plan_id", "response_hash", "ordered_subtask_ids"}:
            raise PlanRejected("FINAL_SCHEMA_MISMATCH")
        if data["schema_version"] != FINAL_SCHEMA or data["plan_id"] != plan_id or data["response_hash"] != response_hash:
            raise PlanRejected("FINAL_BINDING_MISMATCH")
        ordered = data["ordered_subtask_ids"]
        known = {str(item["subtask_id"]): str(item["answer"]) for item in facts}
        if not isinstance(ordered, list) or set(ordered) != set(known) or len(ordered) != len(set(ordered)):
            raise PlanRejected("FINAL_COMPLETENESS_FAILED")
        return "; ".join(known[item] for item in ordered if known[item].strip()) or "Специалисты не вернули содержательный результат."
