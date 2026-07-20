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

from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ModelUsageRecord, ToolResult
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.orchestrators.tools.call_specialist import CallSpecialistTool

PLAN_SCHEMA = "t0006.plan.v1"
FINAL_SCHEMA = "t0006.final.v1"
REPAIRABLE_PLAN_REJECTIONS = frozenset({"INVALID_JSON", "PLAN_SCHEMA_MISMATCH", "PLAN_BINDING_MISMATCH"})


class PlanAuthoritativeLLM(Protocol):
    async def plan(
        self, *, manifest: AgentManifest, task: AgentTask, catalog: dict[str, Any], constraints: dict[str, Any]
    ) -> tuple[str, ModelUsageRecord]: ...
    async def finalize(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        plan_id: str,
        response_hash: str,
        results: list[dict[str, Any]],
    ) -> tuple[str, ModelUsageRecord]: ...


class DeepSeekPlanService:
    """The only live model surface used by the S04 orchestrator."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def plan(
        self, *, manifest: AgentManifest, task: AgentTask, catalog: dict[str, Any], constraints: dict[str, Any]
    ) -> tuple[str, ModelUsageRecord]:
        repair_reason = str(constraints.get("repair_reason") or "")
        payload: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA,
            "plan_id": constraints["plan_id"],
            "request": task.request,
            "request_hash": constraints["request_hash"],
            "capability_catalog": catalog,
            "hard_constraints": {
                key: value
                for key, value in constraints.items()
                if key not in {"plan_id", "request_hash", "repair_reason", "repair_attempt"}
            },
            "required_response": {
                "schema_version": PLAN_SCHEMA,
                "plan_id": constraints["plan_id"],
                "request_hash": constraints["request_hash"],
                "state": "EXECUTE|CLARIFICATION_REQUIRED|CATALOG|NOT_SUPPORTED",
                "clarification": "string or null",
                "max_rounds": "integer 1..3",
                "subtasks": [
                    {
                        "subtask_id": "unique",
                        "specialist_id": "catalog id",
                        "request": "bounded request",
                    }
                ],
            },
        }
        if repair_reason:
            payload["repair_instruction"] = {
                "attempt": int(constraints.get("repair_attempt") or 2),
                "previous_rejection": repair_reason,
                "instruction": (
                    "The previous response was rejected before dispatch. Return one fresh JSON object "
                    "with exactly the required_response keys and exact binding values."
                ),
            }
        completion = await self.client.complete(
            agent_id=manifest.id,
            json_mode=True,
            messages=[
                {
                    "role": "system",
                    "content": "Return only strict JSON. You are an untrusted planner; do not call tools or compose a user answer.",
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        return completion.content, completion.model_usage

    async def finalize(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        plan_id: str,
        response_hash: str,
        results: list[dict[str, Any]],
    ) -> tuple[str, ModelUsageRecord]:
        completion = await self.client.complete(
            agent_id=manifest.id,
            json_mode=True,
            messages=[
                {
                    "role": "system",
                    "content": "Return only strict JSON. You may order, but never alter, executor facts.",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "schema_version": FINAL_SCHEMA,
                            "plan_id": plan_id,
                            "response_hash": response_hash,
                            "executor_results": results,
                            "required_response": {
                                "schema_version": FINAL_SCHEMA,
                                "plan_id": plan_id,
                                "response_hash": response_hash,
                                "ordered_subtask_ids": "all IDs exactly once",
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
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
    if state != "EXECUTE" and (
        items
        or (state == "CLARIFICATION_REQUIRED" and not clarification)
        or (state != "CLARIFICATION_REQUIRED" and clarification is not None)
    ):
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
    def build(cls, manifest: AgentManifest | None, **kwargs: Any) -> PlanAuthoritativeOrchestrator:
        from ai_server.orchestrators.tools import ManageSuspendedTool, ScheduleTaskTool
        from ai_server.specialists import build_specialist_registry

        manifests = kwargs.pop("manifests", None) or []
        planner = kwargs.pop("orchestrator_llm")
        store = kwargs.pop("orchestrator_store", None)
        retriever = kwargs.pop("orchestrator_retriever", None)
        channels = kwargs.pop("channels", None)
        footer_service = kwargs.pop("footer_service", None)
        result_publisher = kwargs.pop("result_publisher", None)
        if not kwargs.get("bitrix_bot"):
            kwargs["bitrix_bot"] = kwargs.get("bitrix_client")
        specialists = build_specialist_registry(
            manifests, audience="employee", **{k: v for k, v in kwargs.items() if v is not None}
        )
        call = CallSpecialistTool(specialists, manifests, scheduler=kwargs.get("scheduler"), store=store)
        from ai_server.orchestrators.internal import _dummy_manifest

        instance = cls(
            manifest or _dummy_manifest(),
            agent_tools=[call, ManageSuspendedTool(store=store), ScheduleTaskTool(scheduler=kwargs.get("scheduler"))],
            llm=planner,
            store=store,
            scheduler=kwargs.get("scheduler"),
            retriever=retriever,
            channels=channels,
            footer_service=footer_service,
            result_publisher=result_publisher,
            conversation_trace=kwargs.get("conversation_trace"),
            dialog_guard=kwargs.get("dialog_guard"),
            planner=planner,
            outbound_queue=kwargs.get("outbound_queue"),
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

    async def _load_authoritative_pending_specialist(self, task: AgentTask) -> tuple[AgentTask, str | None]:
        """Load dialog continuation only from durable KV, never inbound context."""
        context = {key: value for key, value in task.context.items() if key != "pending_specialist"}
        task = task.model_copy(update={"context": context})
        dialog_key = str(context.get("dialog_key") or "")
        if self.store is None or not dialog_key or not hasattr(self.store, "get_kv"):
            return task, None
        pending = await self.store.get_kv(dialog_key, "pending_specialist")  # type: ignore[attr-defined]
        if pending:
            task = task.model_copy(update={"context": {**context, "pending_specialist": str(pending)}})
        return task, None

    async def handle(self, task: AgentTask) -> AgentResult:
        started = time.monotonic()
        dialog_key = str(task.context.get("dialog_key") or "")
        active = False
        if self._dialog_guard is not None and dialog_key:
            generation = await self._dialog_guard.mark_active(task, ttl_seconds=3600)
            task = task.model_copy(update={"context": {**task.context, "dialog_cancel_generation": int(generation)}})
            active = True
        try:
            try:
                task, _ = await self._load_authoritative_pending_specialist(task)
            except Exception:
                usage = ModelUsageRecord(
                    agent_id=self.manifest.id,
                    provider="internal",
                    model="pending-specialist-state-machine",
                    status="not_used",
                    notes=["No model call: durable pending-specialist state was unavailable."],
                )
                result = self._terminal(
                    task,
                    "failed",
                    "Не удалось безопасно проверить состояние текущего диалога.",
                    usage,
                    {"reason": "PENDING_STATE_READ_FAILED", "route": "pending_specialist"},
                )
                result = result.model_copy(
                    update={"metadata": {**result.metadata, "total_ms": round((time.monotonic() - started) * 1000, 1)}}
                )
                await self._send_to_channel(task, result)
                await self._publish_result(task, result)
                return result
            catalog = self._catalog()
            constraints = _constraints(task.request, catalog)
            plan_id = f"plan-{uuid.uuid4().hex}"
            pending_specialist = str(task.context.get("pending_specialist") or "").strip()
            deterministic_route: str | None = None
            planner_usages: list[ModelUsageRecord] = []
            planner_rejections: list[str] = []
            planner_attempt_audit: list[dict[str, Any]] = []
            plan: Plan | None = None
            if pending_specialist:
                # A pending specialist is durable dialog state, not an LLM routing
                # suggestion.  Bind the continuation to the inbound request and
                # let normal plan validation fail closed if the catalog changed.
                raw = json.dumps(
                    {
                        "schema_version": PLAN_SCHEMA,
                        "plan_id": plan_id,
                        "request_hash": _hash(task.request),
                        "state": "EXECUTE",
                        "clarification": None,
                        "max_rounds": 1,
                        "subtasks": [
                            {
                                "subtask_id": f"pending-{uuid.uuid4().hex}",
                                "specialist_id": pending_specialist,
                                "request": task.request,
                            }
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                usage = ModelUsageRecord(
                    agent_id=self.manifest.id,
                    provider="internal",
                    model="pending-specialist-state-machine",
                    status="not_used",
                    notes=["No model call: continued the dialog-bound pending specialist."],
                )
                planner_usages.append(usage)
                deterministic_route = "pending_specialist"
                response_hash = _hash(raw)
                try:
                    plan = _decode_plan(raw, plan_id=plan_id, request=task.request, constraints=constraints)
                except PlanRejected as exc:
                    planner_rejections.append(str(exc))
            else:
                for attempt in range(1, 3):
                    call_constraints = {
                        **constraints,
                        "plan_id": plan_id,
                        "request_hash": _hash(task.request),
                    }
                    if planner_rejections:
                        call_constraints.update(
                            {
                                "repair_reason": planner_rejections[-1],
                                "repair_attempt": attempt,
                            }
                        )
                    try:
                        raw, usage = await self._planner.plan(
                            manifest=self.manifest,
                            task=task,
                            catalog=catalog,
                            constraints=call_constraints,
                        )
                    except Exception:
                        if attempt == 1:
                            raise
                        reason = "MODEL_REPAIR_UNAVAILABLE"
                        planner_rejections.append(reason)
                        planner_attempt_audit.append({"attempt": attempt, "status": "error", "rejection": reason})
                        break
                    planner_usages.append(usage)
                    response_hash = _hash(raw)
                    attempt_audit = {
                        "attempt": attempt,
                        "response_hash": response_hash,
                        "status": "accepted",
                    }
                    planner_attempt_audit.append(attempt_audit)
                    try:
                        plan = _decode_plan(raw, plan_id=plan_id, request=task.request, constraints=constraints)
                    except PlanRejected as exc:
                        reason = str(exc)
                        attempt_audit.update({"status": "rejected", "rejection": reason})
                        planner_rejections.append(reason)
                        if attempt == 1 and reason in REPAIRABLE_PLAN_REJECTIONS:
                            continue
                    break
            if plan is None:
                reason = planner_rejections[-1] if planner_rejections else "MODEL_PLAN_UNAVAILABLE"
                metadata = {
                    "reason": reason,
                    "response_hash": response_hash,
                    "plan_id": plan_id,
                    "planner_attempts": len(planner_attempt_audit),
                    "planner_rejections": planner_rejections,
                    "planner_attempt_audit": planner_attempt_audit,
                }
                if deterministic_route:
                    metadata.update(
                        {
                            "route": deterministic_route,
                            "pending_specialist_id": pending_specialist,
                        }
                    )
                result = self._terminal(
                    task, "failed", "Не удалось безопасно подтвердить план обработки запроса.", planner_usages, metadata
                )
            else:
                result = await self._execute(
                    task,
                    plan,
                    response_hash,
                    planner_usages,
                    planner_rejections=planner_rejections,
                    planner_attempt_audit=planner_attempt_audit,
                    deterministic_route=deterministic_route,
                )
            result = result.model_copy(
                update={"metadata": {**result.metadata, "total_ms": round((time.monotonic() - started) * 1000, 1)}}
            )
            await self._send_to_channel(task, result)
            await self._publish_result(task, result)
            return result
        finally:
            if active and self._dialog_guard is not None:
                await self._dialog_guard.clear_active(task)

    def _terminal(
        self,
        task: AgentTask,
        status: str,
        answer: str,
        usage: ModelUsageRecord | list[ModelUsageRecord],
        metadata: dict[str, Any],
    ) -> AgentResult:
        model_usage = usage if isinstance(usage, list) else [usage]
        return AgentResult(
            status=status,
            agent_id=self.manifest.id,
            answer=answer,
            model_usage=model_usage,
            actions_taken=[ActionRecord(name="plan_validation", status="rejected", details=metadata)],
            metadata=metadata,
        )

    async def _execute(
        self,
        task: AgentTask,
        plan: Plan,
        response_hash: str,
        planner_usages: list[ModelUsageRecord],
        *,
        planner_rejections: list[str] | None = None,
        planner_attempt_audit: list[dict[str, Any]] | None = None,
        deterministic_route: str | None = None,
    ) -> AgentResult:
        base_meta = {
            "plan_id": plan.plan_id,
            "response_hash": response_hash,
            "plan_state": plan.state,
            "planner_attempts": len(planner_attempt_audit or []),
        }
        if planner_rejections:
            base_meta["planner_rejections"] = list(planner_rejections)
        if planner_attempt_audit:
            base_meta["planner_attempt_audit"] = list(planner_attempt_audit)
        if deterministic_route:
            base_meta["route"] = deterministic_route
        if plan.state == "CLARIFICATION_REQUIRED":
            return AgentResult(
                status="needs_clarification",
                agent_id=self.manifest.id,
                answer=plan.clarification or "Уточните запрос.",
                model_usage=list(planner_usages),
                actions_taken=[ActionRecord(name="plan_validation", status="ok", details=base_meta)],
                metadata=base_meta,
            )
        if plan.state == "NOT_SUPPORTED":
            return AgentResult(
                status="completed",
                agent_id=self.manifest.id,
                answer="Запрос не входит в активный каталог возможностей.",
                model_usage=list(planner_usages),
                actions_taken=[ActionRecord(name="plan_validation", status="ok", details=base_meta)],
                metadata=base_meta,
            )
        if plan.state == "CATALOG":
            return AgentResult(
                status="completed",
                agent_id=self.manifest.id,
                answer="Доступны только подтверждённые возможности активных специалистов.",
                model_usage=list(planner_usages),
                actions_taken=[ActionRecord(name="plan_validation", status="ok", details=base_meta)],
                metadata=base_meta,
            )
        call = self._tool_registry.get("call_specialist")
        if not isinstance(call, CallSpecialistTool):
            return self._terminal(
                task,
                "failed",
                "Исполнитель специалистов недоступен.",
                planner_usages,
                {**base_meta, "reason": "CALL_TOOL_UNAVAILABLE"},
            )

        async def run(subtask: Subtask) -> tuple[Subtask, str, ToolResult]:
            # Correlation is created *before* reaching the specialist, so its
            # own trace and any durable result can be joined to this plan.
            attempt_id = f"attempt-{uuid.uuid4().hex}"
            correlated_task = task.model_copy(
                update={
                    "context": {
                        **task.context,
                        "t0006_plan_id": plan.plan_id,
                        "t0006_response_hash": response_hash,
                        "t0006_subtask_id": subtask.subtask_id,
                        "t0006_attempt_id": attempt_id,
                        # The planner is authorized to choose a specialist, never to
                        # silently rewrite the user's request seen by that specialist.
                        # Keep both values for causal audit while dispatching the
                        # original, validated dialog-bound request.
                        "t0006_original_request": task.request,
                        "t0006_planned_subtask_request": subtask.request,
                    }
                }
            )
            value = await call.execute_with_task(
                {"specialist_id": subtask.specialist_id, "request": subtask.request}, task=correlated_task
            )
            return subtask, attempt_id, value

        completed = await asyncio.gather(*(run(item) for item in plan.subtasks), return_exceptions=True)
        facts: list[dict[str, Any]] = []
        actions: list[ActionRecord] = [ActionRecord(name="plan_validation", status="ok", details=base_meta)]
        if deterministic_route:
            actions.append(ActionRecord(name="pending_specialist_route", status="ok", details=base_meta))
        approvals: list[ActionRecord] = []
        for item, value in zip(plan.subtasks, completed, strict=True):
            if isinstance(value, Exception):
                attempt_id = f"attempt-{uuid.uuid4().hex}"
                facts.append(
                    {
                        "subtask_id": item.subtask_id,
                        "specialist_id": item.specialist_id,
                        "attempt_id": attempt_id,
                        "status": "failed",
                        "answer": f"Источник {item.specialist_id}: не завершил обработку.",
                    }
                )
                actions.append(
                    ActionRecord(
                        name="call_specialist",
                        status="error",
                        details={
                            **base_meta,
                            "subtask_id": item.subtask_id,
                            "attempt_id": attempt_id,
                            "specialist_id": item.specialist_id,
                        },
                    )
                )
                continue
            _, attempt_id, tool = value
            data = tool.data if isinstance(tool.data, dict) else {}
            branch_status = str(data.get("status") or tool.status)
            branch_answer = str(data.get("answer") or tool.error or "").strip()
            if len(plan.subtasks) > 1:
                if branch_status in {"error", "failed"}:
                    branch_answer = "не завершил обработку."
                branch_answer = f"Источник {item.specialist_id}: {branch_answer or 'не вернул результат.'}"
            facts.append(
                {
                    "subtask_id": item.subtask_id,
                    "specialist_id": item.specialist_id,
                    "attempt_id": attempt_id,
                    "status": branch_status,
                    "answer": branch_answer,
                }
            )
            for raw_approval in data.get("actions_requiring_approval", []):
                if isinstance(raw_approval, dict):
                    approvals.append(ActionRecord.model_validate(raw_approval))
            actions.append(
                ActionRecord(
                    name="call_specialist",
                    status=str(tool.status),
                    details={
                        **base_meta,
                        "subtask_id": item.subtask_id,
                        "attempt_id": attempt_id,
                        "specialist_id": item.specialist_id,
                    },
                )
            )
        if deterministic_route:
            answer = (
                "; ".join(str(item["answer"]).strip() for item in facts if str(item["answer"]).strip())
                or "Не удалось получить подтверждённый результат специалистов."
            )
            usages = list(planner_usages)
        else:
            usages = list(planner_usages)
            try:
                raw_final, final_usage = await self._planner.finalize(
                    manifest=self.manifest, task=task, plan_id=plan.plan_id, response_hash=response_hash, results=facts
                )
            except Exception:
                answer = (
                    "; ".join(str(item["answer"]).strip() for item in facts if str(item["answer"]).strip())
                    or "Не удалось получить подтверждённый результат специалистов."
                )
                actions.append(
                    ActionRecord(
                        name="final_validation",
                        status="fallback",
                        details={**base_meta, "reason": "FINAL_MODEL_UNAVAILABLE"},
                    )
                )
            else:
                usages.append(final_usage)
                try:
                    answer = self._decode_final(raw_final, plan.plan_id, response_hash, facts)
                except Exception as exc:
                    reason = str(exc) if isinstance(exc, PlanRejected) else "FINAL_SCHEMA_INVALID"
                    answer = (
                        "; ".join(str(item["answer"]).strip() for item in facts if str(item["answer"]).strip())
                        or "Не удалось получить подтверждённый результат специалистов."
                    )
                    actions.append(
                        ActionRecord(
                            name="final_validation",
                            status="fallback",
                            details={**base_meta, "reason": reason, "final_response_hash": _hash(raw_final)},
                        )
                    )
        branch_statuses = {str(item["status"]) for item in facts}
        if approvals or "needs_human" in branch_statuses:
            status = "needs_human"
        elif "needs_clarification" in branch_statuses:
            status = "needs_clarification"
        elif "completed" in branch_statuses:
            status = "completed"
        else:
            status = "failed"
        return AgentResult(
            status=status,
            agent_id=self.manifest.id,
            answer=answer,
            model_usage=usages,
            actions_taken=actions,
            actions_requiring_approval=approvals,
            handoff_to=sorted({item.specialist_id for item in plan.subtasks}),
            metadata={**base_meta, "branches": facts},
        )

    @staticmethod
    def _decode_final(raw: str, plan_id: str, response_hash: str, facts: list[dict[str, Any]]) -> str:
        data = json.loads(raw)
        if not isinstance(data, dict) or set(data) != {
            "schema_version",
            "plan_id",
            "response_hash",
            "ordered_subtask_ids",
        }:
            raise PlanRejected("FINAL_SCHEMA_MISMATCH")
        if (
            data["schema_version"] != FINAL_SCHEMA
            or data["plan_id"] != plan_id
            or data["response_hash"] != response_hash
        ):
            raise PlanRejected("FINAL_BINDING_MISMATCH")
        ordered = data["ordered_subtask_ids"]
        known = {str(item["subtask_id"]): str(item["answer"]) for item in facts}
        if not isinstance(ordered, list) or set(ordered) != set(known) or len(ordered) != len(set(ordered)):
            raise PlanRejected("FINAL_COMPLETENESS_FAILED")
        return (
            "; ".join(known[item] for item in ordered if known[item].strip())
            or "Специалисты не вернули содержательный результат."
        )
