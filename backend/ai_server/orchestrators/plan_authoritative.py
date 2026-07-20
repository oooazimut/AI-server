"""Strict model-plan orchestration for the live worker path.

The model is an untrusted planner: it returns a constrained JSON document, while
this module binds it to the inbound request and deterministically decides whether
any specialist can be called.  There is intentionally no keyword-route fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from ai_server.agents.bitrix24.draft_confirmation import matches_draft_confirmation
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ModelUsageRecord, ToolResult
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.orchestrators.tools.call_specialist import CallSpecialistTool

PLAN_SCHEMA = "t0007.plan.v1"
FINAL_SCHEMA = "t0006.final.v1"
REPAIRABLE_PLAN_REJECTIONS = frozenset(
    {
        "INVALID_JSON",
        "PLAN_SCHEMA_MISMATCH",
        "PLAN_BINDING_MISMATCH",
        "NON_EXECUTION_PLAN_INVALID",
        "DUPLICATE_SUBTASK",
    }
)

_DRAFT_CONTINUE = "продолжить текущий черновик"
_DRAFT_REPLACE = "отменить текущий черновик и запустить новый запрос"


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", value.casefold())).strip()


def _draft_intent(request: str) -> str | None:
    text = _normalized_text(request)
    if re.search(r"\bнапомни\b|\bкалендар", text):
        return "calendar_event"
    if re.search(r"\bсозда(?:й|ть|йте)\s+проект", text):
        return "project_create"
    if re.search(r"\bзакр(?:ой|ыть|ойте)\s+задач", text):
        return "task_close"
    if re.search(r"\bсозда(?:й|ть|йте)\s+задач", text):
        return "task_create"
    return None


def _draft_discard_request(request: str) -> bool:
    text = _normalized_text(request)
    return any(marker in text for marker in ("отмени черновик", "отменить черновик", "не создавай", "удали черновик"))


def _is_new_request_during_draft(request: str, draft: dict[str, Any]) -> bool:
    if matches_draft_confirmation(request, draft) or _draft_discard_request(request):
        return False
    incoming = _draft_intent(request)
    # A read-only request has no draft intent and must remain available while a
    # draft is waiting for confirmation.  Every new write intent, including a
    # second request of the same type, is held as a replacement candidate: two
    # concurrent drafts in one dialog are never safe.
    return bool(incoming)


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
            "dialog_history": list(task.context.get("dialog_history") or []),
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
                        "segment_id": "explicit segment id or null",
                        "specialist_id": "catalog id",
                        "capability": "catalog capability id",
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
                    "content": (
                        "Return only strict JSON. You are an untrusted planner; do not call tools or compose a user answer. "
                        "Use dialog_history when the current request answers a clarification. For a calendar reminder, "
                        "do not ask for a missing time: use 12:00 Moscow; keep an already stated date and title."
                    ),
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
    segment_id: str | None
    specialist_id: str
    capability: str
    request: str


@dataclass(frozen=True)
class Plan:
    plan_id: str
    state: str
    clarification: str | None
    subtasks: list[Subtask]


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _explicit_segments(request: str, catalog: dict[str, Any]) -> list[dict[str, str]]:
    """Extract explicit named segments, including natural voice-style repetitions."""
    aliases = {
        "bitrix": "bitrix24",
        "битрикс": "bitrix24",
        "bitrix24": "bitrix24",
        "логист": "logistics",
        "logistics": "logistics",
    }
    aliases = {name: target for name, target in aliases.items() if target in catalog}
    segments: list[dict[str, str]] = []
    for index, part in enumerate((item.strip() for item in re.split(r"[;\n]+", request)), start=1):
        if not part:
            continue
        match = re.match(r"^\s*([A-Za-zА-Яа-яЁё0-9_]+)\s*[:,]\s*(.+?)\s*$", part, flags=re.DOTALL)
        if match is None:
            continue
        specialist_id = aliases.get(match.group(1).casefold())
        if specialist_id is not None:
            segments.append(
                {"segment_id": f"segment-{index}", "specialist_id": specialist_id, "request": match.group(2)}
            )
    if len(segments) >= 2:
        return segments

    # Voice transcription commonly removes the colon/semicolon between two
    # explicitly named agents. Split only on a known agent name followed by a
    # verb-like word, so an ordinary mention of a specialist stays untouched.
    names = "|".join(re.escape(name) for name in sorted(aliases, key=len, reverse=True))
    matches = list(
        re.finditer(
            rf"(?<![\w-])(?P<name>{names})(?=\s+(?:покажи|найди|создай|проверь|выведи|дай|расскажи)\b)",
            request,
            flags=re.IGNORECASE,
        )
    )
    if len(matches) < 2:
        return segments
    voice_segments: list[dict[str, str]] = []
    for index, match in enumerate(matches, start=1):
        end = matches[index].start() if index < len(matches) else len(request)
        body = request[match.end() : end].strip(" ,;:-")
        specialist_id = aliases.get(match.group("name").casefold())
        if specialist_id and body:
            voice_segments.append(
                {"segment_id": f"segment-{index}", "specialist_id": specialist_id, "request": body}
            )
    return voice_segments or segments


def _constraints(
    request: str, catalog: dict[str, Any], *, pending_specialist: str | None = None
) -> dict[str, Any]:
    text = request.casefold()
    only_bitrix = "только bitrix" in text or "только битрикс" in text
    return {
        "only_source": "bitrix24" if only_bitrix else None,
        "allowed_specialists": sorted(catalog),
        "capability_catalog": catalog,
        "explicit_segments": _explicit_segments(request, catalog),
        "pending_specialist": pending_specialist or None,
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
    expected_segments = {item["segment_id"]: item for item in constraints.get("explicit_segments", [])}
    seen: set[str] = set()
    seen_segments: set[str] = set()
    seen_dispatches: set[tuple[str, str, str]] = set()
    subtasks: list[Subtask] = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {
            "subtask_id",
            "segment_id",
            "specialist_id",
            "capability",
            "request",
        }:
            raise PlanRejected("SUBTASK_SCHEMA_MISMATCH")
        subtask_id = item["subtask_id"]
        segment_id = item["segment_id"]
        specialist_id = item["specialist_id"]
        capability = item["capability"]
        subrequest = item["request"]
        if not isinstance(subtask_id, str) or not subtask_id or subtask_id in seen:
            raise PlanRejected("SUBTASK_ID_INVALID")
        if segment_id is not None and (not isinstance(segment_id, str) or segment_id in seen_segments):
            raise PlanRejected("SEGMENT_BINDING_INVALID")
        if not isinstance(specialist_id, str) or specialist_id not in constraints["allowed_specialists"]:
            raise PlanRejected("FORBIDDEN_SPECIALIST")
        if constraints["only_source"] and specialist_id != "bitrix24":
            raise PlanRejected("SOURCE_RESTRICTION_VIOLATION")
        available = set((constraints["capability_catalog"].get(specialist_id) or {}).get("capabilities", []))
        if not isinstance(capability, str) or capability not in available:
            raise PlanRejected("FORBIDDEN_CAPABILITY")
        if not isinstance(subrequest, str) or not subrequest.strip():
            raise PlanRejected("SUBTASK_REQUEST_INVALID")
        dispatch_key = (specialist_id, capability, _normalized_text(subrequest))
        if dispatch_key in seen_dispatches:
            raise PlanRejected("DUPLICATE_SUBTASK")
        if expected_segments:
            expected = expected_segments.get(segment_id or "")
            if expected is None or expected["specialist_id"] != specialist_id or expected["request"] != subrequest:
                raise PlanRejected("SEGMENT_BINDING_INVALID")
            seen_segments.add(segment_id)
        elif segment_id is not None:
            raise PlanRejected("SEGMENT_BINDING_INVALID")
        seen.add(subtask_id)
        seen_dispatches.add(dispatch_key)
        subtasks.append(Subtask(subtask_id, segment_id, specialist_id, capability, subrequest))
    if expected_segments and seen_segments != set(expected_segments):
        raise PlanRejected("SEGMENT_COMPLETENESS_FAILED")
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
        catalog: dict[str, dict[str, Any]] = {}
        for agent_id, manifest in call._manifests.items():
            specialist = call._specialists.get(agent_id)
            if specialist is None:
                continue
            tools: list[dict[str, str]] = []
            definitions = getattr(specialist, "tool_definitions", None)
            if callable(definitions):
                for definition in definitions():
                    if isinstance(definition, dict):
                        tool_id = str(definition.get("name") or "").strip()
                        if tool_id:
                            tools.append({"id": tool_id, "description": str(definition.get("description") or "")})
            tool_ids = {item["id"] for item in tools}
            catalog[agent_id] = {
                "description": str(manifest.handoff_description or manifest.name),
                "capabilities": sorted(set(manifest.capabilities) | tool_ids),
                "tools": tools,
                "allowed_actions": sorted(manifest.allowed_actions),
                "approval_required": sorted(manifest.approval_required),
            }
        return catalog

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

    async def _load_authoritative_dialog_history(self, task: AgentTask) -> AgentTask:
        """The planner must see its own earlier clarification, not only specialist history."""
        dialog_key = str(task.context.get("dialog_key") or "")
        if self.store is None or not dialog_key or not hasattr(self.store, "load_turns"):
            return task
        history = await self.store.load_turns(dialog_key, limit=12)  # type: ignore[attr-defined]
        return task.model_copy(update={"context": {**task.context, "dialog_history": list(history)}})

    async def _append_authoritative_dialog_turn(self, task: AgentTask, answer: str) -> None:
        dialog_key = str(task.context.get("dialog_key") or "")
        if self.store is not None and dialog_key and answer and hasattr(self.store, "append_turn"):
            await self.store.append_turn(dialog_key, task.request, answer)  # type: ignore[attr-defined]

    def _draft_control_result(self, task: AgentTask, answer: str, *, reason: str) -> AgentResult:
        usage = ModelUsageRecord(
            agent_id=self.manifest.id,
            provider="internal",
            model="active-draft-state-machine",
            status="not_used",
            notes=["No model call: protected active draft state was handled deterministically."],
        )
        return AgentResult(
            status="needs_clarification",
            agent_id=self.manifest.id,
            answer=answer,
            model_usage=[usage],
            actions_taken=[ActionRecord(name="active_draft_guard", status="ok", details={"reason": reason})],
            metadata={"reason": reason, "route": "active_draft_guard"},
        )

    async def _guard_active_draft(self, task: AgentTask) -> tuple[AgentTask, AgentResult | None]:
        """Keep a new explicit write intent out of an existing draft's tool path."""
        dialog_key = str(task.context.get("dialog_key") or "")
        call = self._tool_registry.get("call_specialist")
        if not dialog_key or not isinstance(call, CallSpecialistTool) or self.store is None:
            return task, None
        get_candidate = getattr(self.store, "get_replacement_candidate", None)
        save_candidate = getattr(self.store, "save_replacement_candidate", None)
        delete_candidate = getattr(self.store, "delete_replacement_candidate", None)
        if not callable(get_candidate) or not callable(save_candidate) or not callable(delete_candidate):
            return task, None
        candidate = await get_candidate(dialog_key)
        request = _normalized_text(task.request)
        if candidate:
            if request == _DRAFT_CONTINUE:
                await delete_candidate(dialog_key)
                return task, self._draft_control_result(
                    task,
                    "Продолжаем текущий черновик. Подтвердите его указанной фразой или внесите уточнение.",
                    reason="REPLACEMENT_CANDIDATE_DISMISSED",
                )
            if request == _DRAFT_REPLACE:
                discarded = await call.discard_active_bitrix_draft(
                    dialog_key, expected_draft_id=str(candidate.get("draft_id") or "")
                )
                if not discarded:
                    return task, self._draft_control_result(
                        task,
                        "Не удалось безопасно отменить прежний черновик: он уже изменён или истёк. Проверьте его состояние.",
                        reason="REPLACEMENT_DRAFT_CHANGED",
                    )
                await delete_candidate(dialog_key)
                replacement = task.model_copy(update={"request": str(candidate.get("request_text") or "")})
                return replacement, None
            return task, self._draft_control_result(
                task,
                "Есть незавершённый черновик и сохранён новый запрос. Напишите: «Продолжить текущий черновик» "
                "или «Отменить текущий черновик и запустить новый запрос».",
                reason="REPLACEMENT_CANDIDATE_WAITING",
            )
        draft = await call.get_active_bitrix_draft(dialog_key)
        if not draft or not _is_new_request_during_draft(task.request, draft):
            return task, None
        await save_candidate(
            dialog_key,
            request_text=task.request,
            draft_id=str(draft.get("_draft_id") or ""),
            draft_type=str(draft.get("_draft_type") or ""),
            ttl_minutes=15,
        )
        return task, self._draft_control_result(
            task,
            "Текущий черновик ещё не завершён; новый запрос сохранён на 15 минут и пока не запущен. "
            "Напишите: «Продолжить текущий черновик» или «Отменить текущий черновик и запустить новый запрос».",
            reason="REPLACEMENT_CANDIDATE_SAVED",
        )

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
                task = await self._load_authoritative_dialog_history(task)
                task, draft_guard_result = await self._guard_active_draft(task)
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
            if draft_guard_result is not None:
                result = draft_guard_result.model_copy(
                    update={"metadata": {**draft_guard_result.metadata, "total_ms": round((time.monotonic() - started) * 1000, 1)}}
                )
                await self._append_authoritative_dialog_turn(task, result.answer)
                await self._send_to_channel(task, result)
                await self._publish_result(task, result)
                return result
            catalog = self._catalog()
            pending_specialist = str(task.context.get("pending_specialist") or "").strip()
            constraints = _constraints(task.request, catalog, pending_specialist=pending_specialist)
            plan_id = f"plan-{uuid.uuid4().hex}"
            deterministic_route: str | None = None
            planner_usages: list[ModelUsageRecord] = []
            planner_rejections: list[str] = []
            planner_attempt_audit: list[dict[str, Any]] = []
            plan: Plan | None = None
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
            await self._append_authoritative_dialog_turn(task, result.answer)
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
                        # Keep the raw dialog request for audit.  A single
                        # subtask still receives it verbatim; every branch of a
                        # composite plan receives only its validated atom, so a
                        # Bitrix specialist cannot re-parse the whole request and
                        # accidentally run the last mentioned warehouse for all
                        # branches.
                        "t0006_original_request": task.request,
                        "t0006_planned_subtask_request": subtask.request,
                        "t0007_dispatch_request": subtask.request if len(plan.subtasks) > 1 else task.request,
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
        if deterministic_route or len(facts) == 1:
            answer = (
                "; ".join(str(item["answer"]).strip() for item in facts if str(item["answer"]).strip())
                or "Не удалось получить подтверждённый результат специалистов."
            )
            usages = list(planner_usages)
            if len(facts) == 1 and not deterministic_route:
                actions.append(
                    ActionRecord(
                        name="final_validation",
                        status="deterministic",
                        details={**base_meta, "reason": "SINGLE_VERIFIED_SPECIALIST_RESULT"},
                    )
                )
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
