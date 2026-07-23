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

from ai_server.agents.base import _trace_now_iso
from ai_server.capability_registry import registry_tool, validate_tool_arguments
from ai_server.llm import LLMClient
from ai_server.models import (
    ActionRecord,
    AgentManifest,
    AgentResult,
    AgentTask,
    ModelUsageRecord,
    ToolResult,
    ToolStatus,
)
from ai_server.orchestrators.bitrix_response import render_bitrix_tool_results
from ai_server.orchestrators.bitrix_semantics import (
    SemanticPolicyViolation,
    normalize_plan,
)
from ai_server.orchestrators.conversation_reference import resolve_conversation_reference
from ai_server.orchestrators.internal import OrchestratorTransportRuntime
from ai_server.orchestrators.logistics_response import render_logistics_tool_result
from ai_server.orchestrators.orchestrator_policy import selected_bitrix_policy
from ai_server.orchestrators.tools.call_specialist import CallSpecialistTool

PLAN_SCHEMA = "t0007.plan.v2"
FINAL_SCHEMA = "t0006.final.v1"
REPAIRABLE_PLAN_REJECTIONS = frozenset(
    {
        "INVALID_JSON",
        "PLAN_SCHEMA_MISMATCH",
        "PLAN_BINDING_MISMATCH",
        "NON_EXECUTION_PLAN_INVALID",
        "SEGMENT_BINDING_INVALID",
        "DUPLICATE_SUBTASK",
        "WAREHOUSE_SEGMENT_INCOMPLETE",
        "STRUCTURED_COMMAND_REQUIRED",
        "STRUCTURED_COMMAND_SCHEMA_MISMATCH",
        "STRUCTURED_COMMAND_TOOL_INVALID",
        "STRUCTURED_COMMAND_ARGUMENTS_INVALID",
        "SEMANTIC_ARGUMENTS_INVALID",
        "SEMANTIC_TOOL_MISMATCH",
        "ENTITY_AMBIGUOUS",
        "ENTITY_NOT_FOUND",
        "ENTITY_ID_MISMATCH",
        "ENTITY_CATALOG_UNAVAILABLE",
    }
)

_PLANNER_REGISTRY_BINDING = "CURRENT"
_PLANNER_ALWAYS_DETAILED_TOOLS = frozenset(
    {
        "bitrix_api",
        "task_create_confirm",
        "task_draft_discard",
        "task_close_confirm",
        "task_close_discard",
        "calendar_event_confirm",
        "calendar_event_discard",
        "project_create_confirm",
        "project_create_discard",
    }
)
def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s-]", " ", value.casefold())).strip()


def _planning_draft_view(draft: dict[str, Any]) -> dict[str, Any]:
    """Expose business fields to the planner without store/claim internals."""

    allowed_internal = {"_draft_type", "_draft_version"}

    def clean(value: object) -> object:
        if isinstance(value, dict):
            return {
                str(key): clean(item)
                for key, item in value.items()
                if not str(key).startswith("_") or str(key) in allowed_internal
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return dict(clean(draft))


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

    def __init__(self, client: LLMClient) -> None:
        if client is None:
            raise RuntimeError("PLAN_AUTHORITATIVE_LLM_CLIENT_REQUIRED")
        self.client = client

    async def plan(
        self, *, manifest: AgentManifest, task: AgentTask, catalog: dict[str, Any], constraints: dict[str, Any]
    ) -> tuple[str, ModelUsageRecord]:
        repair_reason = str(constraints.get("repair_reason") or "")
        payload: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA,
            "plan_id": constraints["plan_id"],
            "request": task.request,
            "request_hash": constraints["request_hash"],
            "user": task.user.model_dump(),
            "dialog_history": list(task.context.get("dialog_history") or []),
            "active_bitrix_draft": task.context.get("active_bitrix_draft"),
            "entity_catalog": task.context.get("orchestrator_entity_catalog"),
            "execution_history": list(task.context.get("orchestrator_execution_history") or []),
            "capability_catalog": catalog,
            "hard_constraints": {
                key: value
                for key, value in constraints.items()
                if key
                not in {
                    "plan_id",
                    "request_hash",
                    "repair_reason",
                    "repair_attempt",
                    "capability_catalog",
                }
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
                        "structured_command": (
                            "null for a legacy capability, otherwise an object with exactly: "
                            "registry_version, tool_name, arguments; registry_version must be the literal CURRENT"
                        ),
                    }
                ],
            },
            "schema_contract": (
                "Return exactly one JSON object with exactly these top-level keys: "
                "schema_version, plan_id, request_hash, state, clarification, max_rounds, subtasks. "
                "Do not add wrapper, explanation, markdown, or any other key."
            ),
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
                        "Return only strict JSON. You are the sole semantic planner; do not call tools or compose a user answer. "
                        "Before sending, verify that the top-level keys are exactly the keys in required_response and that "
                        "schema_version, plan_id, and request_hash exactly match their required_response values. "
                        "For every tool marked structured_command=true, choose the exact tool and arguments yourself and "
                        "return a structured_command with registry_version set to the literal CURRENT. The backend binds "
                        "CURRENT to the authoritative live catalog version before dispatch. The specialist will execute it once and is forbidden "
                        "to reinterpret the request. Use selected_orchestrator_rules as the business authority; "
                        "specialist skills are capability labels only and must not change meaning. "
                        "Resolve named employees, projects and warehouses only from entity_catalog and place exact IDs "
                        "into the structured command. If no unique entry exists, ask one clarification. "
                        "The tools list is the complete capability index; tool_contracts contains detailed schemas for this request. "
                        "selected_skill_rules and selected_contract_rules contain the applicable detailed rules. "
                        "For such a subtask, capability and structured_command.tool_name must both equal that exact tool id. "
                        "max_rounds counts this execution round. Use a value above 1 only when the current exact command is "
                        "a lookup whose returned IDs/data are required for a later command. On the next planner call, use "
                        "execution_history and never repeat an already successful command with identical arguments. "
                        "If the user request is ambiguous, return CLARIFICATION_REQUIRED instead of guessing arguments. "
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
    structured_command: StructuredCommand | None = None


@dataclass(frozen=True)
class StructuredCommand:
    registry_version: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Plan:
    plan_id: str
    state: str
    clarification: str | None
    subtasks: list[Subtask]
    max_rounds: int = 1


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _planner_capability_catalog(catalog: dict[str, Any], request: str) -> dict[str, Any]:
    """Return a small, non-authoritative view for the planning model.

    The complete live registry remains in ``constraints`` for backend validation.
    Unknown request types deliberately receive every tool contract so catalog
    compaction cannot silently remove a future capability.
    """

    policy = selected_bitrix_policy(request)
    selected_rules = [
        rule for rule in policy["rules"] if isinstance(rule, dict) and rule.get("id") != "authority"
    ]
    detailed_tool_ids = set(_PLANNER_ALWAYS_DETAILED_TOOLS)
    for rule in selected_rules:
        detailed_tool_ids.update(str(item) for item in rule.get("tools") or [])

    compact: dict[str, Any] = {}
    for specialist_id, entry in catalog.items():
        tools = [item for item in entry.get("tools") or [] if isinstance(item, dict)]
        if specialist_id != "bitrix24" or not selected_rules:
            selected_tools = tools
        else:
            selected_tools = [item for item in tools if str(item.get("id") or "") in detailed_tool_ids]
        skills = [item for item in entry.get("skills") or [] if isinstance(item, dict)]
        contracts = [item for item in entry.get("contracts") or [] if isinstance(item, dict)]
        compact[specialist_id] = {
            "description": entry.get("description"),
            "reasoning_mode": entry.get("reasoning_mode"),
            "capabilities": list(entry.get("capabilities") or []),
            "registry_binding": _PLANNER_REGISTRY_BINDING,
            "tools": [
                {
                    "id": item.get("id"),
                    "description": item.get("description"),
                    "structured_command": bool(item.get("structured_command")),
                }
                for item in tools
            ],
            "tool_contracts": selected_tools,
            "skills": [{"id": item.get("id"), "title": item.get("title")} for item in skills],
            "selected_skill_rules": [],
            "contracts": [{"id": item.get("id")} for item in contracts],
            "selected_contract_rules": [],
            "allowed_actions": list(entry.get("allowed_actions") or []),
            "approval_required": list(entry.get("approval_required") or []),
        }
        if specialist_id == "bitrix24":
            compact[specialist_id]["orchestrator_policy_version"] = policy["schema_version"]
            compact[specialist_id]["orchestrator_defaults"] = policy["defaults"]
            compact[specialist_id]["orchestrator_templates"] = policy["templates"]
            compact[specialist_id]["selected_orchestrator_rules"] = policy["rules"]
    return compact


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
            voice_segments.append({"segment_id": f"segment-{index}", "specialist_id": specialist_id, "request": body})
    return voice_segments or segments


def _required_warehouse_labels(request: str, catalog: dict[str, Any]) -> list[str]:
    """Return unambiguous simple warehouse labels explicitly enumerated by a user.

    This is deliberately a narrow fail-closed guard, not a replacement for the
    Pro planner.  It activates only when a warehouse request contains an
    explicit list connector (comma or ``и``).  Each resulting label must be
    represented by its own validated Bitrix warehouse subtask.
    """
    capabilities = set((catalog.get("bitrix24") or {}).get("capabilities", []))
    if "bitrix_warehouse_search" not in capabilities:
        return []
    text = _normalized_text(request)
    if not re.search(r"\b(?:покажи|найди|выведи)\b", text):
        return []
    match = re.search(r"\bсклад\w*\b\s+(.+)", text)
    if not match:
        return []
    tail = match.group(1)
    if "," not in request and " и " not in f" {tail} ":
        return []
    excluded = {
        "и",
        "или",
        "склад",
        "склады",
        "все",
        "всё",
        "позиции",
        "остатки",
        "остаток",
        "наличие",
        "товары",
        "товаров",
        "по",
        "на",
        "в",
        "что",
        "есть",
        "покажи",
        "найди",
        "выведи",
        "мне",
    }
    labels: list[str] = []
    for token in tail.split():
        if token in excluded or token.isdigit() or token in labels:
            continue
        labels.append(token)
    return labels if len(labels) > 1 else []


def _constraints(
    request: str,
    catalog: dict[str, Any],
    *,
    pending_specialist: str | None = None,
    conversation_reference_error: str | None = None,
    max_round_trips: int = 3,
    prior_command_fingerprints: list[str] | None = None,
) -> dict[str, Any]:
    text = request.casefold()
    only_bitrix = "только bitrix" in text or "только битрикс" in text
    return {
        "only_source": "bitrix24" if only_bitrix else None,
        "allowed_specialists": sorted(catalog),
        "capability_catalog": catalog,
        "explicit_segments": _explicit_segments(request, catalog),
        "required_warehouse_labels": _required_warehouse_labels(request, catalog),
        "pending_specialist": pending_specialist or None,
        "conversation_reference_error": conversation_reference_error or None,
        "max_subtasks": 8,
        "max_round_trips": max(1, min(int(max_round_trips), 3)),
        "prior_command_fingerprints": list(prior_command_fingerprints or []),
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
    if constraints.get("conversation_reference_error") and state != "CLARIFICATION_REQUIRED":
        raise PlanRejected("CONVERSATION_REFERENCE_VIOLATION")
    expected_segments = {item["segment_id"]: item for item in constraints.get("explicit_segments", [])}
    seen: set[str] = set()
    seen_segments: set[str] = set()
    seen_dispatches: set[tuple[str, str, str]] = set()
    subtasks: list[Subtask] = []
    for item in items:
        legacy_keys = {
            "subtask_id",
            "segment_id",
            "specialist_id",
            "capability",
            "request",
        }
        if not isinstance(item, dict) or set(item) not in {
            frozenset(legacy_keys),
            frozenset({*legacy_keys, "structured_command"}),
        }:
            raise PlanRejected("SUBTASK_SCHEMA_MISMATCH")
        subtask_id = item["subtask_id"]
        segment_id = item["segment_id"]
        specialist_id = item["specialist_id"]
        capability = item["capability"]
        subrequest = item["request"]
        raw_command = item.get("structured_command")
        if not isinstance(subtask_id, str) or not subtask_id or subtask_id in seen:
            raise PlanRejected("SUBTASK_ID_INVALID")
        if segment_id is not None and (not isinstance(segment_id, str) or segment_id in seen_segments):
            raise PlanRejected("SEGMENT_BINDING_INVALID")
        if not isinstance(specialist_id, str) or specialist_id not in constraints["allowed_specialists"]:
            raise PlanRejected("FORBIDDEN_SPECIALIST")
        if constraints["only_source"] and specialist_id != "bitrix24":
            raise PlanRejected("SOURCE_RESTRICTION_VIOLATION")
        specialist_catalog = constraints["capability_catalog"].get(specialist_id) or {}
        available = set(specialist_catalog.get("capabilities", []))
        if not isinstance(capability, str) or capability not in available:
            raise PlanRejected("FORBIDDEN_CAPABILITY")
        if not isinstance(subrequest, str) or not subrequest.strip():
            raise PlanRejected("SUBTASK_REQUEST_INVALID")
        tool_contract = registry_tool(specialist_catalog, capability)
        structured_required = bool(tool_contract and tool_contract.get("structured_command"))
        structured_command: StructuredCommand | None = None
        if raw_command is None:
            if structured_required:
                raise PlanRejected("STRUCTURED_COMMAND_REQUIRED")
        else:
            if not isinstance(raw_command, dict) or set(raw_command) != {
                "registry_version",
                "tool_name",
                "arguments",
            }:
                raise PlanRejected("STRUCTURED_COMMAND_SCHEMA_MISMATCH")
            registry_binding = str(raw_command.get("registry_version") or "")
            registry_version = str(specialist_catalog.get("registry_version") or "")
            tool_name = str(raw_command.get("tool_name") or "")
            arguments = raw_command.get("arguments")
            if registry_binding not in {_PLANNER_REGISTRY_BINDING, registry_version}:
                raise PlanRejected("CAPABILITY_REGISTRY_VERSION_MISMATCH")
            if tool_name != capability or tool_contract is None or not tool_contract.get("structured_command"):
                raise PlanRejected("STRUCTURED_COMMAND_TOOL_INVALID")
            if not isinstance(arguments, dict):
                raise PlanRejected("STRUCTURED_COMMAND_ARGUMENTS_INVALID")
            preliminary_errors = validate_tool_arguments(
                dict(tool_contract.get("parameters") or {}), arguments
            )
            # Required/defaulted fields are completed by the authoritative
            # semantic layer with task/user/entity context immediately after
            # decoding. Unknown, malformed and out-of-range values are never
            # allowed through that normalization boundary.
            blocking_errors = [
                error
                for error in preliminary_errors
                if not error.endswith(": required") and "no anyOf contract matched" not in error
            ]
            if blocking_errors:
                raise PlanRejected("STRUCTURED_COMMAND_ARGUMENTS_INVALID")
            structured_command = StructuredCommand(registry_version, tool_name, dict(arguments))
        command_fingerprint = (
            json.dumps(structured_command.arguments, ensure_ascii=False, sort_keys=True)
            if structured_command is not None
            else _normalized_text(subrequest)
        )
        dispatch_key = (specialist_id, capability, command_fingerprint)
        prior_fingerprint = f"{specialist_id}:{capability}:{command_fingerprint}"
        if prior_fingerprint in set(constraints.get("prior_command_fingerprints") or []):
            raise PlanRejected("DUPLICATE_SUBTASK")
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
        subtasks.append(Subtask(subtask_id, segment_id, specialist_id, capability, subrequest, structured_command))
    if expected_segments and seen_segments != set(expected_segments):
        raise PlanRejected("SEGMENT_COMPLETENESS_FAILED")
    required_warehouse_labels = list(constraints.get("required_warehouse_labels") or [])
    if required_warehouse_labels:
        label_counts = {label: 0 for label in required_warehouse_labels}
        for subtask in subtasks:
            if subtask.specialist_id != "bitrix24" or subtask.capability != "bitrix_warehouse_search":
                continue
            subtask_words = set(_normalized_text(subtask.request).split())
            represented = [label for label in required_warehouse_labels if label in subtask_words]
            if len(represented) > 1:
                raise PlanRejected("WAREHOUSE_SEGMENT_INCOMPLETE")
            if represented:
                label_counts[represented[0]] += 1
        if any(count != 1 for count in label_counts.values()):
            raise PlanRejected("WAREHOUSE_SEGMENT_INCOMPLETE")
    return Plan(plan_id, state, clarification, subtasks, value["max_rounds"])


class PlanAuthoritativeOrchestrator(OrchestratorTransportRuntime):
    """The sole live semantic orchestrator runtime."""

    def __init__(
        self,
        *args: Any,
        planner: PlanAuthoritativeLLM,
        entity_catalog: Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._planner = planner
        self._entity_catalog = entity_catalog

    @classmethod
    def build(cls, manifest: AgentManifest | None, **kwargs: Any) -> PlanAuthoritativeOrchestrator:
        from ai_server.orchestrators.tools import ManageSuspendedTool, ScheduleTaskTool
        from ai_server.specialists import build_specialist_registry

        manifests = kwargs.pop("manifests", None) or []
        planner = kwargs.pop("orchestrator_llm")
        if not callable(getattr(planner, "plan", None)) or not callable(
            getattr(planner, "finalize", None)
        ):
            raise RuntimeError("PLAN_AUTHORITATIVE_PLANNER_REQUIRED")
        store = kwargs.pop("orchestrator_store", None)
        retriever = kwargs.pop("orchestrator_retriever", None)
        channels = kwargs.pop("channels", None)
        footer_service = kwargs.pop("footer_service", None)
        result_publisher = kwargs.pop("result_publisher", None)
        entity_catalog = kwargs.pop("orchestrator_entity_catalog", None)
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
            entity_catalog=entity_catalog,
            outbound_queue=kwargs.get("outbound_queue"),
        )
        call.schedule_fn = instance._apply_scheduled_tasks_from_specialist
        instance._validate_live_catalog(instance._catalog())
        return instance

    @staticmethod
    def _validate_live_catalog(catalog: dict[str, dict[str, Any]]) -> None:
        """Fail startup when an advertised structured command is incomplete."""

        for specialist_id, entry in catalog.items():
            reasoning_mode = str(entry.get("reasoning_mode") or "")
            if reasoning_mode not in {"autonomous", "executor"}:
                raise RuntimeError(f"SPECIALIST_REASONING_MODE_INVALID:{specialist_id}")
            registry_version = str(entry.get("registry_version") or "")
            seen: set[str] = set()
            for tool in entry.get("tools") or []:
                if not isinstance(tool, dict):
                    raise RuntimeError(f"CAPABILITY_REGISTRY_INVALID:{specialist_id}")
                tool_id = str(tool.get("id") or "")
                if not tool_id or tool_id in seen:
                    raise RuntimeError(f"CAPABILITY_REGISTRY_DUPLICATE_TOOL:{specialist_id}")
                seen.add(tool_id)
                if tool.get("structured_command") and (
                    not registry_version
                    or not str(tool.get("version") or "")
                    or not isinstance(tool.get("parameters"), dict)
                ):
                    raise RuntimeError(f"CAPABILITY_REGISTRY_STRUCTURED_TOOL_INVALID:{specialist_id}:{tool_id}")
                if reasoning_mode == "executor" and not tool.get("structured_command"):
                    raise RuntimeError(f"EXECUTOR_UNSTRUCTURED_TOOL_FORBIDDEN:{specialist_id}:{tool_id}")

    def _catalog(self) -> dict[str, dict[str, Any]]:
        call = self._tool_registry.get("call_specialist")
        if not isinstance(call, CallSpecialistTool):
            return {}
        catalog: dict[str, dict[str, Any]] = {}
        for agent_id, manifest in call._manifests.items():
            registry = call.capability_registry(agent_id)
            if registry is None:
                continue
            tools = [item for item in registry.get("tools") or [] if isinstance(item, dict)]
            tool_ids = {item["id"] for item in tools}
            capabilities = (
                tool_ids or set(manifest.capabilities)
                if agent_id == "bitrix24"
                else set(manifest.capabilities) | tool_ids
            )
            catalog[agent_id] = {
                "description": str(manifest.handoff_description or manifest.name),
                "reasoning_mode": manifest.reasoning_mode,
                "capabilities": sorted(capabilities),
                "tools": tools,
                "skills": [item for item in registry.get("skills") or [] if isinstance(item, dict)],
                "contracts": [item for item in registry.get("contracts") or [] if isinstance(item, dict)],
                "registry_schema": registry.get("schema_version"),
                "registry_version": registry.get("registry_version"),
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

    async def _guard_active_draft(self, task: AgentTask) -> tuple[AgentTask, AgentResult | None]:
        """Attach branch-local draft state; the mandatory Pro planner decides meaning."""
        dialog_key = str(task.context.get("dialog_key") or "")
        call = self._tool_registry.get("call_specialist")
        if not dialog_key or not isinstance(call, CallSpecialistTool):
            return task, None
        draft = await call.get_active_bitrix_draft(dialog_key)
        if not draft:
            return task, None
        task = task.model_copy(update={"context": {**task.context, "active_bitrix_draft": _planning_draft_view(draft)}})
        return task, None

    async def handle(self, task: AgentTask) -> AgentResult:
        started = time.monotonic()
        started_at = _trace_now_iso()
        await self._record_timing(
            task,
            stage="orchestrator_entry",
            started_at=started_at,
            elapsed_ms=0.0,
            status="claimed",
            details={"task_id": task.task_id},
        )
        # Reference failures deliberately still enter the mandatory Pro plan.
        # The validated plan receives a fail-closed constraint and cannot
        # dispatch a specialist for an unknown, expired or ambiguous branch.
        reference_started_at = _trace_now_iso()
        reference_t0 = time.monotonic()
        task = (await resolve_conversation_reference(task, self.store)).task
        await self._record_timing(
            task,
            stage="conversation_reference",
            started_at=reference_started_at,
            elapsed_ms=(time.monotonic() - reference_t0) * 1000,
            status="restricted" if task.context.get("conversation_reference_error") else "completed",
            details={
                "conversation_number": task.context.get("conversation_number"),
                "dispatch_allowed": bool(task.context.get("conversation_reference_dispatch_allowed", True)),
            },
        )
        dialog_key = str(task.context.get("dialog_key") or "")
        active = False
        if self._dialog_guard is not None and dialog_key:
            generation = await self._dialog_guard.mark_active(task, ttl_seconds=3600)
            task = task.model_copy(update={"context": {**task.context, "dialog_cancel_generation": int(generation)}})
            active = True
        try:
            state_started_at = _trace_now_iso()
            state_t0 = time.monotonic()
            try:
                task, _ = await self._load_authoritative_pending_specialist(task)
                task = await self._load_authoritative_dialog_history(task)
            except Exception:
                await self._record_timing(
                    task,
                    stage="orchestrator_state_load",
                    started_at=state_started_at,
                    elapsed_ms=(time.monotonic() - state_t0) * 1000,
                    status="error",
                )
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
            await self._record_timing(
                task,
                stage="orchestrator_state_load",
                started_at=state_started_at,
                elapsed_ms=(time.monotonic() - state_t0) * 1000,
                status="completed",
                details={"pending_specialist": str(task.context.get("pending_specialist") or "")},
            )
            task, _ = await self._guard_active_draft(task)
            if self._entity_catalog is not None:
                entity_view = self._entity_catalog.view_for_request(task.request)
                task = task.model_copy(
                    update={"context": {**task.context, "orchestrator_entity_catalog": entity_view}}
                )
            catalog = self._catalog()
            planner_catalog = _planner_capability_catalog(catalog, task.request)
            pending_specialist = str(task.context.get("pending_specialist") or "").strip()
            constraints = _constraints(
                task.request,
                catalog,
                pending_specialist=pending_specialist,
                conversation_reference_error=str(task.context.get("conversation_reference_error") or "") or None,
            )
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
                    planner_started_at = _trace_now_iso()
                    planner_t0 = time.monotonic()
                    raw, usage = await self._planner.plan(
                        manifest=self.manifest,
                        task=task,
                        catalog=planner_catalog,
                        constraints=call_constraints,
                    )
                except Exception:
                    await self._record_timing(
                        task,
                        stage="pro_plan",
                        started_at=planner_started_at,
                        elapsed_ms=(time.monotonic() - planner_t0) * 1000,
                        status="error",
                        step=attempt,
                        details={
                            "attempt": attempt,
                            "repair_reason": planner_rejections[-1] if planner_rejections else "",
                        },
                    )
                    if attempt == 1:
                        raise
                    reason = "MODEL_REPAIR_UNAVAILABLE"
                    planner_rejections.append(reason)
                    planner_attempt_audit.append({"attempt": attempt, "status": "error", "rejection": reason})
                    break
                await self._record_timing(
                    task,
                    stage="pro_plan",
                    started_at=planner_started_at,
                    elapsed_ms=(time.monotonic() - planner_t0) * 1000,
                    status="completed",
                    step=attempt,
                    details={
                        "attempt": attempt,
                        "repair_reason": planner_rejections[-1] if planner_rejections else "",
                        "model": usage.model,
                    },
                )
                planner_usages.append(usage)
                response_hash = _hash(raw)
                attempt_audit = {
                    "attempt": attempt,
                    "response_hash": response_hash,
                    "status": "accepted",
                }
                planner_attempt_audit.append(attempt_audit)
                validation_started_at = _trace_now_iso()
                validation_t0 = time.monotonic()
                try:
                    plan = _decode_plan(raw, plan_id=plan_id, request=task.request, constraints=constraints)
                    plan = normalize_plan(
                        plan,
                        task=task,
                        constraints=constraints,
                        entity_catalog=self._entity_catalog.snapshot() if self._entity_catalog is not None else {},
                    )
                except (PlanRejected, SemanticPolicyViolation) as exc:
                    reason = str(exc)
                    await self._record_timing(
                        task,
                        stage="plan_validation",
                        started_at=validation_started_at,
                        elapsed_ms=(time.monotonic() - validation_t0) * 1000,
                        status="rejected",
                        step=attempt,
                        details={"reason": reason, "subtasks": 0},
                    )
                    attempt_audit.update({"status": "rejected", "rejection": reason})
                    planner_rejections.append(reason)
                    if attempt == 1 and reason in REPAIRABLE_PLAN_REJECTIONS:
                        continue
                else:
                    await self._record_timing(
                        task,
                        stage="plan_validation",
                        started_at=validation_started_at,
                        elapsed_ms=(time.monotonic() - validation_t0) * 1000,
                        status="accepted",
                        step=attempt,
                        details={"plan_state": plan.state, "subtasks": len(plan.subtasks)},
                    )
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
                answer=str(
                    task.context.get("conversation_reference_error") or plan.clarification or "Уточните запрос."
                ),
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

        dispatch_group = f"{plan.plan_id}:specialists"

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
                        "t0006_planned_capability": subtask.capability,
                        "t0007_dispatch_request": subtask.request if len(plan.subtasks) > 1 else task.request,
                    }
                }
            )
            dispatch_started_at = _trace_now_iso()
            dispatch_t0 = time.monotonic()
            try:
                structured_command = (
                    {
                        "registry_version": subtask.structured_command.registry_version,
                        "tool_name": subtask.structured_command.tool_name,
                        "arguments": subtask.structured_command.arguments,
                    }
                    if subtask.structured_command is not None
                    else None
                )
                value = await call.execute_with_task(
                    {
                        "specialist_id": subtask.specialist_id,
                        "request": subtask.request,
                        "structured_command": structured_command,
                    },
                    task=correlated_task,
                )
            except Exception:
                await self._record_timing(
                    correlated_task,
                    stage="specialist_dispatch",
                    started_at=dispatch_started_at,
                    elapsed_ms=(time.monotonic() - dispatch_t0) * 1000,
                    status="error",
                    details={
                        "parallel_group": dispatch_group,
                        "subtask_id": subtask.subtask_id,
                        "specialist_id": subtask.specialist_id,
                        "structured_command": subtask.structured_command is not None,
                    },
                )
                raise
            await self._record_timing(
                correlated_task,
                stage="specialist_dispatch",
                started_at=dispatch_started_at,
                elapsed_ms=(time.monotonic() - dispatch_t0) * 1000,
                status=str(value.status),
                details={
                    "parallel_group": dispatch_group,
                    "subtask_id": subtask.subtask_id,
                    "specialist_id": subtask.specialist_id,
                },
            )
            return subtask, attempt_id, value

        completed = await asyncio.gather(*(run(item) for item in plan.subtasks), return_exceptions=True)
        facts: list[dict[str, Any]] = []
        actions: list[ActionRecord] = [ActionRecord(name="plan_validation", status="ok", details=base_meta)]
        if deterministic_route:
            actions.append(ActionRecord(name="pending_specialist_route", status="ok", details=base_meta))
        approvals: list[ActionRecord] = []
        execution_records: list[dict[str, Any]] = []
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
            specialist_status = branch_status
            branch_answer = str(data.get("answer") or tool.error or "").strip()
            specialist_metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            if specialist_metadata.get("structured_command") and isinstance(
                specialist_metadata.get("tool_result"), dict
            ):
                raw_tool_result = ToolResult.model_validate(specialist_metadata["tool_result"])
                command_arguments = (
                    specialist_metadata.get("command_arguments")
                    if isinstance(specialist_metadata.get("command_arguments"), dict)
                    else {}
                )
                if specialist_metadata.get("formatter_domain") == "logistics":
                    rendered = render_logistics_tool_result(
                        tool_result=raw_tool_result,
                        command_arguments=command_arguments,
                    )
                else:
                    rendered = render_bitrix_tool_results(
                        agent_id=self.manifest.id,
                        tool_results=[raw_tool_result],
                        portal_base_url=str(specialist_metadata.get("portal_base_url") or ""),
                        command_arguments=command_arguments,
                    )
                branch_status = rendered.status
                branch_answer = rendered.answer
                if specialist_status in {"needs_human", "needs_clarification", "failed"}:
                    branch_status = specialist_status
                execution_records.append(
                    {
                        "subtask_id": item.subtask_id,
                        "specialist_id": item.specialist_id,
                        "capability": item.capability,
                        "registry_version": specialist_metadata.get("registry_version"),
                        "tool_name": raw_tool_result.tool,
                        "arguments": command_arguments,
                        "result": raw_tool_result.model_dump(),
                        "command_fingerprint": (
                            f"{item.specialist_id}:{item.capability}:"
                            f"{json.dumps(command_arguments, ensure_ascii=False, sort_keys=True)}"
                        ),
                    }
                )
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
                    "terminal": bool(data.get("terminal")),
                    "answer_is_final": bool(data.get("answer_is_final")),
                    "safe_to_send": bool(data.get("safe_to_send")),
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
        if (
            plan.max_rounds > 1
            and execution_records
            and not approvals
            and all(str(fact.get("status") or "") == "completed" for fact in facts)
            and all(str(record["result"].get("status") or "") == str(ToolStatus.OK) for record in execution_records)
        ):
            followup = await self._plan_followup(
                task,
                execution_records=execution_records,
                remaining_rounds=plan.max_rounds - 1,
            )
            if followup is not None:
                next_plan, next_hash, next_usages, next_rejections, next_audit = followup
                next_result = await self._execute(
                    task.model_copy(
                        update={
                            "context": {
                                **task.context,
                                "orchestrator_execution_history": [
                                    *list(task.context.get("orchestrator_execution_history") or []),
                                    *execution_records,
                                ],
                            }
                        }
                    ),
                    next_plan,
                    next_hash,
                    next_usages,
                    planner_rejections=next_rejections,
                    planner_attempt_audit=next_audit,
                )
                return next_result.model_copy(
                    update={
                        "actions_taken": [*actions, *next_result.actions_taken],
                        "model_usage": [*planner_usages, *next_result.model_usage],
                        "handoff_to": sorted(
                            {*next_result.handoff_to, *(record["specialist_id"] for record in execution_records)}
                        ),
                        "metadata": {
                            **next_result.metadata,
                            "structured_command_rounds": 1
                            + int(next_result.metadata.get("structured_command_rounds") or 0),
                        },
                    }
                )
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer="Не удалось безопасно продолжить многошаговую команду Bitrix.",
                model_usage=list(planner_usages),
                actions_taken=[
                    *actions,
                    ActionRecord(
                        name="structured_command_followup",
                        status="rejected",
                        details={"reason": "FOLLOWUP_PLAN_UNAVAILABLE"},
                    ),
                ],
                handoff_to=sorted({record["specialist_id"] for record in execution_records}),
                metadata={"reason": "FOLLOWUP_PLAN_UNAVAILABLE", "structured_command_rounds": 1},
            )
        render_started_at = _trace_now_iso()
        render_t0 = time.monotonic()
        answer = (
            "; ".join(str(item["answer"]).strip() for item in facts if str(item["answer"]).strip())
            or "Не удалось получить подтверждённый результат специалистов."
        )
        usages = list(planner_usages)
        verified_terminal = (
            all(
                item["status"] == "completed"
                and bool(item["answer"].strip())
                and item["terminal"]
                and item["answer_is_final"]
                and item["safe_to_send"]
                for item in facts
            )
            and not approvals
        )
        actions.append(
            ActionRecord(
                name="final_validation",
                status="deterministic",
                details={
                    **base_meta,
                    "reason": "VALIDATED_PLAN_ORDER",
                    "verified_terminal_facts": verified_terminal,
                },
            )
        )
        await self._record_timing(
            task,
            stage="deterministic_render",
            started_at=render_started_at,
            elapsed_ms=(time.monotonic() - render_t0) * 1000,
            status="completed",
            details={
                "branch_count": len(facts),
                "branch_statuses": [str(item["status"]) for item in facts],
                "parallel_group": dispatch_group,
            },
        )
        branch_statuses = {str(item["status"]) for item in facts}
        if approvals or "needs_human" in branch_statuses:
            status = "needs_human"
        elif "needs_clarification" in branch_statuses:
            # A composite request can legitimately contain a completed answer
            # and an informational "not found" branch.  Do not expose that as
            # a blocking clarification unless the branch actually asks the
            # user a follow-up question.
            clarification_answers = [
                str(item["answer"]) for item in facts if str(item["status"]) == "needs_clarification"
            ]
            needs_follow_up = any("?" in answer or "уточните" in answer.casefold() for answer in clarification_answers)
            status = "needs_clarification" if needs_follow_up or len(facts) == 1 else "completed"
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
            metadata={
                **base_meta,
                "branches": facts,
                "structured_command_rounds": 1 if execution_records else 0,
            },
        )

    async def _plan_followup(
        self,
        task: AgentTask,
        *,
        execution_records: list[dict[str, Any]],
        remaining_rounds: int,
    ) -> tuple[Plan, str, list[ModelUsageRecord], list[str], list[dict[str, Any]]] | None:
        history = [*list(task.context.get("orchestrator_execution_history") or []), *execution_records]
        planning_task = task.model_copy(update={"context": {**task.context, "orchestrator_execution_history": history}})
        catalog = self._catalog()
        self._validate_live_catalog(catalog)
        planner_catalog = _planner_capability_catalog(catalog, task.request)
        constraints = _constraints(
            task.request,
            catalog,
            pending_specialist=str(task.context.get("pending_specialist") or "") or None,
            max_round_trips=remaining_rounds,
            prior_command_fingerprints=[
                str(item.get("command_fingerprint") or "") for item in history if isinstance(item, dict)
            ],
        )
        plan_id = f"plan-{uuid.uuid4().hex}"
        usages: list[ModelUsageRecord] = []
        rejections: list[str] = []
        audit: list[dict[str, Any]] = []
        for attempt in range(1, 3):
            call_constraints = {
                **constraints,
                "plan_id": plan_id,
                "request_hash": _hash(task.request),
                "structured_followup_round": True,
            }
            if rejections:
                call_constraints.update({"repair_reason": rejections[-1], "repair_attempt": attempt})
            started_at = _trace_now_iso()
            started = time.monotonic()
            try:
                raw, usage = await self._planner.plan(
                    manifest=self.manifest,
                    task=planning_task,
                    catalog=planner_catalog,
                    constraints=call_constraints,
                )
            except Exception:
                await self._record_timing(
                    planning_task,
                    stage="pro_plan_followup",
                    started_at=started_at,
                    elapsed_ms=(time.monotonic() - started) * 1000,
                    status="error",
                    step=attempt,
                )
                return None
            usages.append(usage)
            response_hash = _hash(raw)
            record = {"attempt": attempt, "response_hash": response_hash, "status": "accepted"}
            audit.append(record)
            try:
                plan = _decode_plan(raw, plan_id=plan_id, request=task.request, constraints=constraints)
                plan = normalize_plan(
                    plan,
                    task=planning_task,
                    constraints=constraints,
                    entity_catalog=self._entity_catalog.snapshot() if self._entity_catalog is not None else {},
                )
            except (PlanRejected, SemanticPolicyViolation) as exc:
                reason = str(exc)
                rejections.append(reason)
                record.update({"status": "rejected", "rejection": reason})
                await self._record_timing(
                    planning_task,
                    stage="pro_plan_followup",
                    started_at=started_at,
                    elapsed_ms=(time.monotonic() - started) * 1000,
                    status="rejected",
                    step=attempt,
                    details={"reason": reason},
                )
                if attempt == 1 and reason in REPAIRABLE_PLAN_REJECTIONS:
                    continue
                return None
            await self._record_timing(
                planning_task,
                stage="pro_plan_followup",
                started_at=started_at,
                elapsed_ms=(time.monotonic() - started) * 1000,
                status="accepted",
                step=attempt,
                details={"subtasks": len(plan.subtasks)},
            )
            return plan, response_hash, usages, rejections, audit
        return None

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
