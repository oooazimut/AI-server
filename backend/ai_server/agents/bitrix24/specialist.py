from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.bitrix24.llm import (
    BitrixAgentLLM,
    BitrixLLMService,
    BitrixLLMToolCall,
    llm_failure_result,
)
from ai_server.agents.bitrix24.quality_control import (
    TASK_QUALITY_WEBHOOK_EVENTS,
    handle_quality_control_task,
)
from ai_server.agents.bitrix24.task_create import (
    BitrixTaskCreateDraft,
    BitrixTaskCreateResolution,
    build_task_create_draft_from_args,
)
from ai_server.agents.bitrix24.tools import (
    BitrixApiTool,
    CurrentUserProfileTool,
    NotifyUsersTool,
    PortalSearchTool,
    ResolveProjectTool,
    ResolveUserTool,
)
from ai_server.agents.ports import PortalSearchPort, SchedulerPort
from ai_server.agents.tool import AgentTool
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.integrations.bitrix.ports import BitrixTaskPort, BitrixToolClientPort
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ToolResult, ToolStatus
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import Settings, get_settings
from ai_server.skills import SkillStore
from ai_server.tracing import TraceRecorder
from ai_server.utils import MOSCOW_TZ, optional_int

logger = logging.getLogger(__name__)

_TASK_CREATE_DRAFT_DEF: dict[str, Any] = {
    "name": "task_create_draft",
    "description": (
        "Prepare a Bitrix task creation draft from fields already understood by the LLM. "
        "The LLM must call this only after it has checked that the contract is complete. "
        "The backend only rejects malformed calls, resolves explicit Bitrix lookup args, applies policy, "
        "and requires confirmation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "responsible_id": {"type": "integer"},
            "responsible_query": {"type": "string"},
            "responsible_self": {"type": "boolean"},
            "group_id": {"type": "integer"},
            "project_query": {"type": "string"},
            "deadline_iso": {"type": "string"},
            "no_deadline": {"type": "boolean"},
        },
        "required": ["title"],
        "allOf": [
            {
                "anyOf": [
                    {"required": ["responsible_id"]},
                    {"required": ["responsible_query"]},
                    {"required": ["responsible_self"]},
                ]
            },
            {
                "anyOf": [
                    {"required": ["deadline_iso"]},
                    {"required": ["no_deadline"]},
                ]
            },
        ],
    },
}

_SAVE_INCOMPLETE_PROPOSAL_DEF: dict[str, Any] = {
    "name": "save_incomplete_proposal",
    "description": (
        "Save partial completion data for a task to the agent's internal DB "
        "and schedule a morning proposal to the manager at 08:30 МСК. "
        "Call this after approving a partially-complete task when some items from the "
        "task description were not covered by the result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task_id": {"type": "integer"},
            "task_title": {"type": "string"},
            "missing_parts": {"type": "string", "description": "What was not done — from task description"},
            "responsible_id": {"type": "integer"},
            "responsible_dialog_id": {"type": "string"},
        },
        "required": ["task_id", "missing_parts"],
    },
}

_DELETE_INCOMPLETE_PROPOSAL_DEF: dict[str, Any] = {
    "name": "delete_incomplete_proposal",
    "description": (
        "Delete a saved incomplete proposal from the agent's internal DB "
        "after the manager agreed to create a new task from it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "proposal_id": {"type": "integer"},
        },
        "required": ["proposal_id"],
    },
}

_SAVE_RESPONSIBLE_RESPONSE_DEF: dict[str, Any] = {
    "name": "save_responsible_response",
    "description": (
        "Save the responsible person's explanation for a pending incomplete proposal. "
        "Call this when the responsible replies to a pending_responsible_question "
        "to store their answer so it appears in the morning proposal to the manager."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "proposal_id": {"type": "integer"},
            "response_text": {"type": "string", "description": "Responsible's explanation text"},
        },
        "required": ["proposal_id", "response_text"],
    },
}

_BITRIX_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _TASK_CREATE_DRAFT_DEF,
    _SAVE_INCOMPLETE_PROPOSAL_DEF,
    _DELETE_INCOMPLETE_PROPOSAL_DEF,
    _SAVE_RESPONSIBLE_RESPONSE_DEF,
]

# Tools in the registry that the LLM can call directly.
_LLM_TOOL_NAMES = frozenset({"portal_search", "bitrix_api", "current_user_profile", "bitrix_notify_users"})


@dataclass(frozen=True)
class IncompleteProposal:
    task_id: int
    missing_parts: str
    task_title: str = ""
    responsible_id: int | None = None
    responsible_dialog_id: str = ""


class BitrixProposalService:
    """Owns proposal persistence and context loading.

    Delivery is now handled by the orchestrator via AgentQueue — the morning proposals
    trigger is published by the scheduler to the bitrix24 queue and the specialist
    builds/sends the message through the orchestrator's channel.
    """

    def __init__(
        self,
        *,
        store: Any,
        settings: Settings,
    ) -> None:
        self._store = store
        self._settings = settings

    def save(self, proposal: IncompleteProposal, scheduled_for: datetime) -> int:
        return self._store.save_proposal(
            task_id=proposal.task_id,
            task_title=proposal.task_title,
            missing_parts=proposal.missing_parts,
            responsible_id=proposal.responsible_id,
            responsible_dialog_id=proposal.responsible_dialog_id,
            scheduled_for=scheduled_for.isoformat(),
        )

    def delete(self, proposal_id: int) -> None:
        self._store.delete_proposal(proposal_id)

    def update_response(self, proposal_id: int, response_text: str) -> None:
        self._store.update_responsible_response(proposal_id, response_text)

    def context_for(self, user_id: int | None) -> dict[str, Any]:
        manager_id = self._settings.task_proposal_manager_bitrix_id
        if manager_id and user_id == manager_id:
            proposed = [p for p in self._store.get_proposals_for_manager() if p.get("status") == "proposed"]
            return {"pending_manager_proposals": proposed} if proposed else {}
        if user_id:
            pending = self._store.get_pending_for_responsible(user_id)
            return {"pending_responsible_question": pending} if pending else {}
        return {}


class Bitrix24Specialist(BaseSpecialist):
    max_steps = 7
    action_prefix = "bitrix"

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        agent_tools: list[AgentTool] | None = None,
        llm: BitrixAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        store: Any | None = None,
        proposals: BitrixProposalService | None = None,
        bitrix_task_client: BitrixTaskPort | None = None,
        settings: Settings | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self._proposals = proposals
        self._bitrix_task_client = bitrix_task_client
        self._settings_for_qc = settings
        super().__init__(
            manifest,
            knowledge_base=knowledge_base,
            skill_store=skill_store,
            retriever=retriever,
            agent_tools=agent_tools,
            llm=llm,
            scheduler=scheduler,
            store=store,
            trace_recorder=trace_recorder,
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        bitrix_client: BitrixToolClientPort | None = None,
        portal_search_index: PortalSearchPort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
        bitrix_retriever: HybridKnowledgeRetriever | None = None,
        bitrix_llm: BitrixAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        bitrix_store: Any | None = None,
        settings: Settings | None = None,
        trace_recorder: TraceRecorder | None = None,
        **_: object,
    ) -> Bitrix24Specialist:
        _settings = settings or get_settings()
        proposals = BitrixProposalService(store=bitrix_store, settings=_settings) if bitrix_store is not None else None
        tools: list[AgentTool] = [
            PortalSearchTool(portal_search=portal_search_index),
            BitrixApiTool(
                client=bitrix_client,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=_settings.agent_dry_run,
            ),
            CurrentUserProfileTool(client=bitrix_client),
            NotifyUsersTool(client=bitrix_client),
            ResolveUserTool(client=bitrix_client),
            ResolveProjectTool(client=bitrix_client),
        ]
        return cls(
            manifest,
            retriever=bitrix_retriever,
            agent_tools=tools,
            llm=bitrix_llm or BitrixLLMService(settings=_settings),
            scheduler=scheduler,
            proposals=proposals,
            store=bitrix_store,
            bitrix_task_client=bitrix_client,
            settings=_settings,
            trace_recorder=trace_recorder,
        )

    async def handle(self, task: AgentTask) -> AgentResult:
        """Dispatch quality-control events to internal handler; all else goes to LLM loop."""
        bitrix_event_type = str(task.context.get("bitrix_event_type") or "").upper()
        if (
            task.request == "quality_control"
            and bitrix_event_type in TASK_QUALITY_WEBHOOK_EVENTS
            and self._bitrix_task_client is not None
            and self._settings_for_qc is not None
        ):
            return await handle_quality_control_task(
                self,
                task,
                bitrix=self._bitrix_task_client,
                settings=self._settings_for_qc,
            )
        return await super().handle(task)

    def tool_definitions(self) -> list[dict]:
        # Expose only LLM-facing registry tools; resolve_user/resolve_project are internal.
        toolset_defs = [t.definition().model_dump() for t in self._tool_registry.values() if t.name in _LLM_TOOL_NAMES]
        return [*toolset_defs, *_BITRIX_TOOL_DEFINITIONS]

    async def _task_create_draft(self, task: AgentTask, args: dict) -> BitrixTaskCreateDraft:
        draft = build_task_create_draft_from_args(task, args)
        resolution = await self._resolve_task_create_draft(draft)
        if resolution is None:
            return draft
        return build_task_create_draft_from_args(task, args, resolution=resolution)

    async def _execute_tool_call(
        self,
        tool_call: BitrixLLMToolCall,
        task: AgentTask,
    ) -> tuple[ToolResult | None, ActionRecord | None, list[ActionRecord]]:
        if tool_call.name == "none":
            return None, None, []

        if tool_call.name == "task_create_draft":
            draft = await self._task_create_draft(task, tool_call.args)
            result = _tool_result_from_task_create_draft(draft)
            action = ActionRecord(
                name="bitrix_task_create_draft", status=result.status, details=draft.as_action_details()
            )
            return result, action, _approval_actions_from_task_create_draft(draft, self.manifest)

        if tool_call.name == "save_incomplete_proposal":
            result = self._save_incomplete_proposal(tool_call.args)
            return (
                result,
                ActionRecord(name="bitrix_save_incomplete_proposal", status=result.status, details=result.model_dump()),
                [],
            )

        if tool_call.name == "delete_incomplete_proposal":
            result = self._delete_incomplete_proposal(tool_call.args)
            return (
                result,
                ActionRecord(
                    name="bitrix_delete_incomplete_proposal", status=result.status, details=result.model_dump()
                ),
                [],
            )

        if tool_call.name == "save_responsible_response":
            result = self._save_responsible_response(tool_call.args)
            return (
                result,
                ActionRecord(
                    name="bitrix_save_responsible_response", status=result.status, details=result.model_dump()
                ),
                [],
            )

        # Registry tools: portal_search, bitrix_api, current_user_profile, send_message, notify_users
        return await super()._execute_tool_call(tool_call, task)

    def _save_incomplete_proposal(self, args: dict[str, Any]) -> ToolResult:
        if self._proposals is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="save_incomplete_proposal",
                error="proposals service is not configured",
            )
        task_id = optional_int(args.get("task_id"))
        if task_id is None:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL, tool="save_incomplete_proposal", error="task_id is required"
            )
        missing_parts = str(args.get("missing_parts") or "").strip()
        if not missing_parts:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="save_incomplete_proposal",
                error="missing_parts is required and must not be empty",
            )
        proposal = IncompleteProposal(
            task_id=task_id,
            task_title=str(args.get("task_title") or ""),
            missing_parts=missing_parts,
            responsible_id=optional_int(args.get("responsible_id")),
            responsible_dialog_id=str(args.get("responsible_dialog_id") or ""),
        )
        run_date = _next_morning_830()
        proposal_id = self._proposals.save(proposal, run_date)
        return ToolResult(
            status=ToolStatus.OK,
            tool="save_incomplete_proposal",
            data={"proposal_id": proposal_id, "scheduled_for": run_date.isoformat()},
        )

    def _delete_incomplete_proposal(self, args: dict[str, Any]) -> ToolResult:
        if self._proposals is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="delete_incomplete_proposal",
                error="proposals service is not configured",
            )
        proposal_id = optional_int(args.get("proposal_id"))
        if proposal_id is None:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="delete_incomplete_proposal",
                error="proposal_id is required",
            )
        self._proposals.delete(proposal_id)
        return ToolResult(status=ToolStatus.OK, tool="delete_incomplete_proposal", data={"proposal_id": proposal_id})

    def _save_responsible_response(self, args: dict[str, Any]) -> ToolResult:
        if self._proposals is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="save_responsible_response",
                error="proposals service is not configured",
            )
        proposal_id = optional_int(args.get("proposal_id"))
        if proposal_id is None:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="save_responsible_response",
                error="proposal_id is required",
            )
        response_text = str(args.get("response_text") or "").strip()
        if not response_text:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="save_responsible_response",
                error="response_text is required and must not be empty",
            )
        self._proposals.update_response(proposal_id, response_text)
        return ToolResult(
            status=ToolStatus.OK,
            tool="save_responsible_response",
            data={"proposal_id": proposal_id},
        )

    async def _load_extra_context(self, task: AgentTask) -> tuple[AgentTask, dict]:
        permission_context = await self._load_permission_context(task)
        proposal_context: dict[str, Any] = (
            self._proposals.context_for(optional_int(task.user.id)) if self._proposals is not None else {}
        )
        merged_task = task.model_copy(update={"context": {**task.context, **permission_context, **proposal_context}})
        extra_details = {
            "bitrix_current_user_profile_status": _tool_context_status(
                permission_context.get("bitrix_current_user_profile")
            ),
            "permission_policy_topics": [
                item.get("topic")
                for item in permission_context.get("permission_policy_context", [])
                if isinstance(item, dict)
            ],
        }
        return merged_task, extra_details

    async def _load_permission_context(self, task: AgentTask) -> dict:
        profile_result = await self._current_user_profile_result(task)
        policy_hits = (
            self.retriever.search(
                self.manifest,
                _permission_policy_query(task.request, profile_result),
                limit=3,
                topic="user_permissions",
            )
            if self.retriever is not None
            else []
        )
        return {
            "bitrix_current_user_profile": profile_result.model_dump(),
            "permission_policy_context": [
                {
                    "topic": hit.chunk.topic,
                    "section": hit.chunk.section,
                    "score": hit.score,
                    "text": hit.chunk.text[:1200],
                }
                for hit in policy_hits
            ],
        }

    async def _current_user_profile_result(self, task: AgentTask) -> ToolResult:
        user_id = optional_int(task.user.id)
        tool = self._tool_registry.get("current_user_profile")
        if user_id is None or tool is None:
            return ToolResult(
                status=ToolStatus.NOT_AVAILABLE,
                tool="current_user_profile",
                error="current Bitrix user id is not available",
            )
        try:
            return await tool.execute({}, user_id=user_id)
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool="current_user_profile",
                error=f"{type(exc).__name__}: {exc}",
                data={"user_id": user_id},
            )

    async def _resolve_user_field(self, query: str, fields: dict[str, Any]) -> tuple[int | None, str, list[str]]:
        if not query or "RESPONSIBLE_ID" in fields:
            return None, "", []
        tool = self._tool_registry.get("resolve_user")
        if tool is None:
            return None, "", ["Bitrix API недоступен для поиска сотрудника."]
        result = await tool.execute({"query": query})
        if result.status == "ok":
            candidate = result.data.get("candidate") if isinstance(result.data, dict) else None
            if isinstance(candidate, dict):
                return (
                    optional_int(candidate.get("id")),
                    f"Ответственный найден в Bitrix: {candidate.get('label') or query}.",
                    [],
                )
        if result.status == "ambiguous":
            return None, "", [_format_ambiguous_note("Найдено несколько сотрудников", result.data.get("candidates"))]
        if result.status == "not_found":
            return None, "", [f"Не нашёл сотрудника в Bitrix по запросу `{query}`."]
        return None, "", [f"Не смог проверить сотрудника через Bitrix: {result.status}."]

    async def _resolve_project_field(self, query: str, fields: dict[str, Any]) -> tuple[int | None, str, list[str]]:
        if not query or "GROUP_ID" in fields:
            return None, "", []
        tool = self._tool_registry.get("resolve_project")
        if tool is None:
            return None, "", ["Bitrix API недоступен для поиска проекта."]
        result = await tool.execute({"query": query})
        if result.status == "ok":
            candidate = result.data.get("candidate") if isinstance(result.data, dict) else None
            if isinstance(candidate, dict):
                return (
                    optional_int(candidate.get("id")),
                    f"Проект найден в Bitrix: {candidate.get('label') or query}.",
                    [],
                )
        if result.status == "ambiguous":
            return None, "", [_format_ambiguous_note("Найдено несколько проектов", result.data.get("candidates"))]
        if result.status == "not_found":
            return None, "", [f"Не нашёл проект в Bitrix по запросу `{query}`."]
        return None, "", [f"Не смог проверить проект через Bitrix: {result.status}."]

    async def _resolve_task_create_draft(
        self,
        draft: BitrixTaskCreateDraft,
    ) -> BitrixTaskCreateResolution | None:
        fields = _draft_fields(draft)
        responsible_id, responsible_note, user_notes = await self._resolve_user_field(draft.responsible_query, fields)
        group_id, group_note, project_notes = await self._resolve_project_field(draft.project_query, fields)
        notes = user_notes + project_notes
        if responsible_id is None and group_id is None and not notes:
            return None
        return BitrixTaskCreateResolution(
            responsible_id=responsible_id,
            responsible_note=responsible_note,
            group_id=group_id,
            group_note=group_note,
            notes=notes,
        )

    def _llm_failure_result(self, message: str):
        return llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Bitrix24 specialist is an LLM subagent; backend tools only execute and validate selected tool calls.",
            "Knowledge context is selected through hybrid retrieval over the agent package.",
        ]


def _format_proposal_message(proposal: dict[str, Any]) -> str:
    task_id = proposal.get("task_id")
    task_title = proposal.get("task_title") or f"задача #{task_id}"
    missing_parts = proposal.get("missing_parts") or ""
    responsible_response = proposal.get("responsible_response") or ""
    parts = [
        f"Задача [{task_title}] (#{task_id}) была выполнена частично.",
        f"Что не сделано: {missing_parts}",
    ]
    if responsible_response:
        parts.append(f"Пояснение исполнителя: {responsible_response}")
    parts.append("Предлагаю создать задачу на оставшиеся работы. Если согласны — сообщите, я подготовлю черновик.")
    return "\n\n".join(parts)


def _next_morning_830() -> datetime:
    now = datetime.now(MOSCOW_TZ)
    target = now.replace(hour=8, minute=30, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return target


def _format_ambiguous_note(prefix: str, candidates: object) -> str:
    if not isinstance(candidates, list):
        return prefix + "; нужно уточнение."
    labels = []
    for candidate in candidates[:5]:
        if isinstance(candidate, dict):
            label = candidate.get("label") or candidate.get("id")
            if label:
                labels.append(str(label))
    if not labels:
        return prefix + "; нужно уточнение."
    return prefix + ": " + "; ".join(labels) + ". Уточни нужный вариант."


def _permission_policy_query(request: str, profile_result: ToolResult) -> str:
    parts = [request, "права разрешения роли администратор монтажники отдел создание закрытие задачи"]
    profile = profile_result.data.get("profile") if isinstance(profile_result.data, dict) else None
    if isinstance(profile, dict):
        for key in ("work_position", "user_type", "label"):
            value = str(profile.get(key) or "").strip()
            if value:
                parts.append(value)
        department_ids = profile.get("department_ids")
        if isinstance(department_ids, list):
            parts.extend(f"department_id:{item}" for item in department_ids)
        if profile.get("is_admin") is True:
            parts.append("администратор admin")
    return " ".join(parts)


def _tool_context_status(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("status") or "")
    return ""


def _tool_result_from_task_create_draft(draft: BitrixTaskCreateDraft) -> ToolResult:
    return ToolResult(
        status=ToolStatus.READY if draft.is_ready else ToolStatus.CONTRACT_VIOLATION,
        tool="task_create_draft",
        data=draft.as_action_details(),
        error=None if draft.is_ready else "LLM called task_create_draft with arguments outside the tool contract.",
    )


def _approval_actions_from_task_create_draft(
    draft: BitrixTaskCreateDraft, manifest: AgentManifest
) -> list[ActionRecord]:
    if not draft.is_ready:
        return []
    needs_approval = "bitrix_write" in manifest.approval_required
    return [
        ActionRecord(
            name="bitrix_api",
            status="approval_required" if needs_approval else "allow",
            details={
                "method": draft.method,
                "params": draft.params,
                "policy": {
                    "decision": "confirm" if needs_approval else "allow",
                    "reason": "manifest.approval_required",
                },
                "summary": draft.summary,
                "specialist_id": manifest.id,
            },
        )
    ]


def _draft_fields(draft: BitrixTaskCreateDraft) -> dict:
    fields = draft.params.get("fields")
    return fields if isinstance(fields, dict) else {}
