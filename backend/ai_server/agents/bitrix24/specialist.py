from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from typing import Any

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.bitrix24.llm import (
    BitrixAgentLLM,
    BitrixLLMService,
    _format_task_close_confirm_answer,
    _format_task_close_draft_answer,
    llm_failure_result,
)
from ai_server.agents.bitrix24.draft_confirmation import draft_confirmation_phrase, matches_draft_confirmation
from ai_server.agents.bitrix24.ports import ProposalStorePort, TaskDraftStorePort
from ai_server.agents.bitrix24.quality_control import (
    TASK_QUALITY_WEBHOOK_EVENTS,
    handle_quality_control_task,
)
from ai_server.agents.bitrix24.tools import (
    BitrixApiTool,
    BitrixMyTasksTool,
    BitrixProjectSearchTool,
    BitrixTaskSearchTool,
    BitrixWarehouseSearchTool,
    CalendarEventConfirmTool,
    CalendarEventDiscardTool,
    CalendarEventDraftTool,
    DeleteIncompleteProposalTool,
    PortalSearchTool,
    ProjectCreateConfirmTool,
    ProjectCreateDiscardTool,
    ProjectCreateDraftTool,
    SaveIncompleteProposalTool,
    SaveResponsibleResponseTool,
    TaskCloseConfirmTool,
    TaskCloseControlGetTool,
    TaskCloseControlUpdateTool,
    TaskCloseDiscardTool,
    TaskCloseDraftTool,
    TaskCloseReportIncidentTool,
    TaskCreateConfirmTool,
    TaskCreateDraftTool,
    TaskDraftDiscardTool,
    proposal_context,
)
from ai_server.agents.ports import SchedulerPort
from ai_server.agents.tool import AgentTool
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.integrations.bitrix.profile import compact_user_profile
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentManifest, AgentResult, AgentTask, ToolResult, ToolStatus
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import Settings, get_settings
from ai_server.skills import SkillStore
from ai_server.tools.bitrix_ports import BitrixTaskPort, BitrixToolClientPort
from ai_server.utils import optional_int

logger = logging.getLogger(__name__)

# Tools in the registry that the LLM can call directly.
_LLM_TOOL_NAMES = frozenset(
    {
        "portal_search",
        "bitrix_warehouse_search",
        "bitrix_my_tasks",
        "bitrix_task_search",
        "bitrix_project_search",
        "bitrix_api",
        "task_create_draft",
        "task_create_confirm",
        "task_draft_discard",
        "task_close_draft",
        "task_close_confirm",
        "task_close_discard",
        "task_close_report_incident",
        "task_close_control_get",
        "task_close_control_update",
        "calendar_event_draft",
        "calendar_event_confirm",
        "calendar_event_discard",
        "project_create_draft",
        "project_create_confirm",
        "project_create_discard",
        "save_incomplete_proposal",
        "delete_incomplete_proposal",
        "save_responsible_response",
    }
)

_FAST_RETURN_READ_TOOLS = frozenset(
    {
        "portal_search",
        "bitrix_warehouse_search",
        "bitrix_my_tasks",
        "bitrix_task_search",
        "bitrix_project_search",
    }
)

_FAST_RETURN_TASK_CREATE_TOOLS = frozenset(
    {
        "task_create_confirm",
        "task_draft_discard",
        "project_create_discard",
    }
)


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
        proposal_store: ProposalStorePort | None = None,
        draft_store: TaskDraftStorePort | None = None,
        bitrix_task_client: BitrixTaskPort | None = None,
        bitrix_user_client: BitrixToolClientPort | None = None,
        settings: Settings | None = None,
        result_publisher: Any | None = None,
        conversation_trace: Any | None = None,
    ) -> None:
        self._proposal_store = proposal_store
        self._draft_store = draft_store
        self._bitrix_task_client = bitrix_task_client
        self._bitrix_user_client = bitrix_user_client
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
            result_publisher=result_publisher,
            conversation_trace=conversation_trace,
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        bitrix_client: BitrixToolClientPort | None = None,
        portal_search_index: Any | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
        bitrix_retriever: HybridKnowledgeRetriever | None = None,
        bitrix_llm: BitrixAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        bitrix_store: Any | None = None,
        orchestrator_store: Any | None = None,
        settings: Settings | None = None,
        specialist_result_publisher: Any | None = None,
        conversation_trace: Any | None = None,
        **_: object,
    ) -> Bitrix24Specialist:
        _settings = settings or get_settings()
        tools: list[AgentTool] = [
            PortalSearchTool(
                portal_search=portal_search_index,
                bitrix_files=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                state_store=orchestrator_store,
                live_fallback_enabled=bitrix_oauth is not None,
                index_max_age_seconds=max(
                    900,
                    (
                        _settings.search_delta_interval_seconds * 4
                        if _settings.search_background_periodic_delta_enabled
                        else _settings.search_background_metadata_interval_seconds * 2
                    ),
                ),
                index_freshness_path=_settings.search_background_state_path,
            ),
            BitrixWarehouseSearchTool(
                client=bitrix_client,
                portal_search=portal_search_index,
                bitrix_oauth=bitrix_oauth,
            ),
            BitrixMyTasksTool(client=bitrix_client, bitrix_oauth=bitrix_oauth),
            BitrixTaskSearchTool(
                client=bitrix_client,
                portal_search=portal_search_index,
                bitrix_oauth=bitrix_oauth,
            ),
            BitrixProjectSearchTool(
                client=bitrix_client,
                portal_search=portal_search_index,
                bitrix_oauth=bitrix_oauth,
            ),
            BitrixApiTool(
                client=bitrix_client,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=_settings.agent_dry_run,
                oauth_required_for_writes=_settings.bitrix_oauth_required_for_writes,
            ),
            TaskCreateDraftTool(
                store=bitrix_store,
                project_client=bitrix_client,
                portal_search=portal_search_index,
                bitrix_oauth=bitrix_oauth,
            ),
            TaskCreateConfirmTool(
                store=bitrix_store,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=_settings.agent_dry_run,
                oauth_required_for_writes=_settings.bitrix_oauth_required_for_writes,
                draft_ttl_minutes=_settings.bitrix_task_draft_ttl_minutes,
            ),
            TaskDraftDiscardTool(store=bitrix_store),
            TaskCloseDraftTool(store=bitrix_store, read_client=bitrix_client, bitrix_oauth=bitrix_oauth),
            TaskCloseConfirmTool(
                store=bitrix_store,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=_settings.agent_dry_run,
                oauth_required_for_writes=_settings.bitrix_oauth_required_for_writes,
                draft_ttl_minutes=_settings.bitrix_task_draft_ttl_minutes,
            ),
            TaskCloseDiscardTool(store=bitrix_store),
            TaskCloseReportIncidentTool(
                client=bitrix_client,
                portal_search=portal_search_index,
                settings=_settings,
            ),
            TaskCloseControlGetTool(store=bitrix_store, user_client=bitrix_client),
            TaskCloseControlUpdateTool(
                store=bitrix_store,
                user_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
            ),
            CalendarEventDraftTool(store=bitrix_store),
            CalendarEventConfirmTool(
                store=bitrix_store,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=_settings.agent_dry_run,
                oauth_required_for_writes=_settings.bitrix_oauth_required_for_writes,
                draft_ttl_minutes=_settings.bitrix_task_draft_ttl_minutes,
            ),
            CalendarEventDiscardTool(store=bitrix_store),
            ProjectCreateDraftTool(store=bitrix_store),
            ProjectCreateConfirmTool(
                store=bitrix_store,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=_settings.agent_dry_run,
                oauth_required_for_writes=_settings.bitrix_oauth_required_for_writes,
                draft_ttl_minutes=_settings.bitrix_task_draft_ttl_minutes,
            ),
            ProjectCreateDiscardTool(store=bitrix_store),
            SaveIncompleteProposalTool(store=bitrix_store),
            DeleteIncompleteProposalTool(store=bitrix_store),
            SaveResponsibleResponseTool(store=bitrix_store),
        ]
        return cls(
            manifest,
            retriever=bitrix_retriever,
            agent_tools=tools,
            llm=bitrix_llm or BitrixLLMService(settings=_settings),
            scheduler=scheduler,
            proposal_store=bitrix_store,
            draft_store=bitrix_store,
            store=bitrix_store,
            bitrix_task_client=bitrix_client,
            bitrix_user_client=bitrix_client,
            settings=_settings,
            result_publisher=specialist_result_publisher,
            conversation_trace=conversation_trace,
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
        result = await super().handle(task)
        return _enforce_task_close_response(result, settings=self._settings_for_qc)

    async def _execute_tool_call(
        self,
        tool_call: Any,
        task: AgentTask,
    ) -> tuple[ToolResult | None, Any | None, list[Any]]:
        confirmation_tools = {
            "task_create_confirm",
            "task_close_confirm",
            "calendar_event_confirm",
            "project_create_confirm",
        }
        is_admin_confirm = tool_call.name == "task_close_control_update" and str(
            (tool_call.args or {}).get("operation") or ""
        ) == "confirm"
        if tool_call.name in confirmation_tools or is_admin_confirm:
            draft = task.context.get("pending_task_draft") if isinstance(task.context, dict) else None
            if not matches_draft_confirmation(str(task.request or ""), draft):
                return (
                    ToolResult(
                        status=ToolStatus.DENIED,
                        tool=tool_call.name,
                        error=f"Draft confirmation requires the exact phrase: {draft_confirmation_phrase((draft or {}).get('_draft_type'))}.",
                    ),
                    None,
                    [],
                )
        if tool_call.name == "task_create_draft":
            args = _draft_args_with_metadata(
                _task_create_args_with_actor_label(dict(tool_call.args or {}), task),
                task,
            )
            tool_call = SimpleNamespace(
                name=tool_call.name,
                args=args,
                summary=getattr(tool_call, "summary", ""),
            )
        elif tool_call.name == "calendar_event_draft":
            args = _draft_args_with_metadata(
                _calendar_event_args_with_actor_label(dict(tool_call.args or {}), task),
                task,
            )
            tool_call = SimpleNamespace(
                name=tool_call.name,
                args=args,
                summary=getattr(tool_call, "summary", ""),
            )
        elif tool_call.name == "project_create_draft":
            args = _draft_args_with_metadata(
                _project_create_args_with_actor_context(dict(tool_call.args or {}), task),
                task,
            )
            tool_call = SimpleNamespace(
                name=tool_call.name,
                args=args,
                summary=getattr(tool_call, "summary", ""),
            )
        elif tool_call.name == "task_close_draft":
            args = _draft_args_with_metadata(dict(tool_call.args or {}), task)
            tool_call = SimpleNamespace(
                name=tool_call.name,
                args=args,
                summary=getattr(tool_call, "summary", ""),
            )
        elif tool_call.name == "bitrix_warehouse_search":
            args = _warehouse_args_with_default_products(dict(tool_call.args or {}), task)
            tool_call = SimpleNamespace(
                name=tool_call.name,
                args=args,
                summary=getattr(tool_call, "summary", ""),
            )
        elif tool_call.name in {"task_close_control_get", "task_close_control_update"}:
            args = _args_with_actor_admin_context(dict(tool_call.args or {}), task)
            if tool_call.name == "task_close_control_update":
                args = _draft_args_with_metadata(args, task)
            tool_call = SimpleNamespace(
                name=tool_call.name,
                args=args,
                summary=getattr(tool_call, "summary", ""),
            )
        return await super()._execute_tool_call(tool_call, task)

    def _terminal_response_metadata(
        self,
        *,
        tool_call: Any,
        result: ToolResult | None,
        action: Any | None,
        approvals: list[Any],
        task: AgentTask,
    ) -> dict[str, Any] | None:
        if result is None or approvals:
            return None
        if tool_call.name not in _FAST_RETURN_READ_TOOLS and tool_call.name not in _FAST_RETURN_TASK_CREATE_TOOLS:
            return None
        if _is_ambiguous_read_result(tool_call.name, result):
            return None
        if result.status == ToolStatus.OK:
            reason = (
                "task_create_tool_success"
                if tool_call.name in _FAST_RETURN_TASK_CREATE_TOOLS
                else "read_only_tool_success"
            )
            return {
                "fast_return_reason": reason,
                "terminal_tool": tool_call.name,
                "tool_status": str(result.status),
            }
        if result.status == ToolStatus.DENIED and _is_oauth_authorization_result(result):
            return {
                "fast_return_reason": "read_only_oauth_authorization_required",
                "terminal_tool": tool_call.name,
                "tool_status": str(result.status),
            }
        return None

    def tool_definitions(self) -> list[dict]:
        return [t.definition().model_dump() for t in self._tool_registry.values() if t.name in _LLM_TOOL_NAMES]

    async def _load_extra_context(self, task: AgentTask) -> tuple[AgentTask, dict]:
        permission_context = await self._load_permission_context(task)
        manager_id = self._settings_for_qc.task_proposal_manager_bitrix_id if self._settings_for_qc else None
        proposal_ctx = proposal_context(self._proposal_store, optional_int(task.user.id), manager_id)
        draft_context: dict[str, Any] = {}
        dialog_key = str(task.context.get("dialog_key") or "")
        if dialog_key and self._draft_store is not None:
            pending = await self._draft_store.get_task_draft(
                dialog_key,
                ttl_minutes=self._settings_for_qc.bitrix_task_draft_ttl_minutes if self._settings_for_qc else None,
            )
            if pending:
                draft_context = {"pending_task_draft": pending}
        merged_task = task.model_copy(
            update={"context": {**task.context, **permission_context, **proposal_ctx, **draft_context}}
        )
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
        if user_id is None or self._bitrix_user_client is None:
            return ToolResult(
                status=ToolStatus.NOT_AVAILABLE,
                tool="current_user_profile",
                error="current Bitrix user id is not available",
            )
        try:
            user = await self._bitrix_user_client.get_user(user_id)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="current_user_profile",
                error=str(exc),
                data={"user_id": user_id},
            )
        if user is None:
            return ToolResult(status=ToolStatus.NOT_FOUND, tool="current_user_profile", data={"user_id": user_id})
        return ToolResult(
            status=ToolStatus.OK,
            tool="current_user_profile",
            data={"user_id": user_id, "profile": compact_user_profile(user)},
        )

    def _llm_failure_result(self, message: str):
        return llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Bitrix24 specialist is an LLM subagent; backend tools only execute and validate selected tool calls.",
            "Knowledge context is selected through hybrid retrieval over the agent package.",
        ]


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


def _task_create_args_with_actor_label(args: dict[str, Any], task: AgentTask) -> dict[str, Any]:
    user_id = optional_int(task.user.id)
    responsible_id = optional_int(args.get("responsible_id"))
    responsible_self = _truthy(args.get("responsible_self"))
    if not responsible_self and (user_id is None or responsible_id != user_id):
        return args
    label = _current_user_label(task)
    if not label:
        return args
    updated = dict(args)
    if not str(updated.get("responsible_name") or updated.get("responsible_label") or "").strip():
        updated["responsible_name"] = label
    if not any(updated.get(key) for key in ("group_id", "project_name", "group_name")):
        updated["project_name"] = _personal_project_name(label)
        updated["_default_personal_project"] = True
    return updated


def _personal_project_name(label: str) -> str:
    parts = [part for part in compact_user_label(label).split() if part]
    if len(parts) >= 2:
        return f"{parts[0]} {parts[1]}"
    return label.strip()


def compact_user_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label or "")).strip()


def _calendar_event_args_with_actor_label(args: dict[str, Any], task: AgentTask) -> dict[str, Any]:
    if str(args.get("owner_name") or args.get("owner_label") or "").strip():
        return args
    if args.get("attendee_ids") or args.get("attendees"):
        return args
    label = _current_user_label(task)
    if label:
        return {**args, "owner_name": label}
    return args


def _project_create_args_with_actor_context(args: dict[str, Any], task: AgentTask) -> dict[str, Any]:
    profile = _current_user_profile(task)
    actor_name = _current_user_label(task)
    actor_is_admin = bool(profile.get("is_admin"))
    return {**args, "_actor_name": actor_name, "_actor_is_admin": actor_is_admin}


def _warehouse_args_with_default_products(args: dict[str, Any], task: AgentTask) -> dict[str, Any]:
    if args.get("include_products") is True:
        return args
    if _warehouse_request_implies_stock(str(task.request or "")):
        return {**args, "include_products": True, "product_limit": int(args.get("product_limit") or 10)}
    return args


def _warehouse_request_implies_stock(request: str) -> bool:
    normalized = request.casefold().replace("ё", "е")
    if any(marker in normalized for marker in ("остат", "налич", "что есть", "что находится")):
        return True
    return bool(re.search(r"\b(?:битрикс\s+)?покажи(?:те)?\s+(?:мне\s+)?склад\b", normalized))


def _args_with_actor_admin_context(
    args: dict[str, Any],
    task: AgentTask,
) -> dict[str, Any]:
    profile = _current_user_profile(task)
    actor_is_admin = bool(profile.get("is_admin"))
    return {**args, "_actor_is_admin": actor_is_admin}


def _draft_args_with_metadata(args: dict[str, Any], task: AgentTask) -> dict[str, Any]:
    return {
        **args,
        "_original_request": str(task.request or ""),
        "_draft_user_id": optional_int(task.user.id) if task.user.id is not None else None,
        "_draft_specialist": "bitrix24",
    }


def _current_user_label(task: AgentTask) -> str:
    display_name = str(task.user.display_name or "").strip()
    if display_name:
        return display_name
    profile_result = task.context.get("bitrix_current_user_profile")
    if not isinstance(profile_result, dict):
        return ""
    data = profile_result.get("data")
    if not isinstance(data, dict):
        return ""
    profile = data.get("profile")
    if not isinstance(profile, dict):
        return ""
    return str(profile.get("label") or "").strip()


def _current_user_profile(task: AgentTask) -> dict[str, Any]:
    profile_result = task.context.get("bitrix_current_user_profile")
    if not isinstance(profile_result, dict):
        return {}
    data = profile_result.get("data")
    if not isinstance(data, dict):
        return {}
    profile = data.get("profile")
    return profile if isinstance(profile, dict) else {}


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "да", "on"}
    return bool(value)


def _is_oauth_authorization_result(result: ToolResult) -> bool:
    data = result.data if isinstance(result.data, dict) else {}
    return bool(data.get("oauth_required") or isinstance(data.get("authorization"), dict))


def _is_ambiguous_read_result(tool_name: str, result: ToolResult) -> bool:
    if result.status != ToolStatus.OK:
        return False
    data = result.data if isinstance(result.data, dict) else {}
    if tool_name == "bitrix_warehouse_search":
        matches = data.get("matches")
        return isinstance(matches, list) and len(matches) > 1
    if tool_name == "bitrix_project_search":
        items = data.get("items")
        return isinstance(items, list) and len(items) > 1
    return False


def _enforce_task_close_response(result: AgentResult, *, settings: Settings | None = None) -> AgentResult:
    for action in reversed(result.actions_taken):
        if str(action.status) != "ok":
            continue
        details = action.details if isinstance(action.details, dict) else {}
        data = details.get("data") if isinstance(details.get("data"), dict) else {}
        if action.name == "task_close_draft":
            answer = _format_task_close_draft_answer(data)
            return result.model_copy(update={"status": "needs_human", "answer": answer})
        if action.name == "task_close_confirm":
            answer = _format_task_close_confirm_answer(
                data,
                portal_base_url=settings.bitrix_portal_base_url if settings is not None else "",
            )
            return result.model_copy(update={"status": "completed", "answer": answer})
        if action.name == "task_close_discard":
            return result.model_copy(update={"status": "completed", "answer": "Черновик закрытия задачи удалён."})
        if action.name == "task_close_control_update":
            operation = str(data.get("operation") or "")
            if operation == "prepare":
                return result.model_copy(
                    update={"status": "needs_human", "answer": _format_admin_change_draft_answer(data)}
                )
            if operation == "discard":
                return result.model_copy(
                    update={"status": "completed", "answer": "Черновик изменения настроек удалён."}
                )
            if operation == "confirm":
                return result.model_copy(update={"status": "completed", "answer": "Изменение настроек подтверждено."})
    return result


def _format_admin_change_draft_answer(data: dict[str, Any]) -> str:
    draft = data.get("draft") if isinstance(data.get("draft"), dict) else {}
    field = str(draft.get("field") or draft.get("action") or "настройка")
    target = str(draft.get("target_user_name") or draft.get("target_user_id") or "").strip()
    subject = f"{field} ({target})" if target else field
    answer = (
        f"Подготовлен черновик изменения: {subject}; "
        f"было — {draft.get('old_value')!s}, станет — {draft.get('new_value')!s}. "
        "Черновик действует 15 минут. Подтвердите или отмените изменение."
    )
    return f"{answer}\n\nДля подтверждения отправьте фразу: «{draft_confirmation_phrase(draft.get('_draft_type'))}»."
