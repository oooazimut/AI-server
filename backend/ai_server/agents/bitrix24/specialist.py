from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from ai_server.agents.base import BaseSpecialist, _trace_now_iso
from ai_server.agents.bitrix24.ports import TaskDraftStorePort
from ai_server.agents.bitrix24.tools import (
    BitrixApiTool,
    BitrixMyTasksTool,
    BitrixProjectSearchTool,
    BitrixTaskSearchTool,
    BitrixWarehouseSearchTool,
    CalendarEventConfirmTool,
    CalendarEventDiscardTool,
    CalendarEventDraftTool,
    PortalSearchTool,
    ProjectCreateConfirmTool,
    ProjectCreateDiscardTool,
    ProjectCreateDraftTool,
    TaskCloseConfirmTool,
    TaskCloseControlGetTool,
    TaskCloseControlUpdateTool,
    TaskCloseDiscardTool,
    TaskCloseDraftTool,
    TaskCloseReportIncidentTool,
    TaskCreateConfirmTool,
    TaskCreateDraftTool,
    TaskDraftDiscardTool,
)
from ai_server.agents.ports import SchedulerPort
from ai_server.agents.tool import AgentTool
from ai_server.capability_registry import build_capability_registry, registry_tool, validate_tool_arguments
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.integrations.bitrix.profile import compact_user_profile
from ai_server.models import (
    ActionRecord,
    AgentManifest,
    AgentResult,
    AgentTask,
    ModelUsageRecord,
    ToolResult,
    ToolStatus,
)
from ai_server.settings import Settings, get_settings
from ai_server.tools.bitrix_ports import BitrixTaskPort, BitrixToolClientPort
from ai_server.utils import optional_int

_STRUCTURED_TOOL_NAMES = frozenset(
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
    }
)

_DRAFT_TOOLS = frozenset(
    {
        "task_create_draft",
        "task_close_draft",
        "calendar_event_draft",
        "project_create_draft",
        "task_close_control_update",
    }
)

_CONFIRM_TOOLS = frozenset(
    {
        "task_create_confirm",
        "task_close_confirm",
        "calendar_event_confirm",
        "project_create_confirm",
    }
)


class Bitrix24Specialist(BaseSpecialist):
    """Version-bound Bitrix executor with no semantic model surface."""

    action_prefix = "bitrix"

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        agent_tools: list[AgentTool] | None = None,
        scheduler: SchedulerPort | None = None,
        store: Any | None = None,
        draft_store: TaskDraftStorePort | None = None,
        bitrix_task_client: BitrixTaskPort | None = None,
        bitrix_user_client: BitrixToolClientPort | None = None,
        settings: Settings | None = None,
        confirmation_phrase_renderer: Any | None = None,
        confirmation_matcher: Any | None = None,
        result_publisher: Any | None = None,
        conversation_trace: Any | None = None,
    ) -> None:
        self._draft_store = draft_store
        self._bitrix_task_client = bitrix_task_client
        self._bitrix_user_client = bitrix_user_client
        self._settings = settings
        self._confirmation_phrase_renderer = confirmation_phrase_renderer
        self._confirmation_matcher = confirmation_matcher
        super().__init__(
            manifest,
            agent_tools=agent_tools,
            scheduler=scheduler,
            store=store,
            result_publisher=result_publisher,
            conversation_trace=conversation_trace,
        )

    async def get_active_draft(self, dialog_key: str) -> dict[str, Any] | None:
        if not dialog_key or self._draft_store is None:
            return None
        return await self._draft_store.get_task_draft(
            dialog_key,
            ttl_minutes=self._settings.bitrix_task_draft_ttl_minutes if self._settings else None,
        )

    async def discard_active_draft(self, dialog_key: str, *, expected_draft_id: str) -> bool:
        draft = await self.get_active_draft(dialog_key)
        if not draft or str(draft.get("_draft_id") or "") != expected_draft_id or self._draft_store is None:
            return False
        await self._draft_store.delete_task_draft(
            dialog_key,
            status="cancelled",
            expected_draft_id=expected_draft_id,
            expected_version=int(draft.get("_draft_version") or 0) or None,
        )
        return True

    def capability_registry(self) -> dict[str, Any]:
        definitions = self.tool_definitions()
        available = {str(item.get("name") or "") for item in definitions if isinstance(item, dict)}
        return build_capability_registry(self.manifest, definitions, structured_tool_names=available)

    async def execute_structured_command(self, task: AgentTask, command: dict[str, Any]) -> AgentResult:
        registry = self.capability_registry()
        failure = self._structured_command_failure(command, registry)
        if failure is not None:
            return failure

        tool_name = str(command["tool_name"])
        arguments = dict(command["arguments"])
        context_started_at = _trace_now_iso()
        context_t0 = time.monotonic()
        task, context_details = await self._load_structured_context(task, tool_name)
        task = task.model_copy(update={"context": {**task.context, "structured_command_execution": True}})
        await self._record_timing(
            task,
            stage="structured_context_load",
            started_at=context_started_at,
            elapsed_ms=(time.monotonic() - context_t0) * 1000,
            status="completed",
            tool=tool_name,
        )

        call = SimpleNamespace(name=tool_name, args=arguments, summary="orchestrator structured command")
        tool_started_at = _trace_now_iso()
        tool_t0 = time.monotonic()
        result, action, approvals = await self._execute_tool_call(call, task)
        if result is None:
            result = ToolResult(status=ToolStatus.ERROR, tool=tool_name, error="structured command returned no result")
        await self._record_timing(
            task,
            stage="structured_tool_execute",
            started_at=tool_started_at,
            elapsed_ms=(time.monotonic() - tool_t0) * 1000,
            status=str(result.status),
            tool=tool_name,
            details={"registry_version": registry["registry_version"]},
        )

        terminal_status = _terminal_status(tool_name, result)
        if result.status not in {ToolStatus.OK, ToolStatus.NOT_FOUND}:
            terminal_status = "failed"
        actions = [
            ActionRecord(
                name="bitrix_capability_contract",
                status="ok",
                details={
                    "registry_version": registry["registry_version"],
                    "tool": tool_name,
                    "mode": "structured_command",
                },
            )
        ]
        if action is not None:
            actions.append(action)
        metadata = {
            "terminal": True,
            "answer_is_final": False,
            "safe_to_send": True,
            "fast_return": True,
            "fast_return_reason": "orchestrator_structured_command",
            "terminal_tool": tool_name,
            "terminal_status": terminal_status,
            "registry_version": registry["registry_version"],
            "structured_command": True,
            "command_arguments": arguments,
            "tool_result": result.model_dump(),
            "portal_base_url": self._settings.bitrix_portal_base_url if self._settings else "",
            "permission_context": context_details,
        }
        return AgentResult(
            status="needs_human" if approvals else terminal_status,
            agent_id=self.manifest.id,
            answer="",
            actions_taken=actions,
            actions_requiring_approval=list(approvals),
            model_usage=[
                ModelUsageRecord(
                    agent_id=self.manifest.id,
                    provider="internal",
                    model="structured-bitrix-executor",
                    status="not_used",
                    notes=["No Bitrix model call and no specialist-side response rendering."],
                )
            ],
            confidence=1.0 if result.status == ToolStatus.OK else 0.0,
            logs=["Bitrix executed one exact orchestrator command."],
            metadata=metadata,
        )

    def _structured_command_failure(self, command: dict[str, Any], registry: dict[str, Any]) -> AgentResult | None:
        reason = ""
        details: dict[str, Any] = {}
        if not isinstance(command, dict) or set(command) != {"registry_version", "tool_name", "arguments"}:
            reason = "STRUCTURED_COMMAND_SCHEMA_MISMATCH"
        elif str(command.get("registry_version") or "") != str(registry.get("registry_version") or ""):
            reason = "CAPABILITY_REGISTRY_VERSION_MISMATCH"
        elif not isinstance(command.get("arguments"), dict):
            reason = "STRUCTURED_COMMAND_ARGUMENTS_INVALID"
        else:
            tool = registry_tool(registry, str(command.get("tool_name") or ""))
            if tool is None or not tool.get("structured_command"):
                reason = "STRUCTURED_COMMAND_TOOL_DISABLED"
            else:
                errors = validate_tool_arguments(dict(tool.get("parameters") or {}), command["arguments"])
                if errors:
                    reason = "STRUCTURED_COMMAND_ARGUMENTS_INVALID"
                    details["validation_errors"] = errors
        if not reason:
            return None
        return AgentResult(
            status="failed",
            agent_id=self.manifest.id,
            answer="Bitrix отклонил несовместимую команду оркестратора.",
            actions_taken=[
                ActionRecord(
                    name="bitrix_capability_contract",
                    status="rejected",
                    details={"reason": reason, **details},
                )
            ],
            confidence=0.0,
            metadata={
                "terminal": True,
                "answer_is_final": True,
                "safe_to_send": True,
                "fast_return": True,
                "fast_return_reason": "structured_command_rejected",
                "reason": reason,
                "registry_version": registry.get("registry_version"),
            },
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        bitrix_client: BitrixToolClientPort | None = None,
        portal_search_index: Any | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
        scheduler: SchedulerPort | None = None,
        bitrix_store: Any | None = None,
        orchestrator_store: Any | None = None,
        task_close_report_renderer: Any | None = None,
        task_close_result_text_renderer: Any | None = None,
        draft_confirmation_phrase_renderer: Any | None = None,
        draft_confirmation_matcher: Any | None = None,
        settings: Settings | None = None,
        specialist_result_publisher: Any | None = None,
        conversation_trace: Any | None = None,
        **_: object,
    ) -> Bitrix24Specialist:
        current_settings = settings or get_settings()
        tools: list[AgentTool] = [
            PortalSearchTool(
                portal_search=portal_search_index,
                bitrix_files=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                state_store=orchestrator_store,
                live_fallback_enabled=bitrix_oauth is not None,
                index_max_age_seconds=max(
                    900,
                    current_settings.search_delta_interval_seconds * 4
                    if current_settings.search_background_periodic_delta_enabled
                    else current_settings.search_background_metadata_interval_seconds * 2,
                ),
                index_freshness_path=current_settings.search_background_state_path,
            ),
            BitrixWarehouseSearchTool(client=bitrix_client, portal_search=portal_search_index, bitrix_oauth=bitrix_oauth),
            BitrixMyTasksTool(client=bitrix_client, bitrix_oauth=bitrix_oauth),
            BitrixTaskSearchTool(client=bitrix_client, portal_search=portal_search_index, bitrix_oauth=bitrix_oauth),
            BitrixProjectSearchTool(client=bitrix_client, bitrix_oauth=bitrix_oauth),
            BitrixApiTool(
                client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
            ),
            TaskCreateDraftTool(store=bitrix_store),
            TaskCreateConfirmTool(
                store=bitrix_store,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=current_settings.agent_dry_run,
                oauth_required_for_writes=current_settings.bitrix_oauth_required_for_writes,
                draft_ttl_minutes=current_settings.bitrix_task_draft_ttl_minutes,
            ),
            TaskDraftDiscardTool(store=bitrix_store),
            TaskCloseDraftTool(
                store=bitrix_store,
                read_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                result_text_renderer=task_close_result_text_renderer,
            ),
            TaskCloseConfirmTool(
                store=bitrix_store,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=current_settings.agent_dry_run,
                oauth_required_for_writes=current_settings.bitrix_oauth_required_for_writes,
                draft_ttl_minutes=current_settings.bitrix_task_draft_ttl_minutes,
                report_renderer=task_close_report_renderer,
                result_text_renderer=task_close_result_text_renderer,
            ),
            TaskCloseDiscardTool(store=bitrix_store),
            TaskCloseReportIncidentTool(client=bitrix_client, portal_search=portal_search_index, settings=current_settings),
            TaskCloseControlGetTool(store=bitrix_store, user_client=bitrix_client),
            TaskCloseControlUpdateTool(store=bitrix_store, user_client=bitrix_client, bitrix_oauth=bitrix_oauth),
            CalendarEventDraftTool(store=bitrix_store),
            CalendarEventConfirmTool(
                store=bitrix_store,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=current_settings.agent_dry_run,
                oauth_required_for_writes=current_settings.bitrix_oauth_required_for_writes,
                draft_ttl_minutes=current_settings.bitrix_task_draft_ttl_minutes,
            ),
            CalendarEventDiscardTool(store=bitrix_store),
            ProjectCreateDraftTool(store=bitrix_store),
            ProjectCreateConfirmTool(
                store=bitrix_store,
                write_client=bitrix_client,
                bitrix_oauth=bitrix_oauth,
                dry_run=current_settings.agent_dry_run,
                oauth_required_for_writes=current_settings.bitrix_oauth_required_for_writes,
                draft_ttl_minutes=current_settings.bitrix_task_draft_ttl_minutes,
            ),
            ProjectCreateDiscardTool(store=bitrix_store),
        ]
        return cls(
            manifest,
            agent_tools=tools,
            scheduler=scheduler,
            draft_store=bitrix_store,
            store=bitrix_store,
            bitrix_task_client=bitrix_client,
            bitrix_user_client=bitrix_client,
            settings=current_settings,
            confirmation_phrase_renderer=draft_confirmation_phrase_renderer,
            confirmation_matcher=draft_confirmation_matcher,
            result_publisher=specialist_result_publisher,
            conversation_trace=conversation_trace,
        )

    async def handle(self, task: AgentTask) -> AgentResult:
        return AgentResult(
            status="failed",
            agent_id=self.manifest.id,
            answer="Bitrix принимает только точную структурированную команду оркестратора.",
            model_usage=[
                ModelUsageRecord(
                    agent_id=self.manifest.id,
                    provider="internal",
                    model="bitrix-executor-guard",
                    status="not_used",
                    notes=["All Bitrix free-text reasoning paths are disabled."],
                )
            ],
            actions_taken=[
                ActionRecord(
                    name="bitrix_executor_guard",
                    status="rejected",
                    details={"reason": "ORCHESTRATOR_STRUCTURED_COMMAND_REQUIRED"},
                )
            ],
            metadata={"reason": "ORCHESTRATOR_STRUCTURED_COMMAND_REQUIRED"},
        )

    async def _execute_tool_call(
        self,
        tool_call: Any,
        task: AgentTask,
    ) -> tuple[ToolResult | None, Any | None, list[Any]]:
        is_admin_confirm = tool_call.name == "task_close_control_update" and str(
            (tool_call.args or {}).get("operation") or ""
        ) == "confirm"
        if tool_call.name in _CONFIRM_TOOLS or is_admin_confirm:
            draft = task.context.get("pending_task_draft") if isinstance(task.context, dict) else None
            trusted_auto_finalize = (
                task.source == "task_close_direct_control"
                and task.context.get("orchestrator_internal_event") is True
                and task.context.get("orchestrator_required_tool") == "task_close_confirm"
                and task.context.get("task_close_confirmation_mode") == "auto_unconfirmed"
            )
            confirmation_matches = bool(
                self._confirmation_matcher
                and self._confirmation_matcher(
                    str(task.request or ""),
                    draft,
                    allow_short_command=bool(task.context.get("conversation_reference_explicit")),
                )
            )
            if not trusted_auto_finalize and not confirmation_matches:
                phrase = (
                    self._confirmation_phrase_renderer((draft or {}).get("_draft_type"))
                    if self._confirmation_phrase_renderer
                    else "явное подтверждение, указанное оркестратором"
                )
                return (
                    ToolResult(
                        status=ToolStatus.DENIED,
                        tool=tool_call.name,
                        error=f"Для подтверждения черновика требуется фраза: {phrase}.",
                    ),
                    None,
                    [],
                )

        args = dict(tool_call.args or {})
        if tool_call.name == "project_create_draft":
            args = _project_create_args_with_actor_context(args, task)
        elif tool_call.name in {"task_close_control_get", "task_close_control_update"}:
            args = _args_with_actor_admin_context(args, task)
        if tool_call.name in _DRAFT_TOOLS:
            args = _draft_args_with_metadata(args, task)
        if args != dict(tool_call.args or {}):
            tool_call = SimpleNamespace(name=tool_call.name, args=args, summary=getattr(tool_call, "summary", ""))

        try:
            return await super()._execute_tool_call(tool_call, task)
        except RuntimeError as exc:
            reason = str(exc).split(":", 1)[0]
            if reason not in {"ACTIVE_DRAFT_CONFLICT", "ACTIVE_DRAFT_IN_PROGRESS"}:
                raise
            answer = (
                "В этом диалоге уже активен другой черновик. Сначала завершите или отмените его."
                if reason == "ACTIVE_DRAFT_CONFLICT"
                else "Текущий черновик уже подтверждается или завершается."
            )
            result = ToolResult(
                status=ToolStatus.DENIED,
                tool=tool_call.name,
                error=answer,
                data={"status": "needs_clarification", "reason": reason, "answer": answer},
            )
            return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump()), []

    def tool_definitions(self) -> list[dict]:
        return [
            tool.definition().model_dump()
            for tool in self._tool_registry.values()
            if tool.name in _STRUCTURED_TOOL_NAMES
        ]

    async def _load_structured_context(self, task: AgentTask, tool_name: str) -> tuple[AgentTask, dict]:
        context: dict[str, Any] = {}
        if tool_name in {
            "task_create_confirm",
            "task_close_confirm",
            "calendar_event_confirm",
            "project_create_confirm",
            "task_draft_discard",
            "task_close_discard",
            "calendar_event_discard",
            "project_create_discard",
        }:
            dialog_key = str(task.context.get("dialog_key") or "")
            if dialog_key and self._draft_store is not None:
                trusted_auto_finalize = (
                    task.source == "task_close_direct_control"
                    and task.context.get("orchestrator_internal_event") is True
                    and task.context.get("task_close_confirmation_mode") == "auto_unconfirmed"
                    and tool_name == "task_close_confirm"
                )
                if trusted_auto_finalize:
                    pending = await self._draft_store.get_task_draft_for_finalizer(dialog_key)
                else:
                    pending = await self._draft_store.get_task_draft(
                        dialog_key,
                        ttl_minutes=self._settings.bitrix_task_draft_ttl_minutes if self._settings else None,
                    )
                if pending:
                    context["pending_task_draft"] = pending
        if tool_name in {"project_create_draft", "task_close_control_get", "task_close_control_update"}:
            context.update(await self._load_permission_context(task))
        merged = task.model_copy(update={"context": {**task.context, **context}})
        return merged, {
            "mode": "structured_minimal",
            "loaded_draft": "pending_task_draft" in context,
            "loaded_permission_profile": "bitrix_current_user_profile" in context,
        }

    async def _load_permission_context(self, task: AgentTask) -> dict:
        profile_result = await self._current_user_profile_result(task)
        return {"bitrix_current_user_profile": profile_result.model_dump()}

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

    def _logs(self) -> list[str]:
        return ["Bitrix is a structured executor; the orchestrator owns request meaning."]


def _project_create_args_with_actor_context(args: dict[str, Any], task: AgentTask) -> dict[str, Any]:
    profile = _current_user_profile(task)
    return {**args, "_actor_name": _current_user_label(task), "_actor_is_admin": bool(profile.get("is_admin"))}


def _args_with_actor_admin_context(args: dict[str, Any], task: AgentTask) -> dict[str, Any]:
    return {**args, "_actor_is_admin": bool(_current_user_profile(task).get("is_admin"))}


def _draft_args_with_metadata(args: dict[str, Any], task: AgentTask) -> dict[str, Any]:
    result = {
        **args,
        "_original_request": str(task.request or ""),
        "_draft_user_id": optional_int(task.user.id) if task.user.id is not None else None,
        "_draft_specialist": "bitrix24",
    }
    event = task.context.get("task_close_event") if isinstance(task.context, dict) else None
    if (
        task.source == "task_close_direct_control"
        and task.context.get("orchestrator_internal_event") is True
        and isinstance(event, dict)
    ):
        result.update(
            {
                "_direct_close_close_event_key": event.get("close_event_key"),
                "_direct_close_closed_at": event.get("closed_at"),
                "_direct_close_already_closed": True,
            }
        )
    return result


def _current_user_label(task: AgentTask) -> str:
    if str(task.user.display_name or "").strip():
        return str(task.user.display_name).strip()
    return str(_current_user_profile(task).get("label") or "").strip()


def _current_user_profile(task: AgentTask) -> dict[str, Any]:
    profile_result = task.context.get("bitrix_current_user_profile")
    if not isinstance(profile_result, dict):
        return {}
    data = profile_result.get("data")
    if not isinstance(data, dict):
        return {}
    profile = data.get("profile")
    return profile if isinstance(profile, dict) else {}


def _terminal_status(tool_name: str, result: ToolResult) -> str:
    if tool_name in {"task_create_draft", "task_close_draft", "calendar_event_draft", "project_create_draft"}:
        return "needs_human"
    if tool_name == "project_create_confirm" and isinstance(result.data.get("followup_task_draft"), dict):
        return "needs_human"
    if tool_name == "task_close_control_update" and str(result.data.get("operation") or "") == "prepare":
        return "needs_human"
    return "completed"


__all__ = ["Bitrix24Specialist"]
