from __future__ import annotations

import base64
import html
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_server.agents.bitrix24.ports import TaskDraftStorePort
from ai_server.agents.bitrix24.tools.read_client import resolve_current_user_read_client
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.integrations.bitrix.portal_search.search_index import PortalSearchIndex
from ai_server.integrations.bitrix.task_close_direct_queue import (
    complete_direct_close_event,
    discard_direct_close_event,
)
from ai_server.integrations.bitrix.task_close_reports import (
    canonical_task_close_report_file_name,
    restore_task_close_report_file,
    stable_task_close_report_file_name,
    task_close_report_key,
    task_close_report_problem_types,
    task_close_report_state_key,
)
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.settings import Settings, get_settings
from ai_server.tools.bitrix_policy import apply_write_policy
from ai_server.tools.bitrix_ports import BitrixToolClientPort, BitrixWritePort
from ai_server.utils import MOSCOW_TZ, compact_text, optional_int

TASK_CLOSE_DRAFT_TYPE = "task_close"
INCOMPLETE_CLOSE_MARKER = "AI_SERVER_TASK_CLOSE_INCOMPLETE"
_TASK_CLOSE_READ_SELECT = ["ID", "TITLE", "DESCRIPTION", "STATUS", "CLOSED_DATE"]
_BITRIX_PAIRED_TAG_RE = re.compile(
    r"\[(USER|URL|B|I|U|S|QUOTE|CODE|COLOR|SIZE)[^\]]*\](.*?)\[/\1\]", re.IGNORECASE | re.DOTALL
)
_BITRIX_SINGLE_TAG_RE = re.compile(r"\[/?[A-Z][A-Z0-9_]*(?:=[^\]]*)?\]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

_TASK_CLOSE_LIST_FIELDS = {
    "unresolved_items": ("unresolved_items", "missing_parts"),
    "task_points": ("task_points", "task_items", "checklist_items"),
    "not_done_items": ("not_done_items", "unfinished_items", "incomplete_items"),
    "unconfirmed_items": ("unconfirmed_items", "unknown_items", "unverified_items"),
    "missing_fields": ("missing_fields", "questions", "needed_details"),
}
_TASK_CLOSE_SCALAR_FIELDS = {
    "task_title": ("task_title", "title"),
    "action": ("action", "close_action"),
    "completion_summary": ("completion_summary", "result_text", "summary"),
    "equipment_consumables": ("equipment_consumables", "equipment", "consumables"),
    "overall_status": ("overall_status", "completion_status"),
    "additional_info": ("additional_info", "extra_info", "object_notes"),
}
_TASK_CLOSE_BOOL_FIELDS = {
    "already_closed": (
        "already_closed",
        "task_already_closed",
        "bitrix_task_already_closed",
        "closed_in_bitrix",
    ),
    "source_task_description_empty": (
        "source_task_description_empty",
        "task_description_empty",
        "empty_task_description",
    ),
}


@dataclass(frozen=True)
class BitrixTaskCloseDraft:
    payload: dict[str, Any] = field(default_factory=dict)
    preview: dict[str, Any] = field(default_factory=dict)
    contract_errors: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.contract_errors and bool(self.payload.get("task_id"))

    def as_action_details(self) -> dict[str, Any]:
        return {
            "payload": self.payload,
            "preview": self.preview,
            "contract_errors": self.contract_errors,
        }


def build_task_close_draft_from_args(args: dict[str, Any]) -> BitrixTaskCloseDraft:
    args = args or {}
    task_id = optional_int(args.get("task_id") or args.get("id") or args.get("ID"))
    task_title = compact_text(str(args.get("task_title") or args.get("title") or ""))
    action = _close_action(args.get("action") or args.get("close_action"))
    completion_summary = compact_text(
        str(args.get("completion_summary") or args.get("result_text") or args.get("summary") or "")
    )
    unresolved_items = _string_list(args.get("unresolved_items") or args.get("missing_parts"))
    task_points = _string_list(args.get("task_points") or args.get("task_items") or args.get("checklist_items"))
    source_task_description_empty = _truthy(
        args.get("source_task_description_empty")
        or args.get("task_description_empty")
        or args.get("empty_task_description")
    )
    already_closed = _truthy(
        args.get("already_closed")
        or args.get("task_already_closed")
        or args.get("bitrix_task_already_closed")
        or args.get("closed_in_bitrix")
    )
    equipment_consumables = compact_text(
        str(args.get("equipment_consumables") or args.get("equipment") or args.get("consumables") or "")
    )
    additional_info = compact_text(
        str(args.get("additional_info") or args.get("extra_info") or args.get("object_notes") or "")
    )
    overall_status = _overall_status(args.get("overall_status") or args.get("completion_status"))
    overall_status_label = _overall_status_label(overall_status)
    not_done_items = _string_list(
        args.get("not_done_items") or args.get("unfinished_items") or args.get("incomplete_items")
    )
    explicit_unconfirmed_items = _string_list(
        args.get("unconfirmed_items") or args.get("unknown_items") or args.get("unverified_items")
    )
    unconfirmed_items = _unique_strings(explicit_unconfirmed_items or unresolved_items)
    missing_fields = _string_list(args.get("missing_fields") or args.get("questions") or args.get("needed_details"))
    problem_types = _problem_types(
        overall_status=overall_status,
        not_done_items=not_done_items,
        unconfirmed_items=unconfirmed_items,
    )
    ai_close_incomplete = bool(problem_types)
    ai_close_marker = INCOMPLETE_CLOSE_MARKER if ai_close_incomplete else ""

    contract_errors: list[str] = []
    if task_id is None:
        contract_errors.append("task_close_draft.task_id is required")
    if not any(
        [
            completion_summary,
            task_points,
            equipment_consumables,
            additional_info,
            overall_status,
            not_done_items,
            unconfirmed_items,
            missing_fields,
        ]
    ):
        contract_errors.append(
            "task_close_draft.completion_summary, task_points, overall_status, "
            "not_done_items or unconfirmed_items is required"
        )

    result_text = _result_text(
        completion_summary=completion_summary,
        task_points=task_points,
        equipment_consumables=equipment_consumables,
        additional_info=additional_info,
        overall_status_label=overall_status_label,
        not_done_items=not_done_items,
        unconfirmed_items=unconfirmed_items,
    )
    payload = {
        "_draft_type": TASK_CLOSE_DRAFT_TYPE,
        "task_id": task_id,
        "task_title": task_title,
        "action": action,
        "completion_summary": completion_summary,
        "task_points": task_points,
        "source_task_description_empty": source_task_description_empty,
        "already_closed": already_closed,
        "equipment_consumables": equipment_consumables,
        "additional_info": additional_info,
        "overall_status": overall_status,
        "overall_status_label": overall_status_label,
        "not_done_items": not_done_items,
        "unconfirmed_items": unconfirmed_items,
        "unresolved_items": _unique_strings([*not_done_items, *unconfirmed_items]),
        "missing_fields": missing_fields,
        "problem_types": problem_types,
        "ai_close_incomplete": ai_close_incomplete,
        "ai_close_marker": ai_close_marker,
        "result_text": result_text,
    }
    preview = {
        "task_title": task_title,
        "action": action,
        "action_label": _action_label(action),
        "completion_summary": completion_summary,
        "task_points": task_points,
        "source_task_description_empty": source_task_description_empty,
        "already_closed": already_closed,
        "equipment_consumables": equipment_consumables,
        "additional_info": additional_info,
        "overall_status": overall_status,
        "overall_status_label": overall_status_label,
        "not_done_items": not_done_items,
        "unconfirmed_items": unconfirmed_items,
        "unresolved_items": payload["unresolved_items"],
        "missing_fields": missing_fields,
        "problem_types": problem_types,
        "ai_close_incomplete": ai_close_incomplete,
        "ai_close_marker": ai_close_marker,
        "result_text": result_text,
    }
    return BitrixTaskCloseDraft(payload=payload, preview=preview, contract_errors=contract_errors)


class TaskCloseDraftTool:
    name = "task_close_draft"

    def __init__(
        self,
        store: TaskDraftStorePort,
        read_client: BitrixToolClientPort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
    ) -> None:
        self._store = store
        self._read_client = read_client
        self._bitrix_oauth = bitrix_oauth

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Prepare a Bitrix task closing draft. Use instead of direct bitrix_api writes for task closing. "
                "Call as soon as the task is identified; if completion details are missing, put them into "
                "missing_fields/unconfirmed_items so the user sees a full draft. The user must confirm before "
                "a protected AI close report file is saved and, when the task is still open, "
                "tasks.task.complete/tasks.task.approve is executed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "task_title": {"type": "string"},
                    "completion_summary": {
                        "type": "string",
                        "description": "What was done / result text that should be added to the task.",
                    },
                    "unresolved_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Items that could not be verified but the user explicitly agreed to close with a marker."
                        ),
                    },
                    "task_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Short task/checklist points and what is known about their completion.",
                    },
                    "source_task_description_empty": {
                        "type": "boolean",
                        "description": (
                            "True when the original Bitrix task has no meaningful description/checklist. "
                            "Then block 1 is a free-form work description instead of task points."
                        ),
                    },
                    "already_closed": {
                        "type": "boolean",
                        "description": (
                            "True when the Bitrix task is already closed. Confirm then saves the AI close report "
                            "without calling tasks.task.complete/tasks.task.approve again."
                        ),
                    },
                    "equipment_consumables": {
                        "type": "string",
                        "description": "Equipment or consumables used, in the user's own words.",
                    },
                    "overall_status": {
                        "type": "string",
                        "enum": ["completed", "partial", "not_done", "unconfirmed"],
                        "description": "Overall completion status for the draft.",
                    },
                    "not_done_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task points that are known to be unfinished or not done.",
                    },
                    "unconfirmed_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task points whose result is unknown or not confirmed.",
                    },
                    "missing_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Draft fields that still need a short answer from the user.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["complete", "approve"],
                        "description": "complete for performer completion, approve for creator approval.",
                    },
                },
                "required": ["task_id"],
            },
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if context_error := _write_context_error(user_id=user_id, dialog_id=dialog_id):
            return ToolResult(status=ToolStatus.DENIED, tool=self.name, error=context_error)
        if not dialog_key:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="task_close_draft requires dialog_key",
            )
        enriched_args, access_error = await self._enrich_args_from_task(args, user_id=user_id)
        if access_error is not None:
            return access_error
        draft = build_task_close_draft_from_args(enriched_args)
        if not draft.is_ready:
            return ToolResult(
                status=ToolStatus.CONTRACT_VIOLATION,
                tool=self.name,
                error=f"task_close_draft called outside contract: {'; '.join(draft.contract_errors)}",
                data=draft.as_action_details(),
            )
        existing = await self._store.get_task_draft(dialog_key)
        if isinstance(existing, dict) and existing.get("_draft_type") == TASK_CLOSE_DRAFT_TYPE:
            current_task_id = optional_int(existing.get("task_id"))
            requested_task_id = optional_int(draft.payload.get("task_id"))
            if current_task_id is not None and requested_task_id is not None and current_task_id != requested_task_id:
                conflict_draft = {
                    **existing,
                    "_task_close_conflict_pending": draft.payload,
                    "_task_close_conflict_pending_preview": draft.preview,
                }
                await self._store.save_task_draft(dialog_key, conflict_draft)
                return ToolResult(
                    status=ToolStatus.OK,
                    tool=self.name,
                    data={
                        "status": "active_draft_conflict",
                        "draft": conflict_draft,
                        "preview": _preview_from_payload(conflict_draft),
                        "incoming_draft": draft.payload,
                        "incoming_preview": draft.preview,
                        "conflict": {
                            "current_task_id": current_task_id,
                            "requested_task_id": requested_task_id,
                            "actions": [
                                "continue_current_draft",
                                "close_current_as_is_with_marker",
                                "discard_current_and_start_requested",
                            ],
                        },
                    },
                )
            merged = build_task_close_draft_from_args(_merge_task_close_draft_args(existing, args, draft.payload))
            if not merged.is_ready:
                return ToolResult(
                    status=ToolStatus.CONTRACT_VIOLATION,
                    tool=self.name,
                    error=f"merged task_close_draft is invalid: {'; '.join(merged.contract_errors)}",
                    data=merged.as_action_details(),
                )
            await self._store.save_task_draft(dialog_key, merged.payload)
            return ToolResult(
                status=ToolStatus.OK,
                tool=self.name,
                data={"status": "updated_draft", "draft": merged.payload, "preview": merged.preview},
            )
        await self._store.save_task_draft(dialog_key, draft.payload)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"status": "new_draft", "draft": draft.payload, "preview": draft.preview},
        )

    async def _enrich_args_from_task(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None,
    ) -> tuple[dict[str, Any], ToolResult | None]:
        task_id = optional_int((args or {}).get("task_id") or (args or {}).get("id") or (args or {}).get("ID"))
        if task_id is None or self._read_client is None:
            return dict(args or {}), None

        read_client, _, access_error = await resolve_current_user_read_client(
            self.name,
            fallback_client=self._read_client,
            bitrix_oauth=self._bitrix_oauth,
            user_id=user_id,
        )
        if access_error is not None:
            return {}, access_error

        try:
            result = await read_client.result("tasks.task.get", {"taskId": task_id, "select": _TASK_CLOSE_READ_SELECT})
        except (BitrixApiError, BitrixConfigError) as exc:
            return (
                {},
                ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error=f"Bitrix task lookup failed for task_close_draft: {exc}",
                    data={"task_id": task_id},
                ),
            )

        task = _extract_bitrix_task(result)
        if not task:
            return dict(args or {}), None

        enriched = dict(args or {})
        title = _first_text(task, "title", "TITLE")
        if title and not compact_text(str(enriched.get("task_title") or enriched.get("title") or "")):
            enriched["task_title"] = title

        description = _first_text(task, "description", "DESCRIPTION")
        description_points = _task_description_points(description)
        if not _string_list(
            enriched.get("task_points") or enriched.get("task_items") or enriched.get("checklist_items")
        ):
            enriched["task_points"] = description_points
        if "source_task_description_empty" not in enriched and "task_description_empty" not in enriched:
            enriched["source_task_description_empty"] = not bool(description_points)

        status = compact_text(_first_text(task, "status", "STATUS"))
        if "already_closed" not in enriched and "task_already_closed" not in enriched:
            enriched["already_closed"] = status == "5" or bool(_first_text(task, "closedDate", "CLOSED_DATE"))

        if not _task_close_has_user_result(enriched):
            enriched.setdefault("overall_status", "unconfirmed")
            enriched.setdefault("unconfirmed_items", description_points or ["результат выполнения не указан"])
            enriched.setdefault(
                "missing_fields",
                [
                    "что сделано по задаче",
                    "оборудование/расходники, если использовались",
                    "итог: выполнена полностью, частично или не выполнена",
                ],
            )
        return enriched, None


class TaskCloseConfirmTool:
    name = "task_close_confirm"

    def __init__(
        self,
        store: TaskDraftStorePort,
        write_client: BitrixWritePort | None = None,
        *,
        bitrix_oauth: BitrixOAuthService | None = None,
        dry_run: bool = False,
        oauth_required_for_writes: bool = True,
        draft_ttl_minutes: int | None = None,
    ) -> None:
        self._store = store
        self._write_client = write_client
        self._bitrix_oauth = bitrix_oauth
        self._dry_run = dry_run
        self._oauth_required_for_writes = oauth_required_for_writes
        self._draft_ttl_minutes = draft_ttl_minutes

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=("Confirm and execute the pending task closing draft as the current Bitrix OAuth user."),
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if not dialog_key:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="task_close_confirm requires dialog_key",
            )
        draft = await self._store.get_task_draft(dialog_key, ttl_minutes=self._draft_ttl_minutes)
        if not draft or draft.get("_draft_type") != TASK_CLOSE_DRAFT_TYPE:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool=self.name,
                error="no pending task closing draft found for this dialog",
            )
        if self._dry_run:
            return ToolResult(status=ToolStatus.DRY_RUN, tool=self.name, data={"draft": draft})

        if self._oauth_required_for_writes:
            if context_error := _write_context_error(user_id=user_id, dialog_id=dialog_id):
                return ToolResult(status=ToolStatus.DENIED, tool=self.name, error=context_error, data={"draft": draft})
            if self._bitrix_oauth is None:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error="Bitrix OAuth is required for task closing.",
                    data={"draft": draft},
                )
            if compact_text(str(draft.get("result_text") or "")) and self._write_client is None:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error="System Bitrix write_client is required to attach protected task close report.",
                    data={"draft": draft},
                )
            try:
                oauth_client = await self._bitrix_oauth.client_for_user(user_id)
                result = await _execute_task_close(
                    close_call=oauth_client.call,
                    report_call=self._write_client.call if self._write_client is not None else None,
                    draft=draft,
                )
            except BitrixOAuthTokenMissing as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED, tool=self.name, error=str(exc), data={"draft": draft}
                )
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                    tool=self.name,
                    error=str(exc),
                    data={"draft": draft},
                )
            except Exception as exc:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error=f"{type(exc).__name__}: {exc}",
                    data={"draft": draft},
                )
            _complete_direct_close_queue_from_draft(self._store, draft, actor_user_id=user_id)
            await self._store.delete_task_draft(dialog_key)
            return ToolResult(status=ToolStatus.OK, tool=self.name, data={**result, "draft": draft})

        if self._write_client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="write_client is not configured",
                data={"draft": draft},
            )
        try:
            result = await _execute_task_close(
                close_call=self._write_client.call,
                report_call=self._write_client.call,
                draft=draft,
            )
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR, tool=self.name, error=f"{type(exc).__name__}: {exc}", data={"draft": draft}
            )
        _complete_direct_close_queue_from_draft(self._store, draft, actor_user_id=user_id)
        await self._store.delete_task_draft(dialog_key)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={**result, "draft": draft})


class TaskCloseDiscardTool:
    name = "task_close_discard"

    def __init__(self, store: TaskDraftStorePort) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Discard the pending task closing draft.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        draft = None
        if dialog_key:
            draft = await self._store.get_task_draft(dialog_key)
        if dialog_key:
            await self._store.delete_task_draft(dialog_key)
        if isinstance(draft, dict):
            _discard_direct_close_queue_from_draft(self._store, draft, actor_user_id=user_id)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={"discarded": bool(dialog_key)})


class TaskCloseReportIncidentTool:
    name = "task_close_report_incident"

    def __init__(
        self,
        *,
        client: Any | None = None,
        portal_search: PortalSearchIndex | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._portal_search = portal_search
        self._settings = settings or get_settings()

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Resolve an AI task close report integrity incident after an admin text reply. "
                "Use action=restore for option 1 and action=accept_missing for option 2."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "action": {"type": "string", "enum": ["restore", "accept_missing"]},
                    "file_name": {"type": "string"},
                },
                "required": ["task_id", "action"],
            },
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if context_error := _write_context_error(user_id=user_id, dialog_id=dialog_id):
            return ToolResult(status=ToolStatus.DENIED, tool=self.name, error=context_error)
        if user_id not in set(self._settings.resolved_task_close_report_admin_user_ids):
            return ToolResult(
                status=ToolStatus.DENIED,
                tool=self.name,
                error="Only configured Bitrix task close report admins can resolve this incident.",
            )
        if self._portal_search is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="PortalSearchIndex is required to resolve task close report incidents.",
            )

        task_id = optional_int(args.get("task_id"))
        action = _report_incident_action(args.get("action"))
        if task_id is None or not action:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="task_close_report_incident requires task_id and action=restore|accept_missing.",
            )

        task = self._portal_search.get_item(entity_type="task", entity_id=task_id)
        if task is None:
            return ToolResult(status=ToolStatus.NOT_FOUND, tool=self.name, error=f"Task #{task_id} is not indexed.")
        metadata = dict(task.metadata or {})
        missing_files = _dict_list(metadata.get("ai_close_report_missing_files"))
        if not missing_files:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool=self.name,
                error=f"No pending AI close report incident found for task #{task_id}.",
            )
        file_record = _select_report_file(missing_files, str(args.get("file_name") or ""))
        if file_record is None:
            return ToolResult(
                status=ToolStatus.NOT_FOUND,
                tool=self.name,
                error="Requested AI close report file is not pending for this task.",
            )

        if action == "restore":
            if self._client is None:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error="Bitrix client is required to restore the AI close report file.",
                )
            try:
                restore_result = await restore_task_close_report_file(
                    self._client,
                    task_id=task_id,
                    file_record=file_record,
                    max_bytes=self._settings.attachment_max_bytes,
                )
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(status=ToolStatus.ERROR, tool=self.name, error=str(exc))
            except Exception as exc:
                return ToolResult(status=ToolStatus.ERROR, tool=self.name, error=f"{type(exc).__name__}: {exc}")
            updated_metadata = _metadata_after_report_restore(metadata, restore_result.get("restored_file"))
            updated_body = _body_with_ai_close_marker(task.body, updated_metadata)
            self._portal_search.update_item_body_metadata(
                entity_type="task",
                entity_id=task_id,
                body=updated_body,
                metadata=updated_metadata,
            )
            _upsert_report_incident_state(
                self._portal_search,
                task_id=task_id,
                file_record=file_record,
                status="restored",
                payload={"restore_result": restore_result},
                actor_user_id=user_id,
            )
            return ToolResult(
                status=ToolStatus.OK,
                tool=self.name,
                data={
                    "action": "restore",
                    "task_id": task_id,
                    "file_name": file_record.get("name"),
                    "restore_result": restore_result,
                },
            )

        updated_metadata = _metadata_after_report_accept_missing(metadata)
        updated_body = _body_without_ai_close_marker(task.body)
        for record in missing_files:
            attached_object_id = optional_int(record.get("attached_object_id"))
            if attached_object_id is not None:
                self._portal_search.delete_item(entity_type="task_attachment", entity_id=attached_object_id)
            _upsert_report_incident_state(
                self._portal_search,
                task_id=task_id,
                file_record=record,
                status="accepted_missing",
                payload={"accepted_file": record},
                actor_user_id=user_id,
            )
        self._portal_search.update_item_body_metadata(
            entity_type="task",
            entity_id=task_id,
            body=updated_body,
            metadata=updated_metadata,
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"action": "accept_missing", "task_id": task_id, "file_name": file_record.get("name")},
        )


async def _execute_task_close(
    *,
    close_call: Callable[[str, dict[str, Any]], Awaitable[Any]],
    report_call: Callable[[str, dict[str, Any]], Awaitable[Any]] | None,
    draft: dict[str, Any],
) -> dict[str, Any]:
    task_id = optional_int(draft.get("task_id"))
    if task_id is None:
        raise ValueError("task closing draft has no task_id")
    result_text = compact_text(str(draft.get("result_text") or ""))
    action = _close_action(draft.get("action"))

    close_method = "tasks.task.approve" if action == "approve" else "tasks.task.complete"
    if _truthy(draft.get("_direct_close_already_closed")):
        close_method = "already_closed"
        close_result = {"skipped": True, "reason": "task_already_closed_directly"}
    elif _truthy(draft.get("already_closed") or draft.get("task_already_closed") or draft.get("_task_already_closed")):
        close_method = "already_closed"
        close_result = {"skipped": True, "reason": "task_already_closed"}
    else:
        close_params = apply_write_policy(close_method, {"taskId": task_id})
        close_result = await close_call(close_method, close_params)

    report_add = None
    report_method = ""
    report_file_name = ""
    report_file_content = ""
    if result_text:
        if report_call is None:
            raise BitrixConfigError("system Bitrix write_client is required to attach protected task close report.")
        report_file_name = _report_file_name(task_id)
        report_file_content = _report_file_content(task_id=task_id, action=action, draft=draft)
        existing_report = await _find_existing_report_file(report_call, task_id=task_id, file_name=report_file_name)
        if existing_report is not None:
            disk_file_id = _report_disk_file_id(existing_report)
            if disk_file_id is None:
                raise BitrixConfigError("Cannot update existing AI close report: disk file id is missing.")
            report_params = apply_write_policy(
                "disk.file.uploadversion",
                {
                    "id": disk_file_id,
                    "fileContent": [
                        report_file_name,
                        base64.b64encode(report_file_content.encode("utf-8")).decode("ascii"),
                    ],
                },
            )
            report_add = await report_call("disk.file.uploadversion", report_params)
            report_method = "disk.file.uploadversion"
        else:
            report_params = apply_write_policy(
                "task.item.addfile",
                {
                    "taskId": task_id,
                    "fileParameters": {
                        "NAME": report_file_name,
                        "CONTENT": base64.b64encode(report_file_content.encode("utf-8")).decode("ascii"),
                    },
                },
            )
            report_add = await report_call("task.item.addfile", report_params)
            report_method = "task.item.addfile"
    return {
        "task_id": task_id,
        "task_title": str(draft.get("task_title") or ""),
        "action": action,
        "result_method": report_method,
        "result_add": report_add,
        "report_method": report_method,
        "report_file_name": report_file_name,
        "report_file_content": report_file_content,
        "report_add": report_add,
        "report_file_owner": "system_bitrix_client" if report_file_name else "",
        "report_file_edit_protection": "content_edit_denied_for_regular_users_when_created_by_system_client"
        if report_file_name
        else "",
        "close_method": close_method,
        "close_result": close_result,
        "not_done_items": _string_list(draft.get("not_done_items")),
        "unconfirmed_items": _string_list(draft.get("unconfirmed_items")),
        "unresolved_items": _string_list(draft.get("unresolved_items")),
        "problem_types": _string_list(draft.get("problem_types")),
        "ai_close_incomplete": bool(draft.get("ai_close_incomplete")),
        "ai_close_marker": str(draft.get("ai_close_marker") or ""),
    }


def _complete_direct_close_queue_from_draft(
    store: Any,
    draft: dict[str, Any],
    *,
    actor_user_id: int | None = None,
) -> None:
    task_id = optional_int(draft.get("task_id"))
    close_event_key = str(draft.get("_direct_close_close_event_key") or "").strip()
    if task_id is None or not close_event_key:
        return
    complete_direct_close_event(
        store,
        task_id=task_id,
        close_event_key=close_event_key,
        now_iso=datetime.now(MOSCOW_TZ).isoformat(timespec="seconds"),
        actor_user_id=actor_user_id,
    )


def _discard_direct_close_queue_from_draft(
    store: Any,
    draft: dict[str, Any],
    *,
    actor_user_id: int | None = None,
) -> None:
    task_id = optional_int(draft.get("task_id"))
    close_event_key = str(draft.get("_direct_close_close_event_key") or "").strip()
    if task_id is None or not close_event_key:
        return
    discard_direct_close_event(
        store,
        task_id=task_id,
        close_event_key=close_event_key,
        now_iso=datetime.now(MOSCOW_TZ).isoformat(timespec="seconds"),
        actor_user_id=actor_user_id,
    )


def _extract_bitrix_task(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        for key in ("task", "TASK"):
            value = result.get(key)
            if isinstance(value, dict):
                return value
        result_value = result.get("result")
        if isinstance(result_value, dict):
            return _extract_bitrix_task(result_value)
    return {}


def _first_text(values: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        text = compact_text(str(value))
        if text:
            return text
    return ""


def _task_description_points(description: object) -> list[str]:
    text = _clean_task_description(description)
    if not text:
        return []
    raw_parts = re.split(r"(?:\r?\n)+|(?:^|\s)(?:\d+[.)]|[-*•])\s+", text)
    points = _unique_strings([compact_text(part).strip(" .;:-") for part in raw_parts if compact_text(part)])
    if len(points) > 1:
        return points[:12]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    points = _unique_strings([compact_text(part).strip(" .;:-") for part in sentences if compact_text(part)])
    if len(points) > 1:
        return points[:12]
    comma_parts = re.split(r"[,;]+", text)
    comma_points = _unique_strings(
        [compact_text(part).strip(" .;:-") for part in comma_parts if len(compact_text(part).strip(" .;:-")) >= 6]
    )
    return comma_points[:12] if len(comma_points) > 1 else (points[:12] if points else [])


def _clean_task_description(description: object) -> str:
    text = html.unescape(str(description or ""))
    previous = None
    while previous != text:
        previous = text
        text = _BITRIX_PAIRED_TAG_RE.sub(lambda match: str(match.group(2) or ""), text)
    text = _BITRIX_SINGLE_TAG_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    return compact_text(text)


def _task_close_has_user_result(args: dict[str, Any]) -> bool:
    if compact_text(str(args.get("completion_summary") or args.get("result_text") or args.get("summary") or "")):
        return True
    if compact_text(str(args.get("equipment_consumables") or args.get("equipment") or args.get("consumables") or "")):
        return True
    if compact_text(str(args.get("additional_info") or args.get("extra_info") or args.get("object_notes") or "")):
        return True
    if _overall_status(args.get("overall_status") or args.get("completion_status")) not in {"", "unconfirmed"}:
        return True
    return bool(
        _string_list(args.get("not_done_items") or args.get("unfinished_items") or args.get("incomplete_items"))
    )


def _result_text(
    *,
    completion_summary: str,
    task_points: list[str],
    equipment_consumables: str,
    additional_info: str,
    overall_status_label: str,
    not_done_items: list[str],
    unconfirmed_items: list[str],
) -> str:
    lines: list[str] = []
    if completion_summary:
        lines.append(completion_summary)
    if task_points:
        lines.append("")
        lines.append("Пункты задачи:")
        lines.extend(f"- {item}" for item in task_points)
    if equipment_consumables:
        lines.append("")
        lines.append(f"Оборудование, расходники: {equipment_consumables}")
    if additional_info:
        lines.append("")
        lines.append(f"Дополнительная информация: {additional_info}")
    if overall_status_label:
        lines.append("")
        lines.append(f"Общий итог: {overall_status_label}")
    if not_done_items or unconfirmed_items:
        lines.append("")
        lines.append(f"Статус AI-закрытия: {_ai_close_human_status(not_done_items, unconfirmed_items)}")
    if not_done_items:
        lines.append("Невыполненные пункты:")
        lines.extend(f"- {item}" for item in not_done_items)
    if unconfirmed_items:
        lines.append("Неподтверждённые пункты:")
        lines.extend(f"- {item}" for item in unconfirmed_items)
    return "\n".join(lines).strip()


def _merge_task_close_draft_args(
    existing: dict[str, Any],
    raw_args: dict[str, Any],
    update_payload: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing)
    merged["task_id"] = update_payload.get("task_id") or existing.get("task_id")

    for payload_field, aliases in _TASK_CLOSE_SCALAR_FIELDS.items():
        if not _has_any_key(raw_args, aliases):
            continue
        value = compact_text(str(update_payload.get(payload_field) or ""))
        if value:
            merged[payload_field] = value

    for payload_field, aliases in _TASK_CLOSE_LIST_FIELDS.items():
        if not _has_any_key(raw_args, aliases):
            continue
        merged[payload_field] = _string_list(update_payload.get(payload_field))

    for payload_field, aliases in _TASK_CLOSE_BOOL_FIELDS.items():
        if not _has_any_key(raw_args, aliases):
            continue
        merged[payload_field] = bool(update_payload.get(payload_field))

    if _task_close_has_user_result(update_payload):
        for field_name in ("unconfirmed_items", "unresolved_items", "missing_fields"):
            if field_name in merged and field_name not in raw_args:
                merged[field_name] = _remove_initial_unknown_close_placeholders(merged.get(field_name))
        if "unconfirmed_items" in raw_args:
            merged["unconfirmed_items"] = _remove_initial_unknown_close_placeholders(merged.get("unconfirmed_items"))
        if "unresolved_items" not in raw_args and "missing_parts" not in raw_args:
            merged["unresolved_items"] = _unique_strings(
                [
                    *_string_list(merged.get("not_done_items")),
                    *_string_list(merged.get("unconfirmed_items")),
                ]
            )

    return merged


def _preview_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_title": str(payload.get("task_title") or ""),
        "action": _close_action(payload.get("action")),
        "action_label": _action_label(_close_action(payload.get("action"))),
        "completion_summary": str(payload.get("completion_summary") or ""),
        "task_points": _string_list(payload.get("task_points")),
        "source_task_description_empty": bool(payload.get("source_task_description_empty")),
        "already_closed": bool(payload.get("already_closed")),
        "equipment_consumables": str(payload.get("equipment_consumables") or ""),
        "additional_info": str(payload.get("additional_info") or ""),
        "overall_status": _overall_status(payload.get("overall_status")),
        "overall_status_label": str(payload.get("overall_status_label") or ""),
        "not_done_items": _string_list(payload.get("not_done_items")),
        "unconfirmed_items": _string_list(payload.get("unconfirmed_items")),
        "unresolved_items": _string_list(payload.get("unresolved_items")),
        "missing_fields": _string_list(payload.get("missing_fields")),
        "problem_types": _string_list(payload.get("problem_types")),
        "ai_close_incomplete": bool(payload.get("ai_close_incomplete")),
        "ai_close_marker": str(payload.get("ai_close_marker") or ""),
        "result_text": str(payload.get("result_text") or ""),
    }


def _has_any_key(values: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in values for key in keys)


def _remove_initial_unknown_close_placeholders(value: object) -> list[str]:
    return [
        item
        for item in _string_list(value)
        if item.casefold() not in {"результат выполнения не указан", "result is not specified"}
    ]


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on", "да"}


async def _find_existing_report_file(
    call: Callable[[str, dict[str, Any]], Awaitable[Any]],
    *,
    task_id: int,
    file_name: str,
) -> dict[str, Any] | None:
    try:
        result = await call("task.item.getfiles", {"taskId": task_id})
    except Exception:
        return None
    logical_name = canonical_task_close_report_file_name(file_name)
    fallback: dict[str, Any] | None = None
    for item in _extract_file_items(result):
        item_name = str(item.get("NAME") or item.get("name") or "")
        if item_name == file_name:
            return item
        if logical_name and canonical_task_close_report_file_name(item_name).casefold() == logical_name.casefold():
            fallback = item
    if fallback is not None:
        return fallback
    return None


def _report_disk_file_id(report_file: dict[str, Any]) -> int | None:
    return optional_int(
        report_file.get("FILE_ID")
        or report_file.get("fileId")
        or report_file.get("OBJECT_ID")
        or report_file.get("objectId")
        or report_file.get("disk_object_id")
    )


def _extract_file_items(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if not isinstance(result, dict):
        return []
    value = result.get("result")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return _extract_file_items(value)
    for key in ("files", "FILES", "items"):
        nested = result.get(key)
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
    return []


def _report_file_name(task_id: int) -> str:
    return stable_task_close_report_file_name(task_id)


def _report_file_status(draft: dict[str, Any]) -> str:
    not_done_items = _string_list(draft.get("not_done_items"))
    unconfirmed_items = _string_list(draft.get("unconfirmed_items") or draft.get("unresolved_items"))
    overall_status = _overall_status(draft.get("overall_status"))
    if unconfirmed_items or overall_status == "unconfirmed":
        return "unconfirmed"
    if overall_status == "not_done":
        return "failed"
    if not_done_items or overall_status == "partial":
        return "partial"
    return "ok"


def _report_file_content(*, task_id: int, action: str, draft: dict[str, Any]) -> str:
    task_title = compact_text(str(draft.get("task_title") or ""))
    completion_summary = compact_text(str(draft.get("completion_summary") or draft.get("result_text") or ""))
    task_points = _string_list(draft.get("task_points"))
    equipment_consumables = compact_text(str(draft.get("equipment_consumables") or ""))
    additional_info = compact_text(str(draft.get("additional_info") or ""))
    overall_status_label = compact_text(str(draft.get("overall_status_label") or ""))
    not_done_items = _string_list(draft.get("not_done_items"))
    unconfirmed_items = _string_list(draft.get("unconfirmed_items") or draft.get("unresolved_items"))
    missing_fields = _string_list(draft.get("missing_fields"))
    status = _report_file_status(draft)
    lines = [
        "AI task close report",
        f"Task ID: {task_id}",
        f"Task: {task_title or f'#{task_id}'}",
        f"Action: {action}",
        f"Status: {status}",
        f"Created at: {datetime.now(MOSCOW_TZ).isoformat(timespec='seconds')}",
        "Source: AI-server task_close_confirm",
    ]
    problem_types = _string_list(draft.get("problem_types"))
    if not problem_types:
        if not_done_items:
            problem_types.append("not_done")
        if unconfirmed_items:
            problem_types.append("unconfirmed")
    if problem_types:
        lines.append(f"Problem types: {', '.join(_unique_strings(problem_types))}")
        lines.append(f"AI marker: {INCOMPLETE_CLOSE_MARKER}")
    if completion_summary:
        lines.extend(["", "Summary:", completion_summary])
    if task_points:
        lines.append("")
        lines.append("Task points:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(task_points, start=1))
    if equipment_consumables:
        lines.extend(["", "Equipment and consumables:", equipment_consumables])
    if additional_info:
        lines.extend(["", "Additional information:", additional_info])
    if overall_status_label:
        lines.extend(["", "Overall:", overall_status_label])
    if not_done_items:
        lines.append("")
        lines.append("Not done:")
        lines.extend(f"- {item}" for item in not_done_items)
    if unconfirmed_items:
        lines.append("")
        lines.append("Unconfirmed:")
        lines.extend(f"- {item}" for item in unconfirmed_items)
    if missing_fields:
        lines.append("")
        lines.append("Missing details:")
        lines.extend(f"- {item}" for item in missing_fields)
    return "\n".join(lines).strip() + "\n"


def _ai_close_human_status(not_done_items: list[str], unconfirmed_items: list[str]) -> str:
    if not_done_items and unconfirmed_items:
        return "выполнена частично, неподтвержденная"
    if not_done_items:
        return "выполнена частично"
    return "неподтвержденная"


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [value]
    elif isinstance(value, list | tuple | set):
        parts = [str(item) for item in value]
    else:
        parts = [str(value)]
    return [compact_text(part).strip(" -•") for part in parts if compact_text(part).strip(" -•")]


def _close_action(value: object) -> str:
    text = str(value or "complete").strip().casefold()
    return "approve" if text in {"approve", "accept", "принять", "утвердить"} else "complete"


def _overall_status(value: object) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    if text in {"completed", "complete", "done", "выполнено", "готово", "полностью", "выполнена полностью"}:
        return "completed"
    if text in {"partial", "partially_done", "partly_done", "частично", "выполнена частично"}:
        return "partial"
    if text in {"not_done", "not_completed", "not done", "не выполнено", "не сделано", "не выполнена"}:
        return "not_done"
    if text in {"unconfirmed", "unknown", "unclear", "не подтверждено", "неизвестно", "непонятно"}:
        return "unconfirmed"
    return text


def _overall_status_label(status: str) -> str:
    return {
        "completed": "выполнена полностью",
        "partial": "выполнена частично",
        "not_done": "не выполнена",
        "unconfirmed": "результат не подтверждён",
    }.get(status, status)


def _problem_types(*, overall_status: str, not_done_items: list[str], unconfirmed_items: list[str]) -> list[str]:
    values: list[str] = []
    if not_done_items or overall_status in {"partial", "not_done"}:
        values.append("not_done")
    if unconfirmed_items or overall_status == "unconfirmed":
        values.append("unconfirmed")
    return values


def _unique_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = compact_text(str(value or "")).strip(" -•")
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _action_label(action: str) -> str:
    return "принять результат и закрыть задачу" if action == "approve" else "отметить задачу выполненной"


def _write_context_error(*, user_id: int | None, dialog_id: str | None) -> str:
    if user_id is None:
        return "Bitrix task closing denied: current Bitrix user_id is missing."
    if not str(dialog_id or "").strip():
        return "Bitrix task closing denied: current Bitrix dialog_id is missing."
    return ""


def _report_incident_action(value: object) -> str:
    text = str(value or "").strip().casefold()
    if text in {"restore", "1", "one", "first", "первый", "первый вариант", "восстановить", "вернуть"}:
        return "restore"
    if text in {
        "accept_missing",
        "accept",
        "2",
        "two",
        "second",
        "второй",
        "второй вариант",
        "удалить",
        "всё в порядке",
        "все в порядке",
    }:
        return "accept_missing"
    return ""


def _select_report_file(files: list[dict[str, Any]], file_name: str) -> dict[str, Any] | None:
    if not files:
        return None
    normalized = file_name.strip().casefold()
    if not normalized:
        return files[0]
    for record in files:
        if str(record.get("name") or "").strip().casefold() == normalized:
            return record
    return None


def _metadata_after_report_restore(metadata: dict[str, Any], restored_file: object) -> dict[str, Any]:
    result = dict(metadata)
    restored = dict(restored_file) if isinstance(restored_file, dict) else {}
    files = _dict_list(result.get("ai_close_report_files"))
    if restored:
        files = _replace_report_file(files, restored)
    result["ai_close_report_files"] = files
    result["ai_close_report_missing"] = False
    result["ai_close_report_missing_files"] = []
    result["ai_close_report_incident_status"] = "restored"
    result["ai_close_report_restored_at"] = datetime.now(MOSCOW_TZ).isoformat(timespec="seconds")
    _drop_report_incident_error_keys(result)
    _apply_ai_close_metadata_from_report_files(result, files)
    return result


def _metadata_after_report_accept_missing(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    missing_files = _dict_list(result.get("ai_close_report_missing_files"))
    missing_keys = {task_close_report_key(record) for record in missing_files}
    files = [
        record
        for record in _dict_list(result.get("ai_close_report_files"))
        if task_close_report_key(record) not in missing_keys
    ]
    result["ai_close_report_files"] = files
    for key in (
        "ai_close_report_missing",
        "ai_close_report_missing_files",
        "ai_close_report_incident_status",
        "ai_close_report_incident_key",
        "ai_close_report_detected_at",
        "ai_close_report_auto_restore_after",
        "ai_close_report_alert_sent_at",
        "ai_close_report_restored_at",
    ):
        result.pop(key, None)
    _drop_report_incident_error_keys(result)
    _apply_ai_close_metadata_from_report_files(result, files)
    return result


def _replace_report_file(files: list[dict[str, Any]], restored: dict[str, Any]) -> list[dict[str, Any]]:
    restored_key = task_close_report_key(restored)
    replaced = False
    result: list[dict[str, Any]] = []
    for record in files:
        if task_close_report_key(record) == restored_key or _same_logical_report_file(record, restored):
            result.append(restored)
            replaced = True
        else:
            result.append(record)
    if not replaced:
        result.append(restored)
    return result


def _same_logical_report_file(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_name = str(left.get("canonical_name") or canonical_task_close_report_file_name(left.get("name"))).casefold()
    right_name = str(right.get("canonical_name") or canonical_task_close_report_file_name(right.get("name"))).casefold()
    return bool(left_name and left_name == right_name)


def _upsert_report_incident_state(
    portal_search: PortalSearchIndex,
    *,
    task_id: int,
    file_record: dict[str, Any],
    status: str,
    payload: dict[str, Any] | None = None,
    actor_user_id: int | None = None,
) -> None:
    upsert = getattr(portal_search, "upsert_task_close_processing_state", None)
    if not callable(upsert):
        return
    upsert(
        task_id=task_id,
        state_key=task_close_report_state_key(file_record),
        status=status,
        payload=payload,
        actor_user_id=actor_user_id,
    )


def _drop_report_incident_error_keys(metadata: dict[str, Any]) -> None:
    for key in ("ai_close_report_alert_error", "ai_close_report_restore_error"):
        metadata.pop(key, None)


def _apply_ai_close_metadata_from_report_files(metadata: dict[str, Any], files: list[dict[str, Any]]) -> None:
    problem_types = _unique_strings(
        [
            problem_type
            for record in files
            for problem_type in [
                *task_close_report_problem_types([str(record.get("name") or "")]),
                *_string_list(record.get("problem_types")),
            ]
        ]
    )
    problem_types = [item for item in problem_types if item in {"not_done", "unconfirmed"}]
    if problem_types:
        metadata["ai_close_incomplete"] = True
        metadata["ai_close_marker"] = INCOMPLETE_CLOSE_MARKER
        metadata["ai_close_problem_types"] = problem_types
        metadata["ai_close_has_not_done"] = "not_done" in problem_types
        metadata["ai_close_has_unconfirmed"] = "unconfirmed" in problem_types
        metadata["ai_close_marker_source"] = "task_attachment"
        return
    if metadata.get("ai_close_marker_source") == "task_attachment":
        metadata["ai_close_incomplete"] = False
        metadata["ai_close_marker"] = ""
        metadata["ai_close_problem_types"] = []
        metadata["ai_close_has_not_done"] = False
        metadata["ai_close_has_unconfirmed"] = False
        metadata["ai_close_marker_source"] = ""


def _body_without_ai_close_marker(body: str) -> str:
    return "\n".join(
        line
        for line in str(body or "").splitlines()
        if not line.startswith("AI close marker:") and not line.startswith("AI close problem types:")
    ).strip()


def _body_with_ai_close_marker(body: str, metadata: dict[str, Any]) -> str:
    cleaned = _body_without_ai_close_marker(body)
    if not metadata.get("ai_close_incomplete"):
        return cleaned
    parts = [cleaned] if cleaned else []
    if INCOMPLETE_CLOSE_MARKER not in cleaned:
        parts.append(f"AI close marker: {INCOMPLETE_CLOSE_MARKER}")
    problem_types = _string_list(metadata.get("ai_close_problem_types"))
    if problem_types and "AI close problem types:" not in cleaned:
        parts.append("AI close problem types: " + ", ".join(problem_types))
    return "\n".join(parts).strip()


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


__all__ = [
    "BitrixTaskCloseDraft",
    "INCOMPLETE_CLOSE_MARKER",
    "TASK_CLOSE_DRAFT_TYPE",
    "TaskCloseDraftTool",
    "TaskCloseConfirmTool",
    "TaskCloseDiscardTool",
    "TaskCloseReportIncidentTool",
    "build_task_close_draft_from_args",
]
