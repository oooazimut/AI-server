from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_server.agents.bitrix24.ports import TaskDraftStorePort
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.integrations.bitrix.portal_search.search_index import PortalSearchIndex
from ai_server.integrations.bitrix.task_close_reports import (
    canonical_task_close_report_file_name,
    restore_task_close_report_file,
    task_close_report_key,
    task_close_report_problem_types,
    task_close_report_state_key,
)
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.settings import Settings, get_settings
from ai_server.tools.bitrix_policy import apply_write_policy
from ai_server.tools.bitrix_ports import BitrixWritePort
from ai_server.utils import MOSCOW_TZ, compact_text, optional_int

TASK_CLOSE_DRAFT_TYPE = "task_close"
INCOMPLETE_CLOSE_MARKER = "AI_SERVER_TASK_CLOSE_INCOMPLETE"

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
    equipment_consumables = compact_text(
        str(args.get("equipment_consumables") or args.get("equipment") or args.get("consumables") or "")
    )
    overall_status = _overall_status(args.get("overall_status") or args.get("completion_status"))
    overall_status_label = _overall_status_label(overall_status)
    not_done_items = _string_list(
        args.get("not_done_items") or args.get("unfinished_items") or args.get("incomplete_items")
    )
    unconfirmed_items = _unique_strings(
        [
            *_string_list(args.get("unconfirmed_items") or args.get("unknown_items") or args.get("unverified_items")),
            *unresolved_items,
        ]
    )
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
        "equipment_consumables": equipment_consumables,
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
        "equipment_consumables": equipment_consumables,
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

    def __init__(self, store: TaskDraftStorePort) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Prepare a Bitrix task closing draft. Use instead of direct bitrix_api writes for task closing. "
                "Call only after the task and completion result are understood. The user must confirm before "
                "a protected AI close report file plus tasks.task.complete/tasks.task.approve is executed."
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
        draft = build_task_close_draft_from_args(args)
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
        if dialog_key:
            await self._store.delete_task_draft(dialog_key)
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
    close_params = apply_write_policy(close_method, {"taskId": task_id})
    close_result = await close_call(close_method, close_params)

    report_add = None
    report_method = ""
    report_file_name = ""
    report_file_content = ""
    if result_text:
        if report_call is None:
            raise BitrixConfigError("system Bitrix write_client is required to attach protected task close report.")
        report_file_name = _report_file_name(task_id, draft)
        report_file_content = _report_file_content(task_id=task_id, action=action, draft=draft)
        existing_report = await _find_existing_report_file(report_call, task_id=task_id, file_name=report_file_name)
        if existing_report is not None:
            report_add = existing_report
            report_method = "task.item.getfiles"
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


def _result_text(
    *,
    completion_summary: str,
    task_points: list[str],
    equipment_consumables: str,
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

    return merged


def _preview_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_title": str(payload.get("task_title") or ""),
        "action": _close_action(payload.get("action")),
        "action_label": _action_label(_close_action(payload.get("action"))),
        "completion_summary": str(payload.get("completion_summary") or ""),
        "task_points": _string_list(payload.get("task_points")),
        "equipment_consumables": str(payload.get("equipment_consumables") or ""),
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
    for item in _extract_file_items(result):
        if str(item.get("NAME") or item.get("name") or "") == file_name:
            return item
    return None


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


def _report_file_name(task_id: int, draft: dict[str, Any]) -> str:
    suffix = _report_file_status(draft)
    return f"AI-close-{task_id}-{suffix}.txt"


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
    if completion_summary:
        lines.extend(["", "Summary:", completion_summary])
    if task_points:
        lines.append("")
        lines.append("Task points:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(task_points, start=1))
    if equipment_consumables:
        lines.extend(["", "Equipment and consumables:", equipment_consumables])
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
