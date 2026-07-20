from __future__ import annotations

import base64
import html
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai_server.agents.bitrix24.ports import TaskDraftStorePort
from ai_server.agents.bitrix24.tools.draft_lifecycle import (
    attach_draft_metadata,
    bitrix_mutation_outcome,
    claim_exact_draft,
    discard_exact_draft,
    release_exact_draft,
    renew_exact_draft_claim,
    resolve_unknown_draft_claim,
)
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
    "status_reasons": ("status_reasons", "incomplete_reasons", "failure_reasons"),
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
    if _task_close_is_close_command_summary(completion_summary, task_id):
        completion_summary = ""
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
    status_reasons = _string_list(
        args.get("status_reasons") or args.get("incomplete_reasons") or args.get("failure_reasons")
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
            status_reasons,
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
        status_reasons=status_reasons,
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
        "status_reasons": status_reasons,
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
        "status_reasons": status_reasons,
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
                "Close an explicitly requested Bitrix task immediately as the current OAuth user, then prepare "
                "the protected four-block result-review draft. Use instead of direct bitrix_api writes. "
                "Call as soon as the task is identified; if completion details are missing, put them into "
                "missing_fields/unconfirmed_items so the user sees a full draft. Confirmation updates the "
                "protected AI close report and never closes the same task again."
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
                    "status_reasons": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Reasons from block 3 only: why the overall work status is partial or not done.",
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
                    "close_now": {
                        "type": "boolean",
                        "description": "True only for the user's explicit chat command to close this task now.",
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
        claimed_recovery = await self._store.get_claimed_task_draft(
            dialog_key,
            expected_type=TASK_CLOSE_DRAFT_TYPE,
        )
        if isinstance(claimed_recovery, dict):
            claimed_task_id = optional_int(claimed_recovery.get("task_id"))
            requested_task_id = optional_int(enriched_args.get("task_id"))
            if claimed_task_id != requested_task_id:
                return ToolResult(
                    status=ToolStatus.DENIED,
                    tool=self.name,
                    error="another task close operation has an unresolved outcome in this dialog",
                    data={"draft_id": claimed_recovery.get("_draft_id"), "task_id": claimed_task_id},
                )
            status_was_read = _truthy(enriched_args.get("_verified_remote_status_read"))
            if _truthy(enriched_args.get("_verified_remote_closed")):
                recovered = build_task_close_draft_from_args(
                    _merge_task_close_draft_args(claimed_recovery, args, enriched_args)
                )
                if not recovered.is_ready:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        tool=self.name,
                        error="closed task recovery draft is invalid",
                    )
                recovered.payload.update(
                    attach_draft_metadata(recovered.payload, source_args=claimed_recovery, user_id=user_id)
                )
                recovered.payload["already_closed"] = True
                recovered.payload["_direct_close_already_closed"] = True
                recovered.payload["_closed_from_chat"] = True
                finalized = await self._store.finalize_task_draft_claim(
                    dialog_key,
                    draft_id=str(claimed_recovery.get("_draft_id") or ""),
                    params=recovered.payload,
                    claim_token=str(claimed_recovery.get("_draft_claim_token") or ""),
                )
                if finalized is None:
                    return ToolResult(
                        status=ToolStatus.ERROR,
                        tool=self.name,
                        error="closed task recovery could not finalize the protected review draft",
                    )
                return ToolResult(
                    status=ToolStatus.OK,
                    tool=self.name,
                    data={"status": "recovered_closed_draft", "draft": finalized, "preview": recovered.preview},
                )
            if not status_was_read:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="task close outcome is unresolved; fresh current-user Bitrix status proof is required",
                    data={"draft_id": claimed_recovery.get("_draft_id"), "task_id": claimed_task_id},
                )
            await self._store.release_task_draft(
                dialog_key,
                draft_id=str(claimed_recovery.get("_draft_id") or ""),
                claim_token=str(claimed_recovery.get("_draft_claim_token") or ""),
            )
        draft = build_task_close_draft_from_args(enriched_args)
        if not draft.is_ready:
            return ToolResult(
                status=ToolStatus.CONTRACT_VIOLATION,
                tool=self.name,
                error=f"task_close_draft called outside contract: {'; '.join(draft.contract_errors)}",
                data=draft.as_action_details(),
            )
        draft.payload.update(attach_draft_metadata(draft.payload, source_args=args, user_id=user_id))
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
            draft = merged
            draft_status = "updated_draft"
        elif isinstance(existing, dict):
            return ToolResult(
                status=ToolStatus.DENIED,
                tool=self.name,
                error="another typed draft is active in this dialog",
                data={
                    "status": "active_draft_conflict",
                    "draft_id": existing.get("_draft_id"),
                    "draft_type": existing.get("_draft_type"),
                },
            )
        else:
            draft_status = "new_draft"

        draft.payload.update(
            attach_draft_metadata(draft.payload, source_args={**(existing or {}), **args}, user_id=user_id)
        )
        await self._store.save_task_draft(dialog_key, draft.payload)
        stored = await self._store.get_task_draft(dialog_key)
        should_close_now = _truthy(enriched_args.get("close_now")) and not _truthy(enriched_args.get("already_closed"))
        if should_close_now:
            if not isinstance(stored, dict):
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="reserved task close draft could not be reloaded",
                )
            claimed = await claim_exact_draft(
                self._store,
                dialog_key=dialog_key,
                draft=stored,
                expected_type=TASK_CLOSE_DRAFT_TYPE,
            )
            if claimed is None:
                return ToolResult(
                    status=ToolStatus.DENIED,
                    tool=self.name,
                    error="task close draft changed, expired, or is already being processed",
                )
            if not await renew_exact_draft_claim(self._store, dialog_key=dialog_key, draft=claimed):
                return ToolResult(status=ToolStatus.DENIED, tool=self.name, error="task close draft claim was lost")
            close_error = await self._close_now_as_current_user(enriched_args, user_id=user_id)
            if close_error is not None:
                if str((close_error.data or {}).get("mutation_outcome") or "") in {
                    "not_started",
                    "rejected",
                }:
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return close_error
            draft.payload["already_closed"] = True
            draft.payload["_direct_close_already_closed"] = True
            draft.payload["_closed_from_chat"] = True
            finalized = await self._store.finalize_task_draft_claim(
                dialog_key,
                draft_id=str(claimed.get("_draft_id") or ""),
                params=draft.payload,
                claim_token=str(claimed.get("_draft_claim_token") or ""),
            )
            if finalized is None:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="task was closed but the protected review draft could not be finalized; audit is required",
                )
            draft.payload.update(finalized)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"status": draft_status, "draft": draft.payload, "preview": draft.preview},
        )

    async def _close_now_as_current_user(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None,
    ) -> ToolResult | None:
        task_id = optional_int(args.get("task_id"))
        if task_id is None:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="task_close_draft.close_now requires task_id",
            )
        if self._bitrix_oauth is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="Bitrix OAuth is required to close a task from chat.",
                data={"task_id": task_id, "mutation_outcome": "not_started"},
            )
        try:
            oauth_client = await self._bitrix_oauth.client_for_user(user_id)
            method = "tasks.task.approve" if _close_action(args.get("action")) == "approve" else "tasks.task.complete"
            await oauth_client.call(method, apply_write_policy(method, {"taskId": task_id}))
        except BitrixOAuthTokenMissing as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error=str(exc),
                data={"task_id": task_id, "mutation_outcome": "not_started"},
            )
        except BitrixConfigError as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error=str(exc),
                data={"task_id": task_id, "mutation_outcome": "not_started"},
            )
        except BitrixApiError as exc:
            outcome = "unknown" if str(exc.error).upper().startswith("HTTP_5") else "rejected"
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=str(exc),
                data={"task_id": task_id, "mutation_outcome": outcome},
            )
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=f"{type(exc).__name__}: {exc}",
                data={"task_id": task_id, "mutation_outcome": "unknown"},
            )
        return None

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
        verified_closed = status == "5" or bool(_first_text(task, "closedDate", "CLOSED_DATE"))
        enriched["_verified_remote_status_read"] = True
        enriched["_verified_remote_closed"] = verified_closed
        enriched["already_closed"] = verified_closed

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
        if not draft:
            unresolved = await resolve_unknown_draft_claim(
                self._store, dialog_key=dialog_key, expected_type=TASK_CLOSE_DRAFT_TYPE
            )
            if unresolved is not None:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error="task close outcome requires durable status reconciliation before retry",
                    data={
                        "mutation_outcome": "unknown",
                        "resolution_status": unresolved.get("_draft_resolution_status"),
                        "draft_id": unresolved.get("_draft_id"),
                    },
                )
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
            claimed = None
            try:
                oauth_client = await self._bitrix_oauth.client_for_user(user_id)
                claimed = await claim_exact_draft(
                    self._store, dialog_key=dialog_key, draft=draft, expected_type=TASK_CLOSE_DRAFT_TYPE
                )
                if claimed is None:
                    return ToolResult(
                        status=ToolStatus.DENIED,
                        tool=self.name,
                        error="task close draft changed, expired, or is already being confirmed",
                    )
                result = await _execute_task_close(
                    close_call=oauth_client.call,
                    report_call=self._write_client.call if self._write_client is not None else None,
                    draft=draft,
                    claim_guard=lambda: renew_exact_draft_claim(self._store, dialog_key=dialog_key, draft=claimed),
                )
            except BitrixOAuthTokenMissing as exc:
                if claimed is not None:
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED, tool=self.name, error=str(exc), data={"draft": draft}
                )
            except BitrixConfigError as exc:
                if claimed is not None:
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED,
                    tool=self.name,
                    error=str(exc),
                    data={"draft": draft, "mutation_outcome": "not_started"},
                )
            except BitrixApiError as exc:
                outcome = bitrix_mutation_outcome(exc)
                if claimed is not None and outcome == "rejected":
                    await release_exact_draft(self._store, dialog_key=dialog_key, draft=claimed)
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error=str(exc),
                    data={"draft": draft, "mutation_outcome": outcome},
                )
            except Exception as exc:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    tool=self.name,
                    error=f"{type(exc).__name__}: {exc}",
                    data={"draft": draft, "mutation_outcome": "unknown"},
                )
            _complete_direct_close_queue_from_draft(self._store, draft, actor_user_id=user_id)
            await self._store.delete_task_draft(
                dialog_key,
                status="confirmed",
                expected_draft_id=str(claimed.get("_draft_id") or ""),
                expected_version=optional_int(claimed.get("_draft_version")),
                expected_claim_token=str(claimed.get("_draft_claim_token") or ""),
            )
            return ToolResult(status=ToolStatus.OK, tool=self.name, data={**result, "draft": draft})

        if self._write_client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="write_client is not configured",
                data={"draft": draft},
            )
        claimed = await claim_exact_draft(
            self._store, dialog_key=dialog_key, draft=draft, expected_type=TASK_CLOSE_DRAFT_TYPE
        )
        if claimed is None:
            return ToolResult(
                status=ToolStatus.DENIED,
                tool=self.name,
                error="task close draft changed, expired, or is already being confirmed",
            )
        try:
            result = await _execute_task_close(
                close_call=self._write_client.call,
                report_call=self._write_client.call,
                draft=draft,
                claim_guard=lambda: renew_exact_draft_claim(self._store, dialog_key=dialog_key, draft=claimed),
            )
        except Exception as exc:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=f"{type(exc).__name__}: {exc}",
                data={"draft": draft, "mutation_outcome": "unknown"},
            )
        _complete_direct_close_queue_from_draft(self._store, draft, actor_user_id=user_id)
        await self._store.delete_task_draft(
            dialog_key,
            status="confirmed",
            expected_draft_id=str(claimed.get("_draft_id") or ""),
            expected_version=optional_int(claimed.get("_draft_version")),
            expected_claim_token=str(claimed.get("_draft_claim_token") or ""),
        )
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
        if dialog_key and isinstance(draft, dict) and draft.get("_draft_type") == TASK_CLOSE_DRAFT_TYPE:
            await discard_exact_draft(
                self._store,
                dialog_key=dialog_key,
                draft=draft,
                expected_type=TASK_CLOSE_DRAFT_TYPE,
            )
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
    claim_guard: Callable[[], Awaitable[bool]] | None = None,
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
        await _require_current_claim(claim_guard)
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
            await _require_current_claim(claim_guard)
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
            await _require_current_claim(claim_guard)
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


async def _require_current_claim(claim_guard: Callable[[], Awaitable[bool]] | None) -> None:
    if claim_guard is not None and not await claim_guard():
        raise RuntimeError("DRAFT_CLAIM_LOST")


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
    task_id = optional_int(args.get("task_id") or args.get("id") or args.get("ID"))
    completion_summary = compact_text(
        str(args.get("completion_summary") or args.get("result_text") or args.get("summary") or "")
    )
    if completion_summary and not _task_close_is_close_command_summary(completion_summary, task_id):
        return True
    if compact_text(str(args.get("equipment_consumables") or args.get("equipment") or args.get("consumables") or "")):
        return True
    if compact_text(str(args.get("additional_info") or args.get("extra_info") or args.get("object_notes") or "")):
        return True
    if _string_list(args.get("status_reasons") or args.get("incomplete_reasons") or args.get("failure_reasons")):
        return True
    if _overall_status(args.get("overall_status") or args.get("completion_status")) not in {"", "unconfirmed"}:
        return True
    return bool(
        _string_list(args.get("not_done_items") or args.get("unfinished_items") or args.get("incomplete_items"))
    )


_TASK_CLOSE_COMMAND_SUMMARIES = frozenset(
    {
        "закрой",
        "закрой задачу",
        "закрой эту задачу",
        "закройте задачу",
        "закрыть",
        "закрыть задачу",
        "задачу закрыть",
        "закрываем задачу",
        "отметь задачу выполненной",
        "отметить задачу выполненной",
        "пометь задачу выполненной",
        "пометить задачу выполненной",
        "close",
        "close task",
        "complete task",
        "mark task complete",
        "mark task completed",
    }
)


def _task_close_is_close_command_summary(value: str, task_id: int | None) -> bool:
    text = compact_text(str(value or "")).casefold().replace("ё", "е")
    if not text:
        return False
    if task_id is not None:
        text = re.sub(rf"(?<!\d)#?{re.escape(str(task_id))}(?!\d)", " ", text)
    text = re.sub(r"[\"'`«»“”№#.,;:!?()\[\]\-]+", " ", text)
    ignored_tokens = {"bitrix", "битрикс", "пожалуйста"}
    text = compact_text(" ".join(token for token in text.split() if token not in ignored_tokens))
    return text in _TASK_CLOSE_COMMAND_SUMMARIES


def _result_text(
    *,
    completion_summary: str,
    task_points: list[str],
    equipment_consumables: str,
    additional_info: str,
    overall_status_label: str,
    not_done_items: list[str],
    unconfirmed_items: list[str],
    status_reasons: list[str],
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
    if status_reasons:
        lines.append("")
        lines.append("Причины неполного выполнения:")
        lines.extend(f"- {item}" for item in status_reasons)
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
    task_points = _string_list(update_payload.get("task_points") or existing.get("task_points"))
    point_status_updates = _task_close_point_status_updates(update_payload, task_points=task_points)

    for payload_field, aliases in _TASK_CLOSE_SCALAR_FIELDS.items():
        if not _has_any_key(raw_args, aliases):
            continue
        value = compact_text(str(update_payload.get(payload_field) or ""))
        if value:
            if payload_field == "completion_summary":
                merged[payload_field] = _merge_task_close_completion_summary(
                    str(existing.get(payload_field) or ""),
                    value,
                    task_points=task_points,
                )
            else:
                merged[payload_field] = value

    for payload_field, aliases in _TASK_CLOSE_LIST_FIELDS.items():
        if not _has_any_key(raw_args, aliases):
            continue
        if point_status_updates and payload_field in {"not_done_items", "unconfirmed_items"}:
            continue
        merged[payload_field] = _string_list(update_payload.get(payload_field))

    for payload_field, aliases in _TASK_CLOSE_BOOL_FIELDS.items():
        if not _has_any_key(raw_args, aliases):
            continue
        merged[payload_field] = bool(update_payload.get(payload_field))

    if point_status_updates:
        _apply_task_close_point_status_updates(
            merged,
            point_status_updates,
            extra_not_done=_task_close_items_without_task_point(update_payload.get("not_done_items"), task_points),
            extra_unconfirmed=_task_close_items_without_task_point(
                update_payload.get("unconfirmed_items"), task_points
            ),
        )
    if (
        _has_any_key(raw_args, _TASK_CLOSE_SCALAR_FIELDS["overall_status"])
        and _overall_status(update_payload.get("overall_status")) == "completed"
    ):
        merged["status_reasons"] = []

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
        "status_reasons": _string_list(payload.get("status_reasons")),
        "unresolved_items": _string_list(payload.get("unresolved_items")),
        "missing_fields": _string_list(payload.get("missing_fields")),
        "problem_types": _string_list(payload.get("problem_types")),
        "ai_close_incomplete": bool(payload.get("ai_close_incomplete")),
        "ai_close_marker": str(payload.get("ai_close_marker") or ""),
        "result_text": str(payload.get("result_text") or ""),
    }


def _has_any_key(values: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in values for key in keys)


def _merge_task_close_completion_summary(existing: str, update: str, *, task_points: list[str]) -> str:
    existing_text = compact_text(existing)
    update_text = compact_text(update)
    if not existing_text or not update_text or not task_points:
        return update_text or existing_text
    updated_points = _task_close_points_mentioned(update_text, task_points=task_points)
    if not updated_points:
        return update_text
    kept_existing = [
        clause
        for clause in _task_close_summary_clauses(existing_text)
        if not any(_task_close_text_matches_point(point, clause) for point in updated_points)
    ]
    return ". ".join(_unique_strings([*kept_existing, *_task_close_summary_clauses(update_text)]))


def _task_close_point_status_updates(update_payload: dict[str, Any], *, task_points: list[str]) -> dict[str, str]:
    if not task_points:
        return {}
    updates: dict[str, str] = {}
    completion_summary = compact_text(str(update_payload.get("completion_summary") or ""))
    for clause in _task_close_summary_clauses(completion_summary):
        status = _task_close_clause_status(clause)
        if not status:
            continue
        for point in _task_close_points_mentioned(clause, task_points=task_points):
            updates[point] = status
    for item in _string_list(update_payload.get("not_done_items")):
        for point in _task_close_points_mentioned(item, task_points=task_points):
            updates[point] = "not_done"
    for item in _string_list(update_payload.get("unconfirmed_items")):
        for point in _task_close_points_mentioned(item, task_points=task_points):
            updates[point] = "unconfirmed"
    return updates


def _apply_task_close_point_status_updates(
    merged: dict[str, Any],
    updates: dict[str, str],
    *,
    extra_not_done: list[str],
    extra_unconfirmed: list[str],
) -> None:
    updated_points = list(updates)
    not_done = [
        item
        for item in _string_list(merged.get("not_done_items"))
        if not any(_task_close_text_matches_point(point, item) for point in updated_points)
    ]
    unconfirmed = [
        item
        for item in _string_list(merged.get("unconfirmed_items"))
        if not any(_task_close_text_matches_point(point, item) for point in updated_points)
    ]
    for point, status in updates.items():
        if status == "not_done":
            not_done.append(point)
        elif status == "unconfirmed":
            unconfirmed.append(point)
    not_done.extend(extra_not_done)
    unconfirmed.extend(extra_unconfirmed)
    merged["not_done_items"] = _unique_strings(not_done)
    merged["unconfirmed_items"] = _unique_strings(unconfirmed)


def _task_close_items_without_task_point(value: object, task_points: list[str]) -> list[str]:
    return [
        item
        for item in _string_list(value)
        if not any(_task_close_text_matches_point(point, item) for point in task_points)
    ]


def _task_close_summary_clauses(value: str) -> list[str]:
    text = compact_text(value)
    if not text:
        return []
    return _unique_strings([compact_text(part).strip(" .;:-") for part in re.split(r"(?:\r?\n)+|(?<!\d)[.,;]+", text)])


def _task_close_points_mentioned(value: str, *, task_points: list[str]) -> list[str]:
    return [point for point in task_points if _task_close_text_matches_point(point, value)]


def _task_close_text_matches_point(point: str, value: str) -> bool:
    point_norm = _task_close_match_text(point)
    value_norm = _task_close_match_text(value)
    if not point_norm or not value_norm:
        return False
    if point_norm in value_norm or value_norm in point_norm:
        return True
    point_tokens = set(point_norm.split())
    value_tokens = set(value_norm.split())
    if not point_tokens or not value_tokens:
        return False
    return len(point_tokens & value_tokens) >= min(len(point_tokens), 2)


def _task_close_match_text(value: str) -> str:
    text = compact_text(value).casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text, flags=re.IGNORECASE)
    ignored = {"и", "или", "по", "на", "в", "во", "с", "со", "к", "ко", "для", "пункт"}
    return compact_text(" ".join(token for token in text.split() if token not in ignored))


def _task_close_clause_status(value: str) -> str:
    text = compact_text(value).casefold().replace("ё", "е")
    if not text:
        return ""
    if (
        "не выполн" in text
        or "не сдел" in text
        or "не постав" in text
        or "не установ" in text
        or "не забрал" in text
        or "не почини" in text
        or "not done" in text
        or "not completed" in text
    ):
        return "not_done"
    if "не подтверж" in text or "неизвест" in text or "не провер" in text or "unconfirmed" in text:
        return "unconfirmed"
    if "выполн" in text or "сделан" in text or "готов" in text or "completed" in text or "done" in text:
        return "completed"
    return ""


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
    equipment_consumables = compact_text(str(draft.get("equipment_consumables") or ""))
    additional_info = compact_text(str(draft.get("additional_info") or ""))
    overall_status_label = compact_text(str(draft.get("overall_status_label") or ""))
    not_done_items = _string_list(draft.get("not_done_items"))
    unconfirmed_items = _string_list(draft.get("unconfirmed_items") or draft.get("unresolved_items"))
    missing_fields = _string_list(draft.get("missing_fields"))
    status = _report_file_status(draft)
    problem_types = _string_list(draft.get("problem_types"))
    if not problem_types:
        if not_done_items:
            problem_types.append("not_done")
        if unconfirmed_items:
            problem_types.append("unconfirmed")
    human_report = format_task_close_draft_message(
        draft,
        heading=f"Задача #{task_id}: {task_title or f'#{task_id}'}",
        include_instruction=False,
    )
    lines = [
        "AI-close report",
        human_report,
        "",
        "---",
        "Machine metadata",
        f"Task ID: {task_id}",
        f"Task: {task_title or f'#{task_id}'}",
        f"Action: {action}",
        f"Status: {status}",
        f"Created at: {datetime.now(MOSCOW_TZ).isoformat(timespec='seconds')}",
        "Source: AI-server task_close_confirm",
    ]
    if problem_types:
        lines.append(f"Problem types: {', '.join(_unique_strings(problem_types))}")
        lines.append(f"AI marker: {INCOMPLETE_CLOSE_MARKER}")
    if completion_summary:
        lines.extend(["", "Summary:", completion_summary])
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


def format_task_close_draft_message(
    draft: dict[str, Any],
    *,
    preview: dict[str, Any] | None = None,
    intro_lines: list[str] | None = None,
    heading: str | None = None,
    include_instruction: bool = True,
) -> str:
    preview = preview if isinstance(preview, dict) else {}
    task_id = _draft_text(draft.get("task_id"))
    task_title = _draft_text(preview.get("task_title")) or _draft_text(draft.get("task_title")) or f"задача #{task_id}"
    raw_result_text = (
        _draft_text(preview.get("completion_summary"))
        or _draft_text(draft.get("completion_summary"))
        or _draft_text(draft.get("result_text"))
    )
    result_text = "" if _task_close_is_placeholder_summary(raw_result_text) else raw_result_text
    task_points = _string_list(preview.get("task_points") or draft.get("task_points"))
    source_task_description_empty = bool(
        preview.get("source_task_description_empty") or draft.get("source_task_description_empty")
    )
    equipment = _draft_text(preview.get("equipment_consumables")) or _draft_text(draft.get("equipment_consumables"))
    additional_info = _draft_text(preview.get("additional_info")) or _draft_text(draft.get("additional_info"))
    overall_status = _draft_text(preview.get("overall_status")) or _draft_text(draft.get("overall_status"))
    overall = _draft_text(preview.get("overall_status_label")) or _draft_text(draft.get("overall_status_label"))
    if overall.casefold() == "результат не подтверждён" and not result_text:
        overall = ""
    not_done = _string_list(preview.get("not_done_items") or draft.get("not_done_items"))
    unconfirmed = _string_list(preview.get("unconfirmed_items") or draft.get("unconfirmed_items"))
    status_reasons = _string_list(preview.get("status_reasons") or draft.get("status_reasons"))
    unresolved = _string_list(preview.get("unresolved_items") or draft.get("unresolved_items"))
    if not unconfirmed and unresolved:
        unconfirmed = unresolved
    unconfirmed = _task_close_visible_unconfirmed(unconfirmed, task_points=task_points, result_text=result_text)

    lines: list[str] = [line for line in (intro_lines or []) if str(line).strip()]
    if lines:
        lines.append("")
    lines.extend([heading or f"Черновик #{task_id or '?'}: {task_title}", ""])
    lines.append("1. Выполняемые работы")
    if source_task_description_empty:
        result_items = _task_close_display_items(result_text)
        if result_items:
            lines.extend(f"1.{index} {item}" for index, item in enumerate(result_items, start=1))
            lines.append(f"1.{len(result_items) + 1} Еще работы - ... ???")
        else:
            lines.append("1.1 Еще работы - ... ???")
    elif task_points:
        lines.extend(
            f"1.{index} {item} - {_task_close_point_status(item, result_text, not_done, unconfirmed)}"
            for index, item in enumerate(task_points, start=1)
        )
        lines.append(f"1.{len(task_points) + 1} Еще работы - ... ???")
    elif result_text:
        result_items = _task_close_display_items(result_text)
        lines.extend(f"1.{index} {item}" for index, item in enumerate(result_items, start=1))
        lines.append(f"1.{len(result_items) + 1} Еще работы - ... ???")
    else:
        lines.append("1.1 Еще работы - ... ???")

    lines.append("")
    lines.append("2. Использовано материалов, оборудование")
    if _task_close_no_equipment_value(equipment):
        lines.append("   (перечисли, что использовали, или укажи: [ВЫБРАНО: не использовалось])")
    else:
        lines.append("   (перечисли, что использовали, или укажи: не использовалось)")
        equipment_items = _task_close_display_items(equipment)
        if equipment_items:
            lines.extend(f"2.{index} {item}" for index, item in enumerate(equipment_items, start=1))
            lines.append(f"2.{len(equipment_items) + 1} Еще материалы - ... ???")

    lines.append("")
    lines.append("3. Статус выполнения работ")
    lines.append(f"   {_task_close_status_prompt(overall_status or overall)}")
    if status_reasons:
        for index, item in enumerate(status_reasons, start=1):
            lines.append(f"3.{index} причина: {item}")
        lines.append(f"3.{len(status_reasons) + 1} Еще причины невыполнения - ... ???")
    elif _task_close_status_needs_reason(overall_status or overall):
        lines.append("3.1 Еще причины невыполнения - ... ???")

    lines.append("")
    lines.append("4. Дополнительная информация")
    if _task_close_absent_value(additional_info):
        lines.append("   (если есть - укажи; если нет - укажи: [ВЫБРАНО: отсутствует])")
    else:
        lines.append("   (если есть - укажи; если нет - укажи: отсутствует)")
    additional_items = _task_close_display_items(additional_info)
    if additional_items and not _task_close_absent_value(additional_info):
        lines.extend(f"4.{index} {item}" for index, item in enumerate(additional_items, start=1))
        lines.append(f"4.{len(additional_items) + 1} Еще информация - ... ???")

    if include_instruction:
        lines.extend(
            [
                "",
                'Внести изменения (укажите пункт или подпункт и нужную информацию) или напишите: "да, закрывай как есть".',
            ]
        )
    return "\n".join(lines)


def _draft_text(value: object) -> str:
    return compact_text(str(value or ""))


def _task_close_status_prompt(status: str) -> str:
    normalized = _task_close_normalized_status(status)
    options = [
        ("completed", "выполнено полностью"),
        ("partial", "выполнено частично"),
        ("not_done", "не выполнено"),
    ]
    rendered = [f"[ВЫБРАНО: {label}]" if normalized == key else label for key, label in options]
    return (
        f"({' / '.join(rendered)}; если выполнено частично или не выполнено - укажи причину неполного выполнения работ)"
    )


def _task_close_normalized_status(status: str) -> str:
    text = compact_text(status).casefold()
    if text in {"completed", "complete", "done"} or "полностью" in text:
        return "completed"
    if text in {"partial", "partially_done", "partly_done"} or "частич" in text:
        return "partial"
    if text in {"not_done", "not_completed", "not done"} or "не выполн" in text or "не сделан" in text:
        return "not_done"
    return ""


def _task_close_status_needs_reason(status: str) -> bool:
    return _task_close_normalized_status(status) in {"partial", "not_done"}


def _task_close_absent_value(value: str) -> bool:
    text = compact_text(value).casefold()
    return text in {"отсутствует", "нет", "не было", "нет дополнительной информации"}


def _task_close_no_equipment_value(value: str) -> bool:
    text = compact_text(value).casefold().strip(" .;:-")
    if not text:
        return False
    if text in {
        "не использовалось",
        "не использовались",
        "не использовали",
        "не применялось",
        "не применяли",
        "не было",
        "нет",
        "ничего",
        "без материалов",
        "без оборудования",
        "материалы не использовались",
        "оборудование не использовалось",
    }:
        return True
    return ("не использ" in text or "не примен" in text) and not any(char.isdigit() for char in text)


def _task_close_is_placeholder_summary(value: str) -> bool:
    text = compact_text(value)
    if not text:
        return False
    lowered = text.casefold()
    if "статус ai-закрытия" in lowered:
        return True
    if "результат выполнения не указан" in lowered:
        return True
    return "общий итог: результат не подтвержд" in lowered and (
        lowered.startswith("пункты задачи:") or "неподтверждённые пункты" in lowered
    )


def _task_close_visible_unconfirmed(
    values: list[str],
    *,
    task_points: list[str],
    result_text: str,
) -> list[str]:
    point_keys = {item.casefold() for item in task_points}
    visible: list[str] = []
    for value in values:
        key = value.casefold()
        if key in {"результат выполнения не указан", "result is not specified"}:
            continue
        if ("результат закрытия" in key and "не подтверж" in key) or "ai-черновик" in key or "ai draft" in key:
            continue
        if key in point_keys:
            if result_text and _task_close_result_explicitly_marks_unconfirmed(value, result_text):
                visible.append(value)
            continue
        visible.append(value)
    return visible


def _task_close_result_explicitly_marks_unconfirmed(point: str, result_text: str) -> bool:
    point_cf = point.casefold()
    for clause in re.split(r"(?:\r?\n)+|[.;]+|,(?=\s*\d+\.\d+)", result_text):
        clause_cf = clause.casefold()
        if not any(marker in clause_cf for marker in ("не подтверж", "неизвест", "не провер", "unconfirmed")):
            continue
        if _task_close_texts_overlap(point_cf, clause):
            return True
    return False


def _task_close_display_items(value: str) -> list[str]:
    text = compact_text(value)
    if not text:
        return []
    raw_parts = re.split(r"(?:\r?\n)+|(?:^|\s)(?:\d+[.)]|[-*•])\s+|[,;]+", text)
    parts = _unique_strings([part.strip(" .;:-") for part in raw_parts])
    return parts[:12] if len(parts) > 1 else ([text] if text else [])


def _task_close_point_status(
    point: str,
    result_text: str,
    not_done: list[str],
    unconfirmed: list[str],
) -> str:
    point_cf = point.casefold()
    for item in not_done:
        if _task_close_texts_overlap(point_cf, item):
            return "не выполнено"
    for item in unconfirmed:
        if _task_close_texts_overlap(point_cf, item):
            return "не подтверждено"
    completed_label = _task_close_point_completed_label(point, result_text)
    if completed_label:
        return completed_label
    return "... ???"


def _task_close_point_completed_label(point: str, result_text: str) -> str:
    if not result_text:
        return ""
    point_cf = point.casefold()
    clauses = re.split(r"(?:\r?\n)+|[.;]+|,(?=\s*\d+\.\d+)", result_text)
    for clause in clauses:
        clause_cf = clause.casefold()
        if "не выполн" in clause_cf or "не подтверж" in clause_cf:
            continue
        if ("выполн" in clause_cf or "сделан" in clause_cf or "готов" in clause_cf) and _task_close_texts_overlap(
            point_cf, clause
        ):
            return "выполнено полностью" if "полностью" in clause_cf else "выполнено"
    return ""


def _task_close_texts_overlap(point_cf: str, value: str) -> bool:
    value_cf = value.casefold()
    if point_cf and point_cf in value_cf:
        return True
    words = [word for word in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", point_cf) if len(word) >= 5]
    return bool(words and any(word in value_cf for word in words[:4]))


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
