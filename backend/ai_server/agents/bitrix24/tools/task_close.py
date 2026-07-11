from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ai_server.agents.bitrix24.ports import TaskDraftStorePort
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_policy import apply_write_policy
from ai_server.tools.bitrix_ports import BitrixWritePort
from ai_server.utils import compact_text, optional_int

TASK_CLOSE_DRAFT_TYPE = "task_close"
INCOMPLETE_CLOSE_MARKER = "AI_SERVER_TASK_CLOSE_INCOMPLETE"


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
                "tasks.task.result.add plus tasks.task.complete/tasks.task.approve is executed."
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
        await self._store.save_task_draft(dialog_key, draft.payload)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"draft": draft.payload, "preview": draft.preview},
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
            try:
                oauth_client = await self._bitrix_oauth.client_for_user(user_id)
                result = await _execute_task_close(oauth_client.call, draft)
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
            result = await _execute_task_close(self._write_client.call, draft)
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


async def _execute_task_close(
    call: Callable[[str, dict[str, Any]], Awaitable[Any]],
    draft: dict[str, Any],
) -> dict[str, Any]:
    task_id = optional_int(draft.get("task_id"))
    if task_id is None:
        raise ValueError("task closing draft has no task_id")
    result_text = compact_text(str(draft.get("result_text") or ""))
    action = _close_action(draft.get("action"))

    result_add = None
    if result_text:
        result_params = apply_write_policy(
            "tasks.task.result.add",
            {"taskId": task_id, "fields": {"TEXT": result_text}},
        )
        result_add = await call("tasks.task.result.add", result_params)

    close_method = "tasks.task.approve" if action == "approve" else "tasks.task.complete"
    close_params = apply_write_policy(close_method, {"taskId": task_id})
    close_result = await call(close_method, close_params)
    return {
        "task_id": task_id,
        "task_title": str(draft.get("task_title") or ""),
        "action": action,
        "result_method": "tasks.task.result.add" if result_text else "",
        "result_add": result_add,
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
        lines.append(f"Метка: {INCOMPLETE_CLOSE_MARKER}")
    if not_done_items:
        lines.append("Невыполненные пункты:")
        lines.extend(f"- {item}" for item in not_done_items)
    if unconfirmed_items:
        lines.append("Неподтверждённые пункты:")
        lines.extend(f"- {item}" for item in unconfirmed_items)
    return "\n".join(lines).strip()


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


__all__ = [
    "BitrixTaskCloseDraft",
    "INCOMPLETE_CLOSE_MARKER",
    "TASK_CLOSE_DRAFT_TYPE",
    "TaskCloseDraftTool",
    "TaskCloseConfirmTool",
    "TaskCloseDiscardTool",
    "build_task_close_draft_from_args",
]
