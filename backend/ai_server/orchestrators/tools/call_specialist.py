from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ai_server.agents.ports import OrchestratorStorePort, SchedulerPort
from ai_server.capability_registry import build_capability_registry
from ai_server.models import (
    AgentManifest,
    AgentTask,
    ScheduledTask,
    ToolDefinition,
    ToolResult,
    ToolStatus,
    UserContext,
)

if TYPE_CHECKING:
    from ai_server.specialists import Specialist

logger = logging.getLogger(__name__)

_ACTIVE_DRAFT_BRANCH_FIELD = "conversation_reference_active_draft_branch"
_ACTIVE_DRAFT_NUMBER_FIELD = "conversation_reference_active_draft_number"


class CallSpecialistTool:
    """Routes a request to a named specialist agent.

    One instance is shared across all specialist calls; the target is chosen via
    the ``specialist_id`` enum argument so the LLM still gets per-specialist descriptions.
    """

    name = "call_specialist"

    def __init__(
        self,
        specialists: dict[str, Specialist],
        manifests: list[AgentManifest],
        *,
        scheduler: SchedulerPort | None = None,
        store: OrchestratorStorePort | None = None,
        schedule_fn: Callable[[list[ScheduledTask]], None] | None = None,
    ) -> None:
        self._specialists = specialists
        self._manifests = {m.id: m for m in manifests if m.kind == "specialist"}
        self._scheduler = scheduler
        self._store = store
        self.schedule_fn = schedule_fn  # set post-init to break circular dep

    def definition(self) -> ToolDefinition:
        specialist_ids = list(self._specialists.keys())
        descriptions: dict[str, str] = {}
        for sid in specialist_ids:
            m = self._manifests.get(sid)
            descriptions[sid] = (m.handoff_description or m.name) if m else sid

        enum_desc = "; ".join(f"{sid} — {desc}" for sid, desc in descriptions.items())
        return ToolDefinition(
            name=self.name,
            description=f"Делегировать задачу специалисту-субагенту. Доступные специалисты: {enum_desc}.",
            parameters={
                "type": "object",
                "properties": {
                    "specialist_id": {
                        "type": "string",
                        "enum": specialist_ids,
                        "description": "ID специалиста",
                    },
                    "request": {
                        "type": "string",
                        "description": "Полный текст задачи для специалиста",
                    },
                },
                "required": ["specialist_id", "request"],
            },
        )

    def capability_registry(self, specialist_id: str) -> dict[str, Any] | None:
        """Read a fresh registry from the same specialist instance that will execute it."""

        specialist = self._specialists.get(specialist_id)
        manifest = self._manifests.get(specialist_id)
        if specialist is None or manifest is None:
            return None
        provider = getattr(specialist, "capability_registry", None)
        if callable(provider):
            value = provider()
            return dict(value) if isinstance(value, dict) else None
        definitions = getattr(specialist, "tool_definitions", None)
        raw = definitions() if callable(definitions) else []
        return build_capability_registry(
            manifest,
            [item for item in raw if isinstance(item, dict)],
            structured_tool_names=set(),
        )

    async def get_active_bitrix_draft(self, dialog_key: str) -> dict[str, Any] | None:
        """Read a Bitrix write draft without invoking a model or Bitrix API."""
        specialist = self._specialists.get("bitrix24")
        getter = getattr(specialist, "get_active_draft", None)
        if not callable(getter) or not dialog_key:
            return None
        draft = await getter(dialog_key)
        return dict(draft) if isinstance(draft, dict) else None

    async def discard_active_bitrix_draft(self, dialog_key: str, *, expected_draft_id: str) -> bool:
        specialist = self._specialists.get("bitrix24")
        discard = getattr(specialist, "discard_active_draft", None)
        if not callable(discard) or not dialog_key or not expected_draft_id:
            return False
        return bool(await discard(dialog_key, expected_draft_id=expected_draft_id))

    async def _persist_pending_state(
        self,
        *,
        dialog_key: str | None,
        specialist_id: str,
        specialist_status: str,
    ) -> ToolResult | None:
        if self._store is None or not dialog_key:
            return None
        try:
            if specialist_status in ("needs_clarification", "needs_human"):
                await self._store.set_kv(dialog_key, "pending_specialist", specialist_id)
                transition = "set"
            else:
                await self._store.delete_kv(dialog_key, "pending_specialist")
                transition = "clear"
        except Exception:
            logger.exception("CallSpecialistTool: KV update failed for dialog_key=%s", dialog_key)
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error="pending specialist state transition failed",
                data={
                    "specialist": specialist_id,
                    "status": "failed",
                    "reason": "PENDING_STATE_WRITE_FAILED",
                    "state_transition": "failed",
                },
            )
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"state_transition": transition},
        )

    async def _sync_active_bitrix_draft(self, task: AgentTask, *, dialog_key: str | None) -> ToolResult | None:
        """Keep the one active write draft addressable from a base Bitrix chat."""
        if self._store is None or not dialog_key:
            return None
        base_key = str(task.context.get("base_dialog_key") or "").strip()
        number = task.context.get("conversation_number")
        if not base_key or number in (None, ""):
            return None
        try:
            draft = await self.get_active_bitrix_draft(dialog_key)
            current = str(await self._store.get_kv(base_key, _ACTIVE_DRAFT_BRANCH_FIELD) or "")
            if draft:
                await self._store.set_kv(base_key, _ACTIVE_DRAFT_BRANCH_FIELD, dialog_key)
                await self._store.set_kv(base_key, _ACTIVE_DRAFT_NUMBER_FIELD, str(number))
            elif current == dialog_key:
                await self._store.delete_kv(base_key, _ACTIVE_DRAFT_BRANCH_FIELD)
                await self._store.delete_kv(base_key, _ACTIVE_DRAFT_NUMBER_FIELD)
        except Exception:
            logger.exception("CallSpecialistTool: active Bitrix draft reference update failed")
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error="active Bitrix draft reference transition failed",
            )
        return None

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        specialist_id = str(args.get("specialist_id") or "").strip()
        request = str(args.get("request") or "").strip()

        specialist = self._specialists.get(specialist_id)
        if specialist is None:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=f"Специалист '{specialist_id}' не найден",
            )

        context: dict[str, Any] = {}
        if dialog_key:
            context["dialog_key"] = dialog_key
        if dialog_id:
            context["dialog_id"] = dialog_id

        sub_task = AgentTask(
            task_id="",
            request=request,
            user=UserContext(id=str(user_id) if user_id is not None else ""),
            context=context,
        )
        try:
            sr = await specialist.handle(sub_task)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.exception("CallSpecialistTool: specialist %s failed", specialist_id)
            return ToolResult(status=ToolStatus.ERROR, tool=self.name, error=err)

        if self.schedule_fn and sr.scheduled_tasks:
            try:
                self.schedule_fn(sr.scheduled_tasks)
            except Exception:
                logger.exception("CallSpecialistTool: schedule_fn failed for %s", specialist_id)

        state_result = await self._persist_pending_state(
            dialog_key=dialog_key,
            specialist_id=specialist_id,
            specialist_status=str(sr.status),
        )
        if state_result is not None and state_result.status == ToolStatus.ERROR:
            return state_result

        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "specialist": specialist_id,
                "answer": sr.answer,
                "status": sr.status,
                "actions_requiring_approval": [a.model_dump() for a in sr.actions_requiring_approval],
                "model_usage": [usage.model_dump() for usage in sr.model_usage],
                "metadata": sr.metadata,
                **_terminal_contract_data(sr.metadata),
            },
        )

    async def execute_with_task(self, args: dict[str, Any], *, task: AgentTask) -> ToolResult:
        specialist_id = str(args.get("specialist_id") or "").strip()
        planned_request = str(args.get("request") or "").strip()
        structured_command = args.get("structured_command")
        original_request = str(task.context.get("t0006_original_request") or "").strip()
        dispatch_request = str(task.context.get("t0007_dispatch_request") or "").strip()
        explicit_segment_request = str(task.context.get("t0007_explicit_segment_request") or "").strip()
        # The orchestrator, rather than a specialist, owns decomposition.  A
        # non-empty dispatch request is therefore the only request the
        # specialist may execute for this branch; the original is audit-only.
        request = dispatch_request or explicit_segment_request or original_request or planned_request

        specialist = self._specialists.get(specialist_id)
        if specialist is None:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=f"unknown specialist: {specialist_id}",
            )

        dialog_key = str(task.context.get("dialog_key") or "") or None
        sub_task = task.model_copy(
            update={
                "request": request,
                "context": {
                    **task.context,
                    "t0006_effective_specialist_request": request,
                    "t0006_planned_subtask_request": str(
                        task.context.get("t0006_planned_subtask_request") or planned_request
                    ),
                },
            }
        )
        try:
            structured_executor = getattr(specialist, "execute_structured_command", None)
            if isinstance(structured_command, dict):
                if not callable(structured_executor):
                    return ToolResult(
                        status=ToolStatus.INVALID_TOOL_CALL,
                        tool=self.name,
                        error=f"specialist {specialist_id} does not accept structured commands",
                    )
                sr = await structured_executor(sub_task, dict(structured_command))
            else:
                sr = await specialist.handle(sub_task)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            logger.exception("CallSpecialistTool: specialist %s failed", specialist_id)
            return ToolResult(status=ToolStatus.ERROR, tool=self.name, error=err)

        if self.schedule_fn and sr.scheduled_tasks:
            try:
                self.schedule_fn(sr.scheduled_tasks)
            except Exception:
                logger.exception("CallSpecialistTool: schedule_fn failed for %s", specialist_id)

        state_result = await self._persist_pending_state(
            dialog_key=dialog_key,
            specialist_id=specialist_id,
            specialist_status=str(sr.status),
        )
        if state_result is not None and state_result.status == ToolStatus.ERROR:
            return state_result

        if specialist_id == "bitrix24":
            draft_result = await self._sync_active_bitrix_draft(task, dialog_key=dialog_key)
            if draft_result is not None:
                return draft_result

        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={
                "specialist": specialist_id,
                "answer": sr.answer,
                "status": sr.status,
                "actions_requiring_approval": [a.model_dump() for a in sr.actions_requiring_approval],
                "model_usage": [usage.model_dump() for usage in sr.model_usage],
                "metadata": sr.metadata,
                **_terminal_contract_data(sr.metadata),
            },
        )


def _terminal_contract_data(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata.get("terminal"):
        return {}
    return {
        "terminal": bool(metadata.get("terminal")),
        "answer_is_final": bool(metadata.get("answer_is_final")),
        "safe_to_send": bool(metadata.get("safe_to_send")),
        "fast_return": bool(metadata.get("fast_return")),
        "fast_return_reason": str(metadata.get("fast_return_reason") or ""),
        "terminal_tool": str(metadata.get("terminal_tool") or ""),
    }
