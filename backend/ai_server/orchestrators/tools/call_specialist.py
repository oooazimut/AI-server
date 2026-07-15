from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ai_server.agents.ports import OrchestratorStorePort, SchedulerPort
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

        if self._store and dialog_key:
            try:
                if sr.status in ("needs_clarification", "needs_human"):
                    await self._store.set_kv(dialog_key, "pending_specialist", specialist_id)
                else:
                    await self._store.delete_kv(dialog_key, "pending_specialist")
            except Exception:
                logger.exception("CallSpecialistTool: KV update failed for dialog_key=%s", dialog_key)

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
        request = str(args.get("request") or "").strip()

        specialist = self._specialists.get(specialist_id)
        if specialist is None:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool=self.name,
                error=f"unknown specialist: {specialist_id}",
            )

        dialog_key = str(task.context.get("dialog_key") or "") or None
        sub_task = task.model_copy(update={"request": request})
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

        if self._store and dialog_key:
            try:
                if sr.status in ("needs_clarification", "needs_human"):
                    await self._store.set_kv(dialog_key, "pending_specialist", specialist_id)
                else:
                    await self._store.delete_kv(dialog_key, "pending_specialist")
            except Exception:
                logger.exception("CallSpecialistTool: KV update failed for dialog_key=%s", dialog_key)

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
