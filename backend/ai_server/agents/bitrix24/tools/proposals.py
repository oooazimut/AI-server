from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ai_server.agents.bitrix24.ports import ProposalStorePort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.utils import MOSCOW_TZ, optional_int


@dataclass(frozen=True)
class IncompleteProposal:
    task_id: int
    missing_parts: str
    task_title: str = ""
    responsible_id: int | None = None
    responsible_dialog_id: str = ""


def _next_morning_830() -> datetime:
    now = datetime.now(MOSCOW_TZ)
    target = now.replace(hour=8, minute=30, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return target


def proposal_context(
    store: ProposalStorePort | None,
    user_id: int | None,
    manager_id: int | None,
) -> dict[str, Any]:
    if store is None or user_id is None:
        return {}
    if manager_id and user_id == manager_id:
        proposed = [p for p in store.get_proposals_for_manager() if p.get("status") == "proposed"]
        return {"pending_manager_proposals": proposed} if proposed else {}
    pending = store.get_pending_for_responsible(user_id)
    return {"pending_responsible_question": pending} if pending else {}


class SaveIncompleteProposalTool:
    name = "save_incomplete_proposal"

    def __init__(self, store: ProposalStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="save_incomplete_proposal",
            description=(
                "Save partial completion data for a task to the agent's internal DB "
                "and schedule a morning proposal to the manager at 08:30 МСК. "
                "Call this after approving a partially-complete task when some items from the "
                "task description were not covered by the result."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer"},
                    "task_title": {"type": "string"},
                    "missing_parts": {
                        "type": "string",
                        "description": "What was not done — from task description",
                    },
                    "responsible_id": {"type": "integer"},
                    "responsible_dialog_id": {"type": "string"},
                },
                "required": ["task_id", "missing_parts"],
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
        if self._store is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="proposal store is not configured",
            )
        task_id = optional_int(args.get("task_id"))
        if task_id is None:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="task_id is required",
            )
        missing_parts = str(args.get("missing_parts") or "").strip()
        if not missing_parts:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
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
        proposal_id = self._store.save_proposal(
            task_id=proposal.task_id,
            task_title=proposal.task_title,
            missing_parts=proposal.missing_parts,
            responsible_id=proposal.responsible_id,
            responsible_dialog_id=proposal.responsible_dialog_id,
            scheduled_for=run_date.isoformat(),
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"proposal_id": proposal_id, "scheduled_for": run_date.isoformat()},
        )


class DeleteIncompleteProposalTool:
    name = "delete_incomplete_proposal"

    def __init__(self, store: ProposalStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="delete_incomplete_proposal",
            description=(
                "Delete a saved incomplete proposal from the agent's internal DB "
                "after the manager agreed to create a new task from it."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "integer"},
                },
                "required": ["proposal_id"],
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
        if self._store is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="proposal store is not configured",
            )
        proposal_id = optional_int(args.get("proposal_id"))
        if proposal_id is None:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="proposal_id is required",
            )
        self._store.delete_proposal(proposal_id)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={"proposal_id": proposal_id})


class SaveResponsibleResponseTool:
    name = "save_responsible_response"

    def __init__(self, store: ProposalStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="save_responsible_response",
            description=(
                "Save the responsible person's explanation for a pending incomplete proposal. "
                "Call this when the responsible replies to a pending_responsible_question "
                "to store their answer so it appears in the morning proposal to the manager."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "integer"},
                    "response_text": {
                        "type": "string",
                        "description": "Responsible's explanation text",
                    },
                },
                "required": ["proposal_id", "response_text"],
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
        if self._store is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="proposal store is not configured",
            )
        proposal_id = optional_int(args.get("proposal_id"))
        if proposal_id is None:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="proposal_id is required",
            )
        response_text = str(args.get("response_text") or "").strip()
        if not response_text:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="response_text is required and must not be empty",
            )
        self._store.update_responsible_response(proposal_id, response_text)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={"proposal_id": proposal_id})


__all__ = [
    "IncompleteProposal",
    "SaveIncompleteProposalTool",
    "DeleteIncompleteProposalTool",
    "SaveResponsibleResponseTool",
    "proposal_context",
]
