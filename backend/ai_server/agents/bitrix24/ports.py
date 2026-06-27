from __future__ import annotations

from typing import Any, Protocol


class TaskDraftStorePort(Protocol):
    """Port for persisting a pending task-creation draft keyed by dialog_key."""

    async def save_task_draft(self, dialog_key: str, params: dict[str, Any]) -> None: ...

    async def get_task_draft(self, dialog_key: str) -> dict[str, Any] | None: ...

    async def delete_task_draft(self, dialog_key: str) -> None: ...


class ProposalStorePort(Protocol):
    """Port for incomplete-proposal persistence and context loading."""

    def save_proposal(
        self,
        *,
        task_id: int,
        task_title: str,
        missing_parts: str,
        responsible_id: int | None,
        responsible_dialog_id: str,
        scheduled_for: str,
    ) -> int: ...

    def delete_proposal(self, proposal_id: int) -> None: ...

    def update_responsible_response(self, proposal_id: int, response_text: str) -> None: ...

    def get_proposals_for_manager(self) -> list[dict[str, Any]]: ...

    def get_pending_for_responsible(self, responsible_id: int) -> dict[str, Any] | None: ...
