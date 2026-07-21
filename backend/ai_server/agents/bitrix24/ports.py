from __future__ import annotations

from typing import Any, Protocol


class TaskDraftStorePort(Protocol):
    """Port for persisting a pending Bitrix write draft keyed by dialog_key."""

    async def save_task_draft(self, dialog_key: str, params: dict[str, Any]) -> None: ...

    async def get_task_draft(self, dialog_key: str, *, ttl_minutes: int | None = None) -> dict[str, Any] | None: ...

    async def claim_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
    ) -> dict[str, Any] | None: ...

    async def claim_expired_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
    ) -> dict[str, Any] | None: ...

    async def reclaim_stale_finalizing_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None: ...

    async def renew_task_draft_claim(
        self,
        dialog_key: str,
        *,
        draft_id: str,
        claim_token: str,
        expected_status: str,
    ) -> bool: ...

    async def resolve_stale_confirming_task_draft(
        self,
        dialog_key: str,
        *,
        expected_draft_id: str,
        expected_version: int,
        expected_type: str,
        lease_seconds: int = 300,
    ) -> dict[str, Any] | None: ...

    async def release_task_draft(self, dialog_key: str, *, draft_id: str, claim_token: str = "") -> None: ...

    async def finalize_task_draft_claim(
        self,
        dialog_key: str,
        *,
        draft_id: str,
        params: dict[str, Any],
        claim_token: str = "",
    ) -> dict[str, Any] | None: ...

    async def get_task_draft_for_finalizer(self, dialog_key: str) -> dict[str, Any] | None: ...

    async def get_claimed_task_draft(
        self,
        dialog_key: str,
        *,
        expected_type: str,
    ) -> dict[str, Any] | None: ...

    async def delete_task_draft(
        self,
        dialog_key: str,
        *,
        status: str = "cancelled",
        expected_draft_id: str = "",
        expected_version: int | None = None,
        expected_claim_token: str = "",
    ) -> None: ...


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
