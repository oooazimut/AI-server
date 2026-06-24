"""Narrow Protocol interfaces for BitrixClient (ISP).

BitrixClient implements all protocols structurally (duck typing — no explicit
declaration needed). Consumers should declare the narrowest port they actually
need so that mock objects in tests stay minimal and responsibilities are clear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class BitrixTaskPort(Protocol):
    """Task lifecycle operations only."""

    async def list_all_tasks(
        self,
        *,
        filter_: dict[str, Any] | None = None,
        select: list[str] | None = None,
        order: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def get_task(self, task_id: int, *, select: list[str] | None = None) -> Any: ...

    async def list_task_results(self, task_id: int) -> Any: ...

    async def add_task_result(self, task_id: int, text: str) -> Any: ...

    async def disapprove_task(self, task_id: int) -> Any: ...

    async def approve_task(self, task_id: int) -> Any: ...

    async def complete_task(self, task_id: int) -> Any: ...

    async def renew_task(self, task_id: int) -> Any: ...

    async def require_task_result(self, task_id: int) -> Any: ...

    async def add_task_comment(self, *, task_id: int, message: str, author_id: int | None = None) -> Any: ...


class BitrixUserPort(Protocol):
    """User and employee lookup / notification."""

    async def list_all_users(
        self,
        *,
        filter_: dict[str, Any] | None = None,
        select: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def search_users(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]: ...

    async def get_user(self, user_id: int) -> dict[str, Any] | None: ...

    async def notify_user(self, *, user_id: int, message: str, tag: str = "ai_server", sub_tag: str = "") -> Any: ...


class BitrixBotPort(Protocol):
    """Bot messaging and chat management."""

    async def send_bot_message(
        self,
        dialog_id: str,
        message: str,
        *,
        bot_id: int | None = None,
        keyboard: object | None = None,
    ) -> Any: ...

    async def get_bot_file_download_url(self, file_id: int, *, bot_id: int | None = None) -> str: ...

    async def get_chat_file_download_url(self, file_id: int, *, dialog_id: str) -> str: ...

    async def create_bot_chat(
        self,
        *,
        title: str,
        user_ids: list[int],
        description: str = "",
        color: str = "mint",
        message: str = "",
        bot_id: int | None = None,
        owner_id: int | None = None,
    ) -> Any: ...


class BitrixSupervisorPort(BitrixTaskPort, BitrixUserPort, Protocol):
    """Combined port for the task supervisor (needs tasks + user lookup/notify)."""


class BitrixDiskPort(Protocol):
    """File and storage access."""

    async def get_disk_file(self, file_id: int) -> Any: ...

    async def get_disk_file_download_url(self, file_id: int) -> str: ...

    async def list_disk_storages(self, *, limit: int | None = None) -> list[dict[str, Any]]: ...

    async def list_disk_folder_children_all(
        self,
        *,
        folder_id: int,
        filter_: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def download_file_from_url(self, url: str, destination: Path, *, max_bytes: int) -> int: ...


class BitrixRestPort(Protocol):
    """Raw Bitrix REST call — used by BitrixToolset for the bitrix_api tool."""

    async def result(self, method: str, params: dict[str, Any]) -> Any: ...


class BitrixToolClientPort(BitrixRestPort, BitrixUserPort, Protocol):
    """Full client for BitrixToolset: REST call + user/project lookup."""

    async def search_projects(self, query: str, *, limit: int = 10) -> list[dict[str, Any]]: ...


class BitrixFileDownloadPort(Protocol):
    """Resolving download URLs and downloading portal files — used by DocumentToolset."""

    async def get_attached_object(self, attached_object_id: int) -> Any: ...

    async def get_disk_file_download_url(self, file_id: int) -> str: ...

    async def download_file_from_url(self, url: str, destination: Path, *, max_bytes: int) -> int: ...


class BitrixWritePort(Protocol):
    """Minimal write interface for executing Bitrix write operations."""

    async def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class BitrixAgentStorePort(Protocol):
    """Persistent store for Bitrix24 specialist proposal state."""

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

    def get_proposal_by_id(self, proposal_id: int) -> dict[str, Any] | None: ...
    def get_proposals_for_manager(self) -> list[dict[str, Any]]: ...
    def get_pending_for_responsible(self, responsible_id: int) -> dict[str, Any] | None: ...
    def update_responsible_response(self, proposal_id: int, response_text: str) -> None: ...
    def mark_status(self, proposal_id: int, status: str) -> None: ...
    def delete_proposal(self, proposal_id: int) -> None: ...
