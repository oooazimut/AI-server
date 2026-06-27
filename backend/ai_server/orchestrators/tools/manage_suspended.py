from __future__ import annotations

import json
import logging
from typing import Any

from ai_server.agents.ports import OrchestratorStorePort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus

logger = logging.getLogger(__name__)

_KV_FIELD_SUSPENDED = "suspended_specialists"
_KV_FIELD_PENDING = "pending_specialist"


class ManageSuspendedTool:
    """Manage suspended specialist dialogs.

    Allows the orchestrator LLM to list, resume, or close dialogs that were
    put on hold when the user switched topics mid-conversation.
    """

    name = "manage_suspended"

    def __init__(self, store: OrchestratorStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Управление подвешенными диалогами со специалистами. "
                "list — показать все подвешенные диалоги. "
                "resume — возобновить диалог с указанным специалистом. "
                "close — закрыть и удалить подвешенный диалог."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "resume", "close"],
                        "description": "Действие над подвешенными диалогами",
                    },
                    "specialist_id": {
                        "type": "string",
                        "description": "ID специалиста (для resume и close)",
                    },
                },
                "required": ["action"],
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
        action = str(args.get("action") or "").strip()
        specialist_id = str(args.get("specialist_id") or "").strip()

        if not self._store or not dialog_key:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool=self.name,
                error="Store не настроен или dialog_key не указан",
            )

        if action == "list":
            return await self._list(dialog_key)
        if action == "resume":
            return await self._resume(dialog_key, specialist_id)
        if action == "close":
            return await self._close(dialog_key, specialist_id)

        return ToolResult(
            status=ToolStatus.INVALID_TOOL_CALL,
            tool=self.name,
            error=f"Неизвестное действие: {action!r}",
        )

    async def _list(self, dialog_key: str) -> ToolResult:
        suspended = await self._load_suspended(dialog_key)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"suspended_specialists": suspended},
        )

    async def _resume(self, dialog_key: str, specialist_id: str) -> ToolResult:
        if not specialist_id:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="specialist_id обязателен для resume",
            )
        suspended = await self._load_suspended(dialog_key)
        if specialist_id in suspended:
            suspended.remove(specialist_id)
            await self._save_suspended(dialog_key, suspended)
        await self._store.set_kv(dialog_key, _KV_FIELD_PENDING, specialist_id)  # type: ignore[union-attr]
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"resumed": specialist_id, "remaining_suspended": suspended},
        )

    async def _close(self, dialog_key: str, specialist_id: str) -> ToolResult:
        if not specialist_id:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=self.name,
                error="specialist_id обязателен для close",
            )
        suspended = await self._load_suspended(dialog_key)
        if specialist_id in suspended:
            suspended.remove(specialist_id)
            await self._save_suspended(dialog_key, suspended)
        return ToolResult(
            status=ToolStatus.OK,
            tool=self.name,
            data={"closed": specialist_id, "remaining_suspended": suspended},
        )

    async def _load_suspended(self, dialog_key: str) -> list[str]:
        raw = await self._store.get_kv(dialog_key, _KV_FIELD_SUSPENDED)  # type: ignore[union-attr]
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return [str(x) for x in parsed] if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    async def _save_suspended(self, dialog_key: str, suspended: list[str]) -> None:
        await self._store.set_kv(dialog_key, _KV_FIELD_SUSPENDED, json.dumps(suspended))  # type: ignore[union-attr]
