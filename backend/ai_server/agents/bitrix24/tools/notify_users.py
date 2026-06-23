from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.ports import BitrixToolClientPort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus


class NotifyUsersTool:
    name = "bitrix_notify_users"

    def __init__(self, client: BitrixToolClientPort | None = None) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bitrix_notify_users",
            description="Отправить уведомления (личные сообщения) одному или нескольким пользователям Bitrix24.",
            parameters={
                "type": "object",
                "properties": {
                    "user_ids": {"type": "array", "items": {"type": "integer"}, "description": "ID пользователей"},
                    "message": {"type": "string", "description": "Текст уведомления"},
                    "tag": {"type": "string", "description": "Тег уведомления (по умолчанию: ai_server)"},
                },
                "required": ["user_ids", "message"],
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
        user_ids_raw = args.get("user_ids") or []
        user_ids = [int(uid) for uid in user_ids_raw if uid] if isinstance(user_ids_raw, list) else []
        message = str(args.get("message") or "").strip()
        tag = str(args.get("tag") or "ai_server").strip()
        if not user_ids or not message:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="bitrix_notify_users",
                error="user_ids and message are required",
            )
        if self._client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="bitrix_notify_users",
                error="BitrixClient is not injected",
            )
        errors = []
        for uid in user_ids:
            try:
                await self._client.notify_user(user_id=uid, message=message, tag=tag)
            except (BitrixApiError, BitrixConfigError) as exc:
                errors.append({"user_id": uid, "error": str(exc)})
        if errors:
            return ToolResult(
                status=ToolStatus.ERROR,
                tool="bitrix_notify_users",
                error="Some notifications failed",
                data={"errors": errors},
            )
        return ToolResult(status=ToolStatus.OK, tool="bitrix_notify_users", data={"user_ids": user_ids})
