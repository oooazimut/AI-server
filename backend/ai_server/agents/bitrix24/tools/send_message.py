from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.ports import BitrixBotPort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus


class SendMessageTool:
    name = "bitrix_send_message"

    def __init__(self, bot: BitrixBotPort | None = None) -> None:
        self._bot = bot

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bitrix_send_message",
            description="Отправить сообщение от бота в диалог или чат Bitrix24.",
            parameters={
                "type": "object",
                "properties": {
                    "dialog_id": {"type": "string", "description": "ID диалога или чата"},
                    "message": {"type": "string", "description": "Текст сообщения"},
                },
                "required": ["dialog_id", "message"],
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
        target_dialog_id = str(args.get("dialog_id") or "").strip()
        message = str(args.get("message") or "").strip()
        if not target_dialog_id or not message:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="bitrix_send_message",
                error="dialog_id and message are required",
            )
        if self._bot is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="bitrix_send_message",
                error="BitrixBotPort is not injected",
            )
        try:
            result = await self._bot.send_bot_message(target_dialog_id, message)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="bitrix_send_message",
                error=str(exc),
                data={"dialog_id": target_dialog_id},
            )
        return ToolResult(
            status=ToolStatus.OK,
            tool="bitrix_send_message",
            data={"dialog_id": target_dialog_id, "result": result},
        )
