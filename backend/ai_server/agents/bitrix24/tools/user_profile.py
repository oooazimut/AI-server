from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.ports import BitrixToolClientPort
from ai_server.integrations.bitrix.profile import compact_user_profile
from ai_server.models import ToolDefinition, ToolResult, ToolStatus


class CurrentUserProfileTool:
    name = "current_user_profile"

    def __init__(self, client: BitrixToolClientPort | None = None) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="current_user_profile",
            description=(
                "Read-only facts about the current Bitrix chat user for LLM permission reasoning: "
                "active flag, admin flags when Bitrix exposes them, department ids, work position, and user type."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
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
        if user_id is None:
            return ToolResult(
                status=ToolStatus.NOT_AVAILABLE,
                tool="current_user_profile",
                error="current Bitrix user id is not available",
            )
        if self._client is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="current_user_profile",
                error="current_user_profile tool is not bound",
            )
        try:
            user = await self._client.get_user(user_id)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="current_user_profile",
                error=str(exc),
                data={"user_id": user_id},
            )
        if user is None:
            return ToolResult(status=ToolStatus.NOT_FOUND, tool="current_user_profile", data={"user_id": user_id})
        return ToolResult(
            status=ToolStatus.OK,
            tool="current_user_profile",
            data={"user_id": user_id, "profile": compact_user_profile(user)},
        )
