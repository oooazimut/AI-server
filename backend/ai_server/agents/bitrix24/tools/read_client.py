from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthError, BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.models import ToolResult, ToolStatus
from ai_server.tools.bitrix_ports import BitrixToolClientPort


async def resolve_current_user_read_client(
    tool_name: str,
    *,
    fallback_client: BitrixToolClientPort | None,
    bitrix_oauth: BitrixOAuthService | None,
    user_id: int | None,
) -> tuple[BitrixToolClientPort, str, ToolResult | None]:
    if bitrix_oauth is None:
        if fallback_client is None:
            return (
                _MissingBitrixClient(),
                "none",
                ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=tool_name, error="BitrixClient is not injected"),
            )
        return fallback_client, "configured_client", None

    if user_id is None:
        return (
            _MissingBitrixClient(),
            "none",
            ToolResult(
                status=ToolStatus.DENIED,
                tool=tool_name,
                error="Bitrix read denied: current Bitrix user_id is missing.",
            ),
        )

    try:
        return await bitrix_oauth.client_for_user(user_id), "oauth_current_user", None
    except BitrixOAuthTokenMissing as exc:
        return (
            _MissingBitrixClient(),
            "none",
            ToolResult(
                status=ToolStatus.DENIED,
                tool=tool_name,
                error=f"Bitrix read denied: OAuth token for user {exc.user_id} is missing.",
            ),
        )
    except (BitrixOAuthError, BitrixConfigError) as exc:
        return (
            _MissingBitrixClient(),
            "none",
            ToolResult(status=ToolStatus.ERROR, tool=tool_name, error=f"Bitrix OAuth read client failed: {exc}"),
        )


class _MissingBitrixClient:
    async def result(self, method: str, params: dict[str, Any]) -> Any:
        raise BitrixConfigError("BitrixClient is not injected")
