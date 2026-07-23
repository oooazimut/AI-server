from __future__ import annotations

import re
from typing import Any

from ai_server.agents.bitrix24.tools.read_client import oauth_authorization_data, oauth_missing_error
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy
from ai_server.tools.bitrix_ports import BitrixToolClientPort

WRITE_METHODS_WITH_DEDICATED_DRAFT_TOOLS = {
    "calendar.event.add": "Use calendar_event_draft/calendar_event_confirm for Bitrix calendar event creation.",
    "sonet_group.create": "Use project_create_draft/project_create_confirm for Bitrix project creation.",
    "tasks.task.add": "Use task_create_draft/task_create_confirm for Bitrix task creation.",
    "tasks.task.result.add": "Use task_close_draft/task_close_confirm for Bitrix task closing results.",
    "tasks.task.complete": "Use task_close_draft/task_close_confirm for Bitrix task closing.",
    "tasks.task.approve": "Use task_close_draft/task_close_confirm for Bitrix task approval/closing.",
}


class BitrixApiTool:
    name = "bitrix_api"

    def __init__(
        self,
        client: BitrixToolClientPort | None = None,
        *,
        bitrix_oauth: BitrixOAuthService | None = None,
    ) -> None:
        self._client = client
        self._bitrix_oauth = bitrix_oauth

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bitrix_api",
            description=(
                "Bitrix24 REST API access. Read methods (ending in .get/.list/.search) execute immediately. "
                "Generic write methods are rejected; every supported write must use a dedicated numbered draft tool. "
                "When OAuth is required, writes execute only as the current Bitrix user. "
                "Dedicated draft workflows must be used for task/project/calendar creation and task closing. "
                "Dangerous methods (user management, bots) are denied."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["method", "params"],
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
        method = str(args.get("method") or "").strip()
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        return await self._call_api(method, params, user_id=user_id)

    async def _call_api(
        self,
        method: str,
        params: dict[str, Any],
        *,
        user_id: int | None,
    ) -> ToolResult:
        if self._client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool="bitrix_api", error="BitrixClient is not injected")
        dedicated_tool_error = WRITE_METHODS_WITH_DEDICATED_DRAFT_TOOLS.get(method.strip().casefold())
        if dedicated_tool_error:
            return ToolResult(
                status=ToolStatus.DENIED,
                tool="bitrix_api",
                error=dedicated_tool_error,
                data={"method": method},
            )
        decision = decide_bitrix_method_policy(method)
        if decision.decision == "deny":
            return ToolResult(
                status=ToolStatus.DENIED,
                tool="bitrix_api",
                data={"method": method, "policy_reason": decision.reason},
            )
        if decision.decision == "confirm":
            return ToolResult(
                status=ToolStatus.DENIED,
                tool="bitrix_api",
                error="Generic Bitrix writes are disabled; use a dedicated structured draft/confirm workflow.",
                data={"method": method, "policy_reason": "DEDICATED_DRAFT_REQUIRED"},
            )
        read_client = self._client
        access_actor = "configured_client"
        if self._bitrix_oauth is not None:
            if user_id is None:
                return ToolResult(
                    status=ToolStatus.DENIED,
                    tool="bitrix_api",
                    error="Bitrix read operation denied: current Bitrix user_id is missing.",
                    data={"method": method, "params": params},
                )
            try:
                read_client = await self._bitrix_oauth.client_for_user(user_id)
                access_actor = "oauth_current_user"
            except BitrixOAuthTokenMissing as exc:
                data = {"method": method, "params": params}
                data.update(oauth_authorization_data(self._bitrix_oauth, user_id=exc.user_id))
                return ToolResult(
                    status=ToolStatus.DENIED,
                    tool="bitrix_api",
                    error=oauth_missing_error(
                        "Bitrix read operation denied",
                        user_id=exc.user_id,
                        authorization=data.get("authorization"),
                    ),
                    data=data,
                )
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                    tool="bitrix_api",
                    error=str(exc),
                    data={"method": method, "params": params},
                )
        if read_client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool="bitrix_api", error="BitrixClient is not injected")
        try:
            result = await read_client.result(method, params)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="bitrix_api",
                error=str(exc),
                data={"method": method, "params": params},
            )
        return ToolResult(
            status=ToolStatus.OK,
            tool="bitrix_api",
            data={"method": method, "params": params, "result": result, "access_actor": access_actor},
        )

def _extract_sonet_groups(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ("groups", "workgroups", "items", "result"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _extract_sonet_groups(value)
                if nested:
                    return nested
    return []


_PROJECT_ALIASES = {
    "almira": "альмера",
    "almera": "альмера",
    "largus": "ларгус",
    "logan": "логан",
}


def _normalize_project_name(value: str) -> str:
    text = value.casefold().replace("ё", "е")
    for source, target in _PROJECT_ALIASES.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    text = re.sub(r"[-–—_/\\]+", " ", text)
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()
