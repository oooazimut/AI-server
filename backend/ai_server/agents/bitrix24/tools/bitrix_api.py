from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.bitrix_policy import decide_bitrix_method_policy
from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.dialog_state import BitrixPendingActionService, PendingBitrixAction
from ai_server.integrations.bitrix.ports import BitrixRestPort, BitrixToolClientPort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus


class BitrixApiTool:
    name = "bitrix_api"

    def __init__(
        self,
        client: BitrixToolClientPort | None = None,
        *,
        pending_actions: BitrixPendingActionService | None = None,
        actor_client: BitrixRestPort | None = None,
        auto_execute: bool = False,
    ) -> None:
        self._client = client
        self._pending_actions = pending_actions
        self._actor_client = actor_client
        self._auto_execute = auto_execute

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bitrix_api",
            description=(
                "Bitrix24 REST API access. Read methods (ending in .get/.list/.search) execute immediately. "
                "Write methods require user confirmation. Dangerous methods (user management, bots) are denied."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["call", "confirm_pending", "cancel_pending"]},
                    "method": {"type": "string"},
                    "params": {"type": "object"},
                    "summary": {"type": "string"},
                },
                "required": ["action", "method", "params"],
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
        action = str(args.get("action") or "call").strip().lower()
        method = str(args.get("method") or "").strip()
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        summary = str(args.get("summary") or method).strip()

        if action in {"confirm_pending", "cancel_pending"}:
            return await self._handle_pending_action(action, dialog_key=dialog_key, user_id=user_id)
        if action != "call":
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool="bitrix_api", error=f"unknown action: {action}")
        return await self._call_api(method, params, summary, dialog_key=dialog_key, user_id=user_id)

    async def _handle_pending_action(self, action: str, *, dialog_key: str | None, user_id: int | None) -> ToolResult:
        if not self._pending_actions or not dialog_key:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="bitrix_api",
                data={"action": action, "message": "Pending-action store is not bound to this tool call."},
            )
        if action == "cancel_pending":
            result = self._pending_actions.cancel(dialog_key)
        else:
            result = await self._pending_actions.confirm(dialog_key, user_id=user_id)
        return ToolResult(
            status=result.status,
            tool="bitrix_api",
            data={"action": action, "message": result.message, **result.data},
        )

    async def _call_api(
        self, method: str, params: dict[str, Any], summary: str, *, dialog_key: str | None, user_id: int | None
    ) -> ToolResult:
        if self._client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool="bitrix_api", error="BitrixClient is not injected")
        decision = decide_bitrix_method_policy(method)
        if decision.decision == "deny":
            return ToolResult(
                status=ToolStatus.DENIED,
                tool="bitrix_api",
                data={"method": method, "policy_reason": decision.reason},
            )
        if decision.decision == "confirm":
            return await self._confirm_or_queue(
                method, params, summary, policy_reason=decision.reason, dialog_key=dialog_key, user_id=user_id
            )
        try:
            result = await self._client.result(method, params)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                tool="bitrix_api",
                error=str(exc),
                data={"method": method, "params": params},
            )
        return ToolResult(
            status=ToolStatus.OK, tool="bitrix_api", data={"method": method, "params": params, "result": result}
        )

    async def _confirm_or_queue(
        self,
        method: str,
        params: dict[str, Any],
        summary: str,
        *,
        policy_reason: str,
        dialog_key: str | None,
        user_id: int | None,
    ) -> ToolResult:
        if self._auto_execute:
            execute_client = self._actor_client or self._client
            try:
                result = await execute_client.result(method, params)
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                    tool="bitrix_api",
                    error=str(exc),
                    data={"method": method, "params": params},
                )
            return ToolResult(
                status=ToolStatus.OK, tool="bitrix_api", data={"method": method, "params": params, "result": result}
            )
        if not params:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="bitrix_api",
                error="Bitrix write methods require real params before confirmation.",
                data={"method": method},
            )
        if self._pending_actions and dialog_key:
            self._pending_actions.save_pending(
                dialog_key,
                PendingBitrixAction(
                    method=method,
                    params=params,
                    summary=summary,
                    created_by=user_id,
                ),
            )
        return ToolResult(
            status=ToolStatus.CONFIRMATION_REQUIRED,
            tool="bitrix_api",
            data={"method": method, "params": params, "summary": summary, "policy_reason": policy_reason},
        )
