from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.bitrix_policy import apply_write_policy, decide_bitrix_method_policy
from ai_server.tools.bitrix_ports import BitrixToolClientPort, BitrixWritePort


class BitrixApiTool:
    name = "bitrix_api"

    def __init__(
        self,
        client: BitrixToolClientPort | None = None,
        *,
        write_client: BitrixWritePort | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
        dry_run: bool = False,
    ) -> None:
        self._client = client
        self._write_client = write_client
        self._bitrix_oauth = bitrix_oauth
        self._dry_run = dry_run

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bitrix_api",
            description=(
                "Bitrix24 REST API access. Read methods (ending in .get/.list/.search) execute immediately. "
                "Write methods execute after explicit user confirmation in the conversation. "
                "Dangerous methods (user management, bots) are denied."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "params": {"type": "object"},
                    "summary": {"type": "string"},
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
        summary = str(args.get("summary") or method).strip()
        return await self._call_api(method, params, summary, user_id=user_id)

    async def _call_api(self, method: str, params: dict[str, Any], summary: str, *, user_id: int | None) -> ToolResult:
        if self._client is None and self._write_client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool="bitrix_api", error="BitrixClient is not injected")
        decision = decide_bitrix_method_policy(method)
        if decision.decision == "deny":
            return ToolResult(
                status=ToolStatus.DENIED,
                tool="bitrix_api",
                data={"method": method, "policy_reason": decision.reason},
            )
        if decision.decision == "confirm":
            return await self._execute_write(method, params, summary, user_id=user_id)
        if self._client is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool="bitrix_api", error="BitrixClient is not injected")
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

    async def _execute_write(
        self, method: str, params: dict[str, Any], summary: str, *, user_id: int | None
    ) -> ToolResult:
        if not params:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="bitrix_api",
                error="Write methods require non-empty params.",
                data={"method": method},
            )
        if self._dry_run:
            return ToolResult(
                status=ToolStatus.DRY_RUN,
                tool="bitrix_api",
                data={"method": method, "summary": summary, "dry_run": True},
            )
        params = apply_write_policy(method, params)

        # Attempt OAuth per-user write
        if self._bitrix_oauth is not None and user_id is not None:
            try:
                oauth_client = await self._bitrix_oauth.client_for_user(user_id)
                raw = await oauth_client.call(method, params)
                result = raw.get("result") if isinstance(raw, dict) else raw
                return ToolResult(
                    status=ToolStatus.OK,
                    tool="bitrix_api",
                    data={"method": method, "params": params, "result": result},
                )
            except (BitrixApiError, BitrixConfigError) as exc:
                return ToolResult(
                    status=ToolStatus.NOT_CONFIGURED if isinstance(exc, BitrixConfigError) else ToolStatus.ERROR,
                    tool="bitrix_api",
                    error=str(exc),
                    data={"method": method, "params": params},
                )
            except Exception:
                pass  # OAuth unavailable — fall through to write_client

        # Fallback: dedicated write client (BitrixWritePort)
        write_cl = self._write_client
        if write_cl is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="bitrix_api",
                error="No write client configured for Bitrix write operations.",
                data={"method": method},
            )
        try:
            raw = await write_cl.call(method, params)
            result = raw.get("result") if isinstance(raw, dict) else raw
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
