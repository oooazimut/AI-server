from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixClient, BitrixConfigError
from ai_server.models import ToolDefinition, ToolResult
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy


class BitrixToolset:
    def __init__(self, client: BitrixClient | None = None) -> None:
        self.client = client or BitrixClient()

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="bitrix_api",
                description="Controlled Bitrix REST API access. Read methods can run immediately; writes return approval requests.",
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
            ),
            ToolDefinition(
                name="portal_search",
                description="Search local Bitrix portal index. The current MVP exposes the contract; index backend comes next.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "scope": {"type": "string", "enum": ["all", "documents", "files", "tasks", "projects"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                    },
                    "required": ["query"],
                },
            ),
        ]

    async def bitrix_api(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "call").strip().lower()
        method = str(args.get("method") or "").strip()
        params = args.get("params") if isinstance(args.get("params"), dict) else {}
        summary = str(args.get("summary") or method).strip()

        if action in {"confirm_pending", "cancel_pending"}:
            return ToolResult(
                status="not_implemented",
                tool="bitrix_api",
                data={"action": action, "message": "Pending-action store will be added with dialog state."},
            )
        if action != "call":
            return ToolResult(status="invalid_tool_call", tool="bitrix_api", error=f"unknown action: {action}")

        decision = decide_bitrix_method_policy(method)
        if decision.decision == "deny":
            return ToolResult(
                status="denied",
                tool="bitrix_api",
                data={"method": method, "policy_reason": decision.reason},
            )
        if decision.decision == "confirm":
            return ToolResult(
                status="confirmation_required",
                tool="bitrix_api",
                data={"method": method, "params": params, "summary": summary, "policy_reason": decision.reason},
            )

        try:
            result = await self.client.result(method, params)
        except (BitrixApiError, BitrixConfigError) as exc:
            return ToolResult(
                status="not_configured" if isinstance(exc, BitrixConfigError) else "error",
                tool="bitrix_api",
                error=str(exc),
                data={"method": method, "params": params},
            )
        return ToolResult(status="ok", tool="bitrix_api", data={"method": method, "params": params, "result": result})

    def portal_search_contract(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        scope = str(args.get("scope") or "all").strip().lower()
        limit = max(1, min(int(args.get("limit") or 10), 30))
        return ToolResult(
            status="not_connected",
            tool="portal_search",
            data={
                "query": query,
                "scope": scope,
                "limit": limit,
                "message": "Local portal search index will be connected in the Bitrix specialist integration step.",
            },
        )
