from __future__ import annotations

import os
from typing import Any

import httpx

from ai_server.models import ToolDefinition, ToolResult
from ai_server.tools.bitrix_policy import decide_bitrix_method_policy


class BitrixApiError(RuntimeError):
    def __init__(self, method: str, error: str, description: str = "") -> None:
        self.method = method
        self.error = error
        self.description = description
        super().__init__(f"Bitrix REST error in {method}: {error} {description}".strip())


class BitrixClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("BITRIX_REST_WEBHOOK_URL", "")).rstrip("/") + "/"
        self.timeout = httpx.Timeout(30.0)

    @property
    def configured(self) -> bool:
        return bool(self.base_url.strip("/"))

    async def call(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.configured:
            raise BitrixApiError(method, "NOT_CONFIGURED", "BITRIX_REST_WEBHOOK_URL is empty")
        url = f"{self.base_url}{method}.json"
        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            response = await client.post(url, json=dict(payload or {}))
            response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise BitrixApiError(method, str(data.get("error", "")), str(data.get("error_description", "")))
        return data

    async def result(self, method: str, payload: dict[str, Any] | None = None) -> Any:
        data = await self.call(method, payload)
        return data.get("result")


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
        except BitrixApiError as exc:
            return ToolResult(
                status="not_configured" if exc.error == "NOT_CONFIGURED" else "error",
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
