from __future__ import annotations

from typing import Any

from ai_server.integrations.ports import VehicleUsageStorePort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.vehicle_usage import _request_date


class VehicleContextTool:
    name = "vehicle_usage_context"

    def __init__(self, store: VehicleUsageStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_context",
            description="Read staff roster, known vehicles and latest vehicle usage draft/request.",
            parameters={
                "type": "object",
                "properties": {"request_date": {"type": "string"}},
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
        if self._store is None:
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="vehicle_usage_context",
                error="VehicleUsageStore is not configured",
            )
        request_date = _request_date(args.get("request_date"))
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_context",
            data=self._store.context(
                request_date=request_date,
                user_id=user_id,
                dialog_id=dialog_id or "",
            ),
        )
