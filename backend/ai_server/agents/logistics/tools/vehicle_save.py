from __future__ import annotations

from typing import Any

from ai_server.integrations.ports import VehicleUsageStorePort
from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.vehicle_usage import VehicleReportProcessor, _request_date


class VehicleSaveDraftTool:
    name = "vehicle_usage_save_draft"

    def __init__(self, store: VehicleUsageStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_save_draft",
            description="Save the logistics LLM parsed draft; does not finalize daily report.",
            parameters={
                "type": "object",
                "properties": {
                    "request_date": {"type": "string"},
                    "response_text": {"type": "string"},
                    "parsed": {"type": "object"},
                    "status": {"type": "string"},
                },
                "required": ["parsed"],
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
                tool="vehicle_usage_save_draft",
                error="VehicleUsageStore is not configured",
            )
        parsed = args.get("parsed")
        if not isinstance(parsed, dict):
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="vehicle_usage_save_draft",
                error="parsed object is required",
            )
        request_id = self._store.save_draft(
            request_date=_request_date(args.get("request_date") or parsed.get("date")),
            user_id=user_id,
            dialog_id=dialog_id or "",
            response_text=str(args.get("response_text") or ""),
            parsed=parsed,
            status=str(args.get("status") or "pending_confirmation"),
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_save_draft",
            data={"request_id": request_id},
        )


class VehicleSaveReportTool:
    name = "vehicle_usage_save_report"

    def __init__(self, store: VehicleUsageStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_save_report",
            description="Finalize confirmed daily vehicle/staff report using the JSON chosen by Logistics LLM.",
            parameters={
                "type": "object",
                "properties": {
                    "request_date": {"type": "string"},
                    "source_text": {"type": "string"},
                    "parsed": {"type": "object"},
                },
                "required": ["parsed"],
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
                tool="vehicle_usage_save_report",
                error="VehicleUsageStore is not configured",
            )
        parsed = args.get("parsed")
        if not isinstance(parsed, dict):
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="vehicle_usage_save_report",
                error="parsed object is required",
            )
        saved = VehicleReportProcessor(self._store).save_report(
            request_date=_request_date(args.get("request_date") or parsed.get("date")),
            user_id=user_id,
            dialog_id=dialog_id or "",
            source_text=str(args.get("source_text") or ""),
            parsed=parsed,
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_save_report",
            data=saved,
        )
