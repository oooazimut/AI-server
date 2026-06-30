from __future__ import annotations

from typing import Any

from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.vehicle_usage import VehicleUsageStorePort, _request_date
from ai_server.utils import optional_int


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
        request_date = _request_date(args.get("request_date") or parsed.get("date"))
        request_id = self._store.save_draft(
            request_date=request_date,
            user_id=user_id,
            dialog_id=dialog_id or "",
            response_text=str(args.get("source_text") or ""),
            parsed=parsed,
            status="answered",
        )
        employees_by_name = {
            str(row["full_name"]).casefold(): int(row["display_order"]) for row in self._store.staff_roster()
        }
        employee_statuses: list[tuple[int, str, str]] = []
        for entry in _staff_entries(parsed):
            employee_id = optional_int(entry.get("staff_order")) or employees_by_name.get(
                str(entry.get("full_name") or "").casefold()
            )
            if employee_id is None:
                continue
            employee_statuses.append(
                (employee_id, str(entry.get("status") or "unknown"), str(entry.get("notes") or ""))
            )
        vehicle_assignments: list[tuple[int, int | None, str]] = []
        for entry in _vehicle_entries(parsed):
            vehicle_id = optional_int(entry.get("vehicle_id"))
            if vehicle_id is None:
                continue
            resolved_employee_id = optional_int(entry.get("employee_id")) or employees_by_name.get(
                str(entry.get("employee_name") or "").casefold()
            )
            vehicle_assignments.append((vehicle_id, resolved_employee_id, str(entry.get("notes") or "")))
        self._store.replace_day_report(
            status_date=request_date,
            employee_statuses=employee_statuses,
            vehicle_assignments=vehicle_assignments,
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_save_report",
            data={
                "request_id": request_id,
                "staff_entries_saved": len(employee_statuses),
                "vehicle_assignments_saved": len(vehicle_assignments),
            },
        )


def _staff_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    people = parsed.get("people")
    return [item for item in people if isinstance(item, dict)] if isinstance(people, list) else []


def _vehicle_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    vehicles = parsed.get("vehicles")
    return [item for item in vehicles if isinstance(item, dict)] if isinstance(vehicles, list) else []
