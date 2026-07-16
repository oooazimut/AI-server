from __future__ import annotations

import re
from typing import Any

from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.tools.vehicle_usage import (
    SentRequestData,
    VehicleUsageStorePort,
    _now,
    _request_date,
    resolve_vehicle_usage_operator_ids,
)
from ai_server.utils import optional_int

DEFAULT_START_MESSAGE = (
    "Доброе утро. Подготовьте, пожалуйста, отчет по машинам и людям за сегодня: "
    "кто работает, кто выходной/болеет/в отпуске, кто на какой машине, "
    "и какие машины свободны, на ремонте или не работают."
)


class VehicleSetOperatorsTool:
    name = "vehicle_usage_set_operators"

    def __init__(
        self,
        store: VehicleUsageStorePort | None = None,
        *,
        admin_user_ids: frozenset[int] = frozenset(),
    ) -> None:
        self._store = store
        self._admin_user_ids = admin_user_ids

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_set_operators",
            description="Replace the Bitrix user ids allowed to start/fill/cancel vehicle usage reports. Admin only.",
            parameters={
                "type": "object",
                "properties": {
                    "operator_user_ids": {"type": "array", "items": {"type": "integer"}},
                    "reason": {"type": "string"},
                },
                "required": ["operator_user_ids"],
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
                tool="vehicle_usage_set_operators",
                error="VehicleUsageStore is not configured",
            )
        denied = _deny_if_not_admin("vehicle_usage_set_operators", user_id, self._admin_user_ids)
        if denied is not None:
            return denied
        raw_ids = args.get("operator_user_ids")
        if not isinstance(raw_ids, list):
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="vehicle_usage_set_operators",
                error="operator_user_ids list is required",
            )
        operator_ids = sorted({int(item) for item in raw_ids if optional_int(item) is not None})
        if not operator_ids:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="vehicle_usage_set_operators",
                error="at least one operator user id is required",
            )
        setter = getattr(self._store, "set_vehicle_usage_operators", None)
        if not callable(setter):
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="vehicle_usage_set_operators",
                error="VehicleUsageStore does not support operator management",
            )
        saved = setter(operator_user_ids=operator_ids, actor_user_id=user_id)
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_set_operators",
            data={"operator_user_ids": saved},
        )


class VehicleGetOperatorsTool:
    name = "vehicle_usage_get_operators"

    def __init__(self, store: VehicleUsageStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_get_operators",
            description="Read the current Bitrix user ids allowed to start/fill/cancel vehicle usage reports.",
            parameters={"type": "object", "properties": {}},
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
                tool="vehicle_usage_get_operators",
                error="VehicleUsageStore is not configured",
            )
        getter = getattr(self._store, "vehicle_usage_operator_ids", None)
        if not callable(getter):
            return ToolResult(
                status=ToolStatus.NOT_CONFIGURED,
                tool="vehicle_usage_get_operators",
                error="VehicleUsageStore does not support operator management",
            )
        operator_ids = sorted({int(item) for item in getter() if optional_int(item) is not None and int(item) > 0})
        roster = self._store.staff_roster()
        names_by_user_id = {
            int(row["user_id"]): str(row.get("full_name") or "").strip()
            for row in roster
            if optional_int(row.get("user_id")) is not None
        }
        operators = [
            {"user_id": operator_id, "full_name": names_by_user_id.get(operator_id, "")} for operator_id in operator_ids
        ]
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_get_operators",
            data={"operator_user_ids": operator_ids, "operators": operators},
        )


class VehicleStartDayTool:
    name = "vehicle_usage_start_day"

    def __init__(
        self, store: VehicleUsageStorePort | None = None, *, allowed_user_ids: frozenset[int] = frozenset()
    ) -> None:
        self._store = store
        self._allowed_user_ids = allowed_user_ids

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_start_day",
            description="Manually start the daily vehicle/staff report from chat; reminders are based on sent_at.",
            parameters={
                "type": "object",
                "properties": {
                    "request_date": {"type": "string"},
                    "message": {"type": "string"},
                    "reason": {"type": "string"},
                },
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
                tool="vehicle_usage_start_day",
                error="VehicleUsageStore is not configured",
            )
        denied = _deny_if_not_operator("vehicle_usage_start_day", user_id, self._allowed_user_ids, self._store)
        if denied is not None:
            return denied
        request_date = _request_date(args.get("request_date"))
        message = str(args.get("message") or DEFAULT_START_MESSAGE).strip()
        sent_at = _now().isoformat()
        request_id = self._store.create_sent_request(
            SentRequestData(
                request_date=request_date,
                user_id=user_id,
                dialog_id=dialog_id or "",
                message=message,
                sent_at=sent_at,
                reminder_count=0,
                source="manual",
            )
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_start_day",
            data={
                "request_id": request_id,
                "request_date": request_date,
                "message": message,
                "status": "sent",
                "sent_at": sent_at,
            },
        )


class VehicleSaveDraftTool:
    name = "vehicle_usage_save_draft"

    def __init__(
        self, store: VehicleUsageStorePort | None = None, *, allowed_user_ids: frozenset[int] = frozenset()
    ) -> None:
        self._store = store
        self._allowed_user_ids = allowed_user_ids

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
        denied = _deny_if_not_operator("vehicle_usage_save_draft", user_id, self._allowed_user_ids, self._store)
        if denied is not None:
            return denied
        parsed = args.get("parsed")
        if not isinstance(parsed, dict):
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="vehicle_usage_save_draft",
                error="parsed object is required",
            )
        request_date = _request_date(args.get("request_date") or parsed.get("date"))
        response_text = str(args.get("response_text") or "")
        _augment_vehicle_assignments_from_text(
            parsed, response_text, self._store.staff_roster(), self._store.vehicles()
        )
        request_id = self._store.save_draft(
            request_date=request_date,
            user_id=user_id,
            dialog_id=dialog_id or "",
            response_text=response_text,
            parsed=parsed,
            status=str(args.get("status") or "pending_confirmation"),
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_save_draft",
            data={"request_id": request_id, "request_date": request_date},
        )


class VehicleSaveReportTool:
    name = "vehicle_usage_save_report"

    def __init__(
        self, store: VehicleUsageStorePort | None = None, *, allowed_user_ids: frozenset[int] = frozenset()
    ) -> None:
        self._store = store
        self._allowed_user_ids = allowed_user_ids

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
        denied = _deny_if_not_operator("vehicle_usage_save_report", user_id, self._allowed_user_ids, self._store)
        if denied is not None:
            return denied
        parsed = args.get("parsed")
        if not isinstance(parsed, dict):
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="vehicle_usage_save_report",
                error="parsed object is required",
            )
        request_date = _request_date(args.get("request_date") or parsed.get("date"))
        source_text = str(args.get("source_text") or "")
        roster = self._store.staff_roster()
        vehicles = self._store.vehicles()
        _augment_vehicle_assignments_from_text(parsed, source_text, roster, vehicles)
        employees_by_name = _employees_by_name(roster)
        vehicles_by_name = _vehicles_by_name(vehicles)
        employee_statuses_raw: list[tuple[int, str, str]] = []
        employee_requires_vehicle_ids: set[int] = set()
        for entry in _staff_entries(parsed):
            employee_id = optional_int(entry.get("staff_order")) or employees_by_name.get(
                _norm_name(entry.get("full_name") or entry.get("name"))
            )
            if employee_id is None:
                employee_id = _employee_id_by_name(entry.get("full_name") or entry.get("name"), employees_by_name)
            if employee_id is None:
                continue
            if _status_requires_vehicle(entry.get("status")):
                employee_requires_vehicle_ids.add(employee_id)
            employee_statuses_raw.append(
                (employee_id, str(entry.get("status") or "unknown"), str(entry.get("notes") or ""))
            )
        vehicle_assignments: list[tuple[int, int | None, str, str]] = []
        driver_employee_ids: set[int] = set()
        unknown_vehicle_refs: list[str] = []
        for entry in _vehicle_entries(parsed):
            vehicle_ref = entry.get("vehicle_name") or entry.get("name") or entry.get("vehicle")
            vehicle_id = optional_int(entry.get("vehicle_id")) or vehicles_by_name.get(_norm_name(vehicle_ref))
            if vehicle_id is None:
                unknown_ref = str(vehicle_ref or entry.get("vehicle_id") or "").strip()
                if unknown_ref and unknown_ref not in unknown_vehicle_refs:
                    unknown_vehicle_refs.append(unknown_ref)
                continue
            status = str(entry.get("status") or entry.get("assignment_status") or "")
            notes = str(entry.get("notes") or "")
            driver_ids = _driver_ids(entry, employees_by_name)
            if not driver_ids:
                vehicle_assignments.append((vehicle_id, None, status or "unknown", notes))
                continue
            driver_employee_ids.update(driver_ids)
            for driver_id in driver_ids:
                vehicle_assignments.append((vehicle_id, driver_id, status or "in_use", notes))
        employee_statuses_by_id = {
            employee_id: (_status_with_vehicle_assignment(status, employee_id in driver_employee_ids), notes)
            for employee_id, status, notes in employee_statuses_raw
        }
        for employee_id in driver_employee_ids:
            employee_statuses_by_id.setdefault(employee_id, ("worked", ""))
        employee_statuses = [
            (employee_id, status, notes) for employee_id, (status, notes) in sorted(employee_statuses_by_id.items())
        ]
        completeness = _validate_report_completeness(
            roster=roster,
            vehicles=vehicles,
            employee_statuses_by_id=employee_statuses_by_id,
            vehicle_assignments=vehicle_assignments,
            driver_employee_ids=driver_employee_ids,
            employee_requires_vehicle_ids=employee_requires_vehicle_ids,
            unknown_vehicle_refs=unknown_vehicle_refs,
        )
        if completeness["needs_clarification"]:
            draft = dict(parsed)
            draft["validation"] = completeness
            request_id = self._store.save_draft(
                request_date=request_date,
                user_id=user_id,
                dialog_id=dialog_id or "",
                response_text=source_text,
                parsed=draft,
                status="pending_clarification",
            )
            return ToolResult(
                status=ToolStatus.OK,
                tool="vehicle_usage_save_report",
                data={
                    "request_id": request_id,
                    "request_date": request_date,
                    "draft_saved": True,
                    "needs_clarification": True,
                    "missing": completeness["missing"],
                    "unknown": completeness["unknown"],
                    "questions": completeness["questions"],
                    "staff_entries_parsed": len(employee_statuses),
                    "vehicle_assignments_parsed": len(vehicle_assignments),
                },
            )
        request_id = self._store.save_draft(
            request_date=request_date,
            user_id=user_id,
            dialog_id=dialog_id or "",
            response_text=source_text,
            parsed=parsed,
            status="answered",
        )
        self._store.replace_day_report(
            status_date=request_date,
            employee_statuses=employee_statuses,
            vehicle_assignments=vehicle_assignments,
            actor_user_id=user_id,
        )
        vehicles_saved = len({vehicle_id for vehicle_id, _driver_id, _status, _notes in vehicle_assignments})
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_save_report",
            data={
                "request_id": request_id,
                "request_date": request_date,
                "staff_entries_saved": len(employee_statuses),
                "vehicles_saved": vehicles_saved,
                "vehicle_assignments_saved": len(vehicle_assignments),
            },
        )


class VehicleCancelReportTool:
    name = "vehicle_usage_cancel_day"

    def __init__(
        self, store: VehicleUsageStorePort | None = None, *, allowed_user_ids: frozenset[int] = frozenset()
    ) -> None:
        self._store = store
        self._allowed_user_ids = allowed_user_ids

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_cancel_day",
            description="Cancel the daily vehicle/staff report and save the day as day_off/not_required.",
            parameters={
                "type": "object",
                "properties": {
                    "request_date": {"type": "string"},
                    "reason": {"type": "string"},
                },
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
                tool="vehicle_usage_cancel_day",
                error="VehicleUsageStore is not configured",
            )
        denied = _deny_if_not_operator("vehicle_usage_cancel_day", user_id, self._allowed_user_ids, self._store)
        if denied is not None:
            return denied
        request_date = _request_date(args.get("request_date"))
        request_id = self._store.cancel_day_report(
            report_date=request_date,
            user_id=user_id,
            dialog_id=dialog_id or "",
            reason=str(args.get("reason") or "Отчет не требуется: день отмечен как выходной."),
        )
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_cancel_day",
            data={"request_id": request_id, "request_date": request_date, "status": "cancelled_day_off"},
        )


class VehicleGetReportTool:
    name = "vehicle_usage_get_report"

    def __init__(self, store: VehicleUsageStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_get_report",
            description="Read a saved vehicle/staff report by date before showing or editing it.",
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
                tool="vehicle_usage_get_report",
                error="VehicleUsageStore is not configured",
            )
        request_date = _request_date(args.get("request_date"))
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_get_report",
            data=self._store.get_day_report(report_date=request_date),
        )


class VehicleGetEmployeePeriodReportTool:
    name = "vehicle_usage_get_employee_period_report"

    def __init__(self, store: VehicleUsageStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_get_employee_period_report",
            description="Read a period report for one employee by date range.",
            parameters={
                "type": "object",
                "properties": {
                    "employee_name": {"type": "string"},
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                },
                "required": ["employee_name", "date_from", "date_to"],
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
                tool="vehicle_usage_get_employee_period_report",
                error="VehicleUsageStore is not configured",
            )
        employee_name = str(args.get("employee_name") or "").strip()
        if not employee_name:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="vehicle_usage_get_employee_period_report",
                error="employee_name is required",
            )
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_get_employee_period_report",
            data=self._store.get_employee_period_report(
                employee_name=employee_name,
                date_from=_request_date(args.get("date_from")),
                date_to=_request_date(args.get("date_to")),
            ),
        )


class VehicleGetVehiclePeriodReportTool:
    name = "vehicle_usage_get_vehicle_period_report"

    def __init__(self, store: VehicleUsageStorePort | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_get_vehicle_period_report",
            description="Read a period report for one vehicle by date range.",
            parameters={
                "type": "object",
                "properties": {
                    "vehicle_name": {"type": "string"},
                    "date_from": {"type": "string"},
                    "date_to": {"type": "string"},
                },
                "required": ["vehicle_name", "date_from", "date_to"],
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
                tool="vehicle_usage_get_vehicle_period_report",
                error="VehicleUsageStore is not configured",
            )
        vehicle_name = str(args.get("vehicle_name") or "").strip()
        if not vehicle_name:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="vehicle_usage_get_vehicle_period_report",
                error="vehicle_name is required",
            )
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_get_vehicle_period_report",
            data=self._store.get_vehicle_period_report(
                vehicle_name=vehicle_name,
                date_from=_request_date(args.get("date_from")),
                date_to=_request_date(args.get("date_to")),
            ),
        )


class VehicleUpdateReportTool:
    name = "vehicle_usage_update_report"

    def __init__(
        self, store: VehicleUsageStorePort | None = None, *, allowed_user_ids: frozenset[int] = frozenset()
    ) -> None:
        self._store = store
        self._allowed_user_ids = allowed_user_ids

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle_usage_update_report",
            description="Apply confirmed partial corrections to an existing daily vehicle/staff report.",
            parameters={
                "type": "object",
                "properties": {
                    "request_date": {"type": "string"},
                    "people": {"type": "array", "items": {"type": "object"}},
                    "vehicles": {"type": "array", "items": {"type": "object"}},
                    "change_summary": {"type": "string"},
                },
                "required": ["request_date"],
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
                tool="vehicle_usage_update_report",
                error="VehicleUsageStore is not configured",
            )
        denied = _deny_if_not_operator("vehicle_usage_update_report", user_id, self._allowed_user_ids, self._store)
        if denied is not None:
            return denied
        people = args.get("people") if isinstance(args.get("people"), list) else []
        vehicles = args.get("vehicles") if isinstance(args.get("vehicles"), list) else []
        if not people and not vehicles:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="vehicle_usage_update_report",
                error="people or vehicles patch is required",
            )
        return ToolResult(
            status=ToolStatus.OK,
            tool="vehicle_usage_update_report",
            data=self._store.update_day_report(
                report_date=_request_date(args.get("request_date")),
                people=[item for item in people if isinstance(item, dict)],
                vehicles=[item for item in vehicles if isinstance(item, dict)],
                actor_user_id=user_id,
                change_summary=str(args.get("change_summary") or ""),
            ),
        )


def _deny_if_not_operator(
    tool: str, user_id: int | None, fallback_user_ids: frozenset[int], store: VehicleUsageStorePort | None
) -> ToolResult | None:
    allowed_user_ids = _operator_ids(store, fallback_user_ids)
    if not allowed_user_ids:
        return None
    if user_id is not None and user_id in allowed_user_ids:
        return None
    return ToolResult(
        status=ToolStatus.DENIED,
        tool=tool,
        error="Only the configured vehicle usage responsible user can change the daily report.",
        data={"allowed_user_ids": sorted(allowed_user_ids), "user_id": user_id},
    )


def _deny_if_not_admin(tool: str, user_id: int | None, admin_user_ids: frozenset[int]) -> ToolResult | None:
    if not admin_user_ids:
        return None
    if user_id is not None and user_id in admin_user_ids:
        return None
    return ToolResult(
        status=ToolStatus.DENIED,
        tool=tool,
        error="Only the configured vehicle usage administrator can change report operators.",
        data={"admin_user_ids": sorted(admin_user_ids), "user_id": user_id},
    )


def _operator_ids(store: VehicleUsageStorePort | None, fallback_user_ids: frozenset[int]) -> frozenset[int]:
    return frozenset(resolve_vehicle_usage_operator_ids(store, fallback_user_ids))


def _staff_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    people = parsed.get("people") or parsed.get("staff") or parsed.get("staff_entries")
    return [item for item in people if isinstance(item, dict)] if isinstance(people, list) else []


def _vehicle_entries(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    vehicles = parsed.get("vehicles") or parsed.get("vehicle_entries") or parsed.get("vehicle_assignments")
    return [item for item in vehicles if isinstance(item, dict)] if isinstance(vehicles, list) else []


def _employees_by_name(roster: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in roster:
        employee_id = optional_int(row.get("display_order") or row.get("id") or row.get("employee_id"))
        if employee_id is None:
            continue
        name = _norm_name(row.get("full_name") or row.get("name"))
        if name:
            result[name] = employee_id
    return result


def _vehicles_by_name(vehicles: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in vehicles:
        vehicle_id = optional_int(row.get("id") or row.get("vehicle_id"))
        if vehicle_id is None:
            continue
        for value in (row.get("brand_model"), row.get("registration_number"), row.get("vehicle_name"), row.get("name")):
            name = _norm_name(value)
            if name:
                result[name] = vehicle_id
    return result


def _driver_ids(entry: dict[str, Any], employees_by_name: dict[str, int]) -> list[int]:
    direct_id = optional_int(entry.get("employee_id"))
    if direct_id is not None:
        return [direct_id]
    raw_drivers = entry.get("drivers")
    if not isinstance(raw_drivers, list):
        raw_drivers = [entry.get("driver") or entry.get("employee_name")]
    result: list[int] = []
    for raw_name in raw_drivers:
        employee_id = _employee_id_by_name(raw_name, employees_by_name)
        if employee_id is not None and employee_id not in result:
            result.append(employee_id)
    return result


def _employee_id_by_name(raw_name: Any, employees_by_name: dict[str, int]) -> int | None:
    normalized = _norm_name(raw_name)
    if not normalized:
        return None
    direct = employees_by_name.get(normalized)
    if direct is not None:
        return direct
    raw_tokens = normalized.split()
    best: tuple[int, int] | None = None
    for candidate, employee_id in employees_by_name.items():
        candidate_tokens = candidate.split()
        if not candidate_tokens:
            continue
        matched = sum(1 for token in raw_tokens if any(_name_token_close(token, known) for known in candidate_tokens))
        required = len(raw_tokens) if len(raw_tokens) <= 2 else 2
        if matched < required:
            continue
        score = matched * 100 - abs(len(candidate) - len(normalized))
        if best is None or score > best[0]:
            best = (score, employee_id)
    return best[1] if best is not None else None


def _name_token_close(left: str, right: str) -> bool:
    if left == right or left.startswith(right) or right.startswith(left):
        return True
    if min(len(left), len(right)) < 4:
        return False
    return _levenshtein_distance_limited(left, right, 1) <= 1


def _levenshtein_distance_limited(left: str, right: str, limit: int) -> int:
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        row_min = current[0]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _augment_vehicle_assignments_from_text(
    parsed: dict[str, Any],
    source_text: str,
    roster: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
) -> None:
    normalized_text = _norm_name(source_text)
    if not normalized_text:
        return
    vehicle_aliases = _vehicle_aliases(vehicles)
    vehicle_entries = _vehicle_entries(parsed)
    if not vehicle_aliases:
        return
    if not vehicle_entries:
        inferred = _infer_vehicle_entries_from_text(source_text, roster, vehicle_aliases)
        if inferred:
            parsed["vehicles"] = inferred
        return
    for entry in vehicle_entries:
        if not _is_blank_vehicle_entry(entry):
            continue
        drivers = entry.get("drivers") if isinstance(entry.get("drivers"), list) else []
        if not drivers:
            drivers = [entry.get("driver") or entry.get("employee_name")]
        for raw_driver in drivers:
            driver_alias = _norm_name(raw_driver)
            if not driver_alias:
                continue
            vehicle_match = _vehicle_before_driver(normalized_text, driver_alias, vehicle_aliases)
            if vehicle_match is None:
                continue
            vehicle_id, vehicle_name = vehicle_match
            entry["vehicle_id"] = vehicle_id
            entry["vehicle_name"] = vehicle_name
            entry.setdefault("status", "in_use")
            break


def _infer_vehicle_entries_from_text(
    source_text: str,
    roster: list[dict[str, Any]],
    vehicle_aliases: list[tuple[str, int, str]],
) -> list[dict[str, Any]]:
    employee_aliases = _employee_aliases(roster)
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw_line in source_text.splitlines():
        line = _norm_name(raw_line)
        if not line:
            continue
        vehicle_match = _vehicle_in_line(line, vehicle_aliases)
        if vehicle_match is None:
            continue
        vehicle_id, vehicle_name = vehicle_match
        if vehicle_id in seen:
            continue
        seen.add(vehicle_id)
        drivers = _drivers_in_line(line, employee_aliases)
        status = _vehicle_status_from_line(line, bool(drivers))
        result.append(
            {
                "vehicle_id": vehicle_id,
                "vehicle_name": vehicle_name,
                "status": status,
                "drivers": drivers,
            }
        )
    return result


def _employee_aliases(roster: list[dict[str, Any]]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in roster:
        name = str(row.get("full_name") or row.get("name") or "").strip()
        alias = _norm_name(name)
        if len(alias) < 3 or alias in seen:
            continue
        seen.add(alias)
        result.append((alias, name))
    return sorted(result, key=lambda item: len(item[0]), reverse=True)


def _vehicle_in_line(line: str, vehicle_aliases: list[tuple[str, int, str]]) -> tuple[int, str] | None:
    for vehicle_alias, vehicle_id, vehicle_name in vehicle_aliases:
        if re.search(rf"(?<!\w){re.escape(vehicle_alias)}(?!\w)", line):
            return (vehicle_id, vehicle_name)
    return None


def _drivers_in_line(line: str, employee_aliases: list[tuple[str, str]]) -> list[str]:
    result: list[str] = []
    for employee_alias, employee_name in employee_aliases:
        if employee_name in result:
            continue
        if re.search(rf"(?<!\w){re.escape(employee_alias)}(?!\w)", line):
            result.append(employee_name)
    return result


def _vehicle_status_from_line(line: str, has_drivers: bool) -> str:
    if any(marker in line for marker in ("простой", "свобод", "не работ", "idle")):
        return "idle"
    if any(marker in line for marker in ("ремонт", "repair")):
        return "repair"
    if has_drivers or any(marker in line for marker in ("работ", "выезд", "in use", "in_use")):
        return "in_use"
    return "unknown"


def _validate_report_completeness(
    *,
    roster: list[dict[str, Any]],
    vehicles: list[dict[str, Any]],
    employee_statuses_by_id: dict[int, tuple[str, str]],
    vehicle_assignments: list[tuple[int, int | None, str, str]],
    driver_employee_ids: set[int],
    employee_requires_vehicle_ids: set[int],
    unknown_vehicle_refs: list[str] | None = None,
) -> dict[str, Any]:
    employees_by_id = _employees_by_id(roster)
    vehicles_by_id = _vehicles_by_id(vehicles)
    assigned_vehicle_ids = {vehicle_id for vehicle_id, _, _, _ in vehicle_assignments}
    missing_employees = [
        name for employee_id, name in employees_by_id.items() if employee_id not in employee_statuses_by_id
    ]
    missing_vehicles = [name for vehicle_id, name in vehicles_by_id.items() if vehicle_id not in assigned_vehicle_ids]
    unknown_employees = [
        employees_by_id.get(employee_id, str(employee_id))
        for employee_id, (status, _) in employee_statuses_by_id.items()
        if _is_unknown_status(status)
    ]
    unknown_vehicles = [
        vehicles_by_id.get(vehicle_id, str(vehicle_id))
        for vehicle_id, _, status, _ in vehicle_assignments
        if _is_unknown_status(status)
    ]
    employees_without_vehicle = [
        employees_by_id.get(employee_id, str(employee_id))
        for employee_id in sorted(employee_requires_vehicle_ids - driver_employee_ids)
    ]
    in_use_without_drivers = [
        vehicles_by_id.get(vehicle_id, str(vehicle_id))
        for vehicle_id, employee_id, status, _ in vehicle_assignments
        if employee_id is None and _norm_name(status) in {"in_use", "used", "work", "worked", "в работе", "работает"}
    ]
    missing = []
    if missing_employees:
        missing.append({"kind": "employees", "items": missing_employees})
    if missing_vehicles:
        missing.append({"kind": "vehicles", "items": missing_vehicles})
    if employees_without_vehicle:
        missing.append({"kind": "employee_vehicle_links", "items": employees_without_vehicle})
    if in_use_without_drivers:
        missing.append({"kind": "vehicle_drivers", "items": in_use_without_drivers})
    if unknown_vehicle_refs:
        missing.append({"kind": "unknown_vehicle_references", "items": unknown_vehicle_refs})
    unknown = []
    if unknown_employees:
        unknown.append({"kind": "employee_statuses", "items": unknown_employees})
    if unknown_vehicles:
        unknown.append({"kind": "vehicle_statuses", "items": unknown_vehicles})
    questions = _clarification_questions(missing, unknown)
    return {
        "needs_clarification": bool(missing or unknown),
        "missing": missing,
        "unknown": unknown,
        "questions": questions,
    }


def _employees_by_id(roster: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in roster:
        employee_id = optional_int(row.get("display_order") or row.get("id") or row.get("employee_id"))
        name = str(row.get("full_name") or row.get("name") or employee_id or "").strip()
        if employee_id is not None and name:
            result[employee_id] = name
    return result


def _vehicles_by_id(vehicles: list[dict[str, Any]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in vehicles:
        vehicle_id = optional_int(row.get("id") or row.get("vehicle_id"))
        name = str(row.get("brand_model") or row.get("vehicle_name") or row.get("name") or vehicle_id or "").strip()
        if vehicle_id is not None and name:
            result[vehicle_id] = name
    return result


def _status_requires_vehicle(value: Any) -> bool:
    normalized = _norm_name(value)
    return normalized in {"car", "on_car", "auto", "vehicle"} or "авто" in normalized


def _is_unknown_status(value: Any) -> bool:
    normalized = _norm_name(value)
    return not normalized or normalized in {"unknown", "неизвестно", "неизвестен", "не распознано"}


def _clarification_questions(missing: list[dict[str, Any]], unknown: list[dict[str, Any]]) -> list[str]:
    questions: list[str] = []
    for block in missing:
        items = ", ".join(str(item) for item in block.get("items", []) if str(item).strip())
        if not items:
            continue
        kind = block.get("kind")
        if kind == "employees":
            questions.append(f"Уточните статус сотрудников: {items}.")
        elif kind == "vehicles":
            questions.append(f"Уточните статус машин: {items}.")
        elif kind == "employee_vehicle_links":
            questions.append(f"Уточните машину для сотрудников: {items}.")
        elif kind == "vehicle_drivers":
            questions.append(f"Уточните водителей/сотрудников для машин: {items}.")
        elif kind == "unknown_vehicle_references":
            questions.append(f"Машины не найдены в справочнике: {items}. Уточните одну из известных машин.")
    for block in unknown:
        items = ", ".join(str(item) for item in block.get("items", []) if str(item).strip())
        if not items:
            continue
        kind = block.get("kind")
        if kind == "employee_statuses":
            questions.append(f"Статус сотрудников не распознан: {items}.")
        elif kind == "vehicle_statuses":
            questions.append(f"Статус машин не распознан: {items}.")
    return questions


def _vehicle_aliases(vehicles: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    result: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for row in vehicles:
        vehicle_id = optional_int(row.get("id") or row.get("vehicle_id"))
        vehicle_name = str(row.get("brand_model") or row.get("vehicle_name") or row.get("name") or "").strip()
        if vehicle_id is None or not vehicle_name:
            continue
        for value in (vehicle_name, row.get("registration_number")):
            alias = _norm_name(value)
            if len(alias) < 3 or (alias, vehicle_id) in seen:
                continue
            seen.add((alias, vehicle_id))
            result.append((alias, vehicle_id, vehicle_name))
    return sorted(result, key=lambda item: len(item[0]), reverse=True)


def _is_blank_vehicle_entry(entry: dict[str, Any]) -> bool:
    return optional_int(entry.get("vehicle_id")) is None and not _norm_name(
        entry.get("vehicle_name") or entry.get("name") or entry.get("vehicle")
    )


def _vehicle_before_driver(
    normalized_text: str,
    driver_alias: str,
    vehicle_aliases: list[tuple[str, int, str]],
) -> tuple[int, str] | None:
    for driver_match in re.finditer(rf"(?<!\w){re.escape(driver_alias)}(?!\w)", normalized_text):
        prefix = normalized_text[max(0, driver_match.start() - 60) : driver_match.start()]
        best: tuple[int, str, int] | None = None
        for vehicle_alias, vehicle_id, vehicle_name in vehicle_aliases:
            for vehicle_match in re.finditer(rf"(?<!\w){re.escape(vehicle_alias)}(?!\w)", prefix):
                distance = len(prefix) - vehicle_match.end()
                if best is None or distance < best[2]:
                    best = (vehicle_id, vehicle_name, distance)
        if best is not None:
            return (best[0], best[1])
    return None


def _status_with_vehicle_assignment(status: str, has_vehicle_assignment: bool) -> str:
    normalized = str(status or "").strip()
    if _is_non_work_status(normalized):
        return _canonical_non_work_status(normalized)
    if has_vehicle_assignment or _is_work_status(normalized):
        return "worked"
    return normalized or "unknown"


def _is_work_status(status: str) -> bool:
    return status.casefold() in {
        "",
        "unknown",
        "office",
        "in_office",
        "at_office",
        "work",
        "worked",
        "working",
        "on_car",
        "shift",
        "object",
        "on_object",
        "site",
        "on_site",
        "field",
        "trip",
        "работа",
        "работал",
        "работает",
        "на работе",
        "в офисе",
        "офис",
        "объект",
        "на объекте",
        "выезд",
        "на выезде",
        "на авто",
        "на машине",
        "на смене",
        "смена",
    }


def _is_non_work_status(status: str) -> bool:
    return status.casefold() in {
        "vacation",
        "on_leave",
        "leave",
        "holiday",
        "sick",
        "day_off",
        "not_required",
        "отпуск",
        "болеет",
        "больничный",
        "выходной",
        "не работает",
        "не требуется",
    }


def _canonical_non_work_status(status: str) -> str:
    lowered = status.casefold()
    if lowered in {"vacation", "on_leave", "leave", "holiday", "отпуск"}:
        return "vacation"
    if lowered in {"sick", "болеет", "больничный"}:
        return "sick"
    if lowered in {"day_off", "выходной"}:
        return "day_off"
    if lowered in {"not_required", "не требуется", "не работает"}:
        return "not_required"
    return status


def _norm_name(value: object) -> str:
    return " ".join(str(value or "").casefold().split())
