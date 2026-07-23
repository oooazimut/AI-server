"""Orchestrator-owned rendering of raw Logistics executor results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ai_server.models import ToolResult


@dataclass(frozen=True)
class LogisticsFormattedResult:
    status: str
    answer: str


def render_logistics_tool_result(
    *,
    tool_result: ToolResult,
    command_arguments: dict[str, Any] | None = None,
) -> LogisticsFormattedResult:
    """Expose exact Logistics facts to the Pro finalizer without a second specialist model."""

    data = tool_result.data if isinstance(tool_result.data, dict) else {}
    if tool_result.status in {"error", "not_configured", "not_available"}:
        return LogisticsFormattedResult(
            status="failed",
            answer=str(tool_result.error or "Logistics не вернул результат выполнения команды."),
        )
    if tool_result.status in {"invalid_tool_call", "contract_violation", "ambiguous", "denied"}:
        return LogisticsFormattedResult(
            status="needs_clarification",
            answer=str(tool_result.error or "Нужно уточнить параметры логистической команды."),
        )

    formatter = _FORMATTERS.get(tool_result.tool)
    if formatter is not None:
        answer = formatter(data)
    else:
        answer = _facts_answer(tool_result.tool, data or (command_arguments or {}))
    status = "needs_clarification" if bool(data.get("needs_clarification")) else "completed"
    return LogisticsFormattedResult(status=status, answer=answer)


def _operators(data: dict[str, Any]) -> str:
    ids = data.get("operator_user_ids")
    return f"Ответственные за отчёт по машинам: {', '.join(map(str, ids)) or 'не назначены'}." if isinstance(ids, list) else _facts_answer("vehicle_usage_get_operators", data)


def _report_saved(data: dict[str, Any]) -> str:
    if data.get("needs_clarification"):
        questions = data.get("questions") if isinstance(data.get("questions"), list) else []
        return "Отчёт сохранён как черновик. " + (
            "Нужно уточнить: " + "; ".join(str(item) for item in questions)
            if questions
            else "Нужно уточнить неполные данные."
        )
    return (
        "Отчёт по машинам сохранён. "
        f"Сотрудников: {data.get('staff_entries_saved', 0)}, "
        f"машин: {data.get('vehicles_saved', 0)}, "
        f"назначений: {data.get('vehicle_assignments_saved', 0)}."
    )


def _draft_saved(data: dict[str, Any]) -> str:
    return (
        f"Черновик отчёта по машинам сохранён"
        f"{' за ' + str(data.get('request_date')) if data.get('request_date') else ''}."
    )


def _day_started(data: dict[str, Any]) -> str:
    return (
        f"Запрос ежедневного отчёта по машинам подготовлен"
        f"{' за ' + str(data.get('request_date')) if data.get('request_date') else ''}."
    )


def _day_cancelled(data: dict[str, Any]) -> str:
    return (
        f"Отчёт по машинам отменён"
        f"{' за ' + str(data.get('request_date')) if data.get('request_date') else ''}."
    )


def _report_updated(data: dict[str, Any]) -> str:
    return (
        f"Отчёт по машинам обновлён"
        f"{' за ' + str(data.get('report_date')) if data.get('report_date') else ''}. "
        f"Изменений по сотрудникам: {data.get('employee_updates', 0)}, "
        f"по машинам: {data.get('vehicle_updates', 0)}."
    )


def _facts_answer(tool: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    if len(payload) > 20000:
        payload = payload[:20000] + "…"
    return f"Результат Logistics ({tool}): {payload}"


_FORMATTERS = {
    "vehicle_usage_context": lambda data: _facts_answer("vehicle_usage_context", data),
    "vehicle_usage_reference": lambda data: _facts_answer("vehicle_usage_reference", data),
    "vehicle_usage_get_operators": _operators,
    "vehicle_usage_set_operators": _operators,
    "vehicle_usage_start_day": _day_started,
    "vehicle_usage_get_report": lambda data: _facts_answer("vehicle_usage_get_report", data),
    "vehicle_usage_get_employee_period_report": lambda data: _facts_answer(
        "vehicle_usage_get_employee_period_report", data
    ),
    "vehicle_usage_get_vehicle_period_report": lambda data: _facts_answer(
        "vehicle_usage_get_vehicle_period_report", data
    ),
    "vehicle_usage_save_draft": _draft_saved,
    "vehicle_usage_save_report": _report_saved,
    "vehicle_usage_update_report": _report_updated,
    "vehicle_usage_cancel_day": _day_cancelled,
}


__all__ = ["LogisticsFormattedResult", "render_logistics_tool_result"]
