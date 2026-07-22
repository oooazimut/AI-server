"""Orchestrator-owned Bitrix meaning, defaults and semantic validation.

The planning model may suggest a command, but this module is the authority for
what common user phrases mean.  Bitrix specialists receive an already formed
command and remain transport/access/schema executors only.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, time, timedelta
from typing import Any

from ai_server.capability_registry import registry_tool, validate_tool_arguments
from ai_server.utils import MOSCOW_TZ

DEFAULT_RESULT_LIMIT = 10
DEFAULT_WAREHOUSE_PRODUCT_LIMIT = 50
DEFAULT_TASK_DEADLINE_HOUR = 19
DEFAULT_CALENDAR_HOUR = 12
DEFAULT_CALENDAR_DURATION_MINUTES = 30


class SemanticPolicyViolation(ValueError):
    """The proposed command does not preserve the meaning of the user request."""


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()


def _add_working_days(value: datetime, days: int) -> datetime:
    result = value
    remaining = days
    while remaining:
        result += timedelta(days=1)
        if result.weekday() < 5:
            remaining -= 1
    return result


def _default_task_deadline(now: datetime) -> str:
    day = _add_working_days(now.astimezone(MOSCOW_TZ), 3)
    return datetime.combine(day.date(), time(DEFAULT_TASK_DEADLINE_HOUR), tzinfo=MOSCOW_TZ).isoformat()


def _default_calendar_start(now: datetime) -> datetime:
    day = now.astimezone(MOSCOW_TZ) + timedelta(days=3)
    return datetime.combine(day.date(), time(DEFAULT_CALENDAR_HOUR), tzinfo=MOSCOW_TZ)


def _warehouse_semantics(request: str, arguments: dict[str, Any]) -> dict[str, Any]:
    text = _text(request)
    result = dict(arguments)
    list_all = bool(re.search(r"\b(?:все|список)\s+(?:склад|склады|складов)\b", text))
    product_match = re.search(
        r"\b(?:найди|найдите|покажи|покажите)\s+(.+?)\s+(?:на|в)\s+склад(?:е|у)?\s+(.+)$",
        text,
    )
    warehouse_match = re.search(r"\bсклад(?:е|у|а|ов|ы)?\s+(.+)$", text)
    if list_all:
        result.update({"query": "все", "list_all": True, "include_products": False})
        return result
    if warehouse_match:
        warehouse = warehouse_match.group(1).strip(" .,:;-")
        if warehouse:
            result["query"] = warehouse
    if product_match:
        product = product_match.group(1).strip(" .,:;-")
        warehouse = product_match.group(2).strip(" .,:;-")
        result.update(
            {
                "query": warehouse,
                "product_query": product,
                "include_products": True,
                "product_limit": DEFAULT_WAREHOUSE_PRODUCT_LIMIT,
            }
        )
    elif any(marker in text for marker in ("покажи склад", "остат", "товар", "налич")):
        result.update({"include_products": True, "product_limit": DEFAULT_WAREHOUSE_PRODUCT_LIMIT})
    elif "найди склад" in text:
        result["include_products"] = False

    page_match = re.search(r"\b(\d+)\s*(?:-?ю|страниц)", text)
    if page_match:
        result["product_offset"] = max(0, (int(page_match.group(1)) - 1) * DEFAULT_WAREHOUSE_PRODUCT_LIMIT)
        result["include_products"] = True
    result.setdefault("limit", DEFAULT_RESULT_LIMIT)
    return result


def _task_read_semantics(tool_name: str, request: str, arguments: dict[str, Any]) -> dict[str, Any]:
    text = _text(request)
    result = dict(arguments)
    if tool_name == "bitrix_my_tasks":
        result.setdefault("status", "closed" if "закрыт" in text else "open")
        result.setdefault("limit", DEFAULT_RESULT_LIMIT)
        return result
    if tool_name == "bitrix_task_search":
        if result.get("scope") == "all" and not re.search(r"\b(?:все|всех)\s+задач", text):
            result["scope"] = "my"
        if not any(key in result for key in ("task_id", "query", "project_id", "project_name", "comment_query")):
            result.setdefault("scope", "my")
        result.setdefault("status", "overdue" if "просроч" in text else "active")
        result.setdefault("limit", DEFAULT_RESULT_LIMIT)
    return result


def _task_create_semantics(arguments: dict[str, Any], now: datetime) -> dict[str, Any]:
    result = dict(arguments)
    if result.get("responsible_id") in (None, ""):
        result["responsible_self"] = True
    if not result.get("deadline_iso") and not result.get("no_deadline"):
        result["deadline_iso"] = _default_task_deadline(now)
    return result


def _calendar_semantics(arguments: dict[str, Any], now: datetime) -> dict[str, Any]:
    result = dict(arguments)
    if not result.get("start_iso") and not result.get("date_iso"):
        start = _default_calendar_start(now)
        result["start_iso"] = start.isoformat()
        result.setdefault("end_iso", (start + timedelta(minutes=DEFAULT_CALENDAR_DURATION_MINUTES)).isoformat())
    return result


def _portal_search_semantics(request: str, arguments: dict[str, Any]) -> dict[str, Any]:
    text = _text(request)
    result = dict(arguments)
    if any(marker in text for marker in ("диск", "файл", "документ", "папк")):
        result.setdefault("scope", "documents")
    result.setdefault("limit", DEFAULT_RESULT_LIMIT)
    return result


def _expected_tool(request: str) -> str | None:
    text = _text(request)
    if "склад" in text or "остат" in text:
        return "bitrix_warehouse_search"
    if any(marker in text for marker in ("напомни", "напоминани", "календар")):
        return "calendar_event_draft"
    if re.search(r"\bсозда(?:й|ть|йте)\s+задач", text):
        return "task_create_draft"
    if any(marker in text for marker in ("диск", "файл", "документ", "папк")):
        return "portal_search"
    if "мои задач" in text or text.startswith("покажи задачи"):
        return "bitrix_my_tasks"
    return None


def normalize_command_arguments(
    tool_name: str,
    request: str,
    arguments: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Apply orchestrator defaults before JSON-schema validation."""

    current = (now or datetime.now(MOSCOW_TZ)).astimezone(MOSCOW_TZ)
    if tool_name == "bitrix_warehouse_search":
        return _warehouse_semantics(request, arguments)
    if tool_name in {"bitrix_my_tasks", "bitrix_task_search"}:
        return _task_read_semantics(tool_name, request, arguments)
    if tool_name == "task_create_draft":
        return _task_create_semantics(arguments, current)
    if tool_name == "calendar_event_draft":
        return _calendar_semantics(arguments, current)
    if tool_name == "portal_search":
        return _portal_search_semantics(request, arguments)
    return dict(arguments)


def normalize_plan(plan: Any, *, task: Any, constraints: dict[str, Any], now: datetime | None = None) -> Any:
    """Normalize and validate structured commands before any specialist call."""

    if getattr(plan, "state", None) != "EXECUTE":
        return plan
    current = (now or datetime.now(MOSCOW_TZ)).astimezone(MOSCOW_TZ)
    normalized_subtasks = []
    for subtask in plan.subtasks:
        command = subtask.structured_command
        if subtask.specialist_id != "bitrix24" or command is None:
            normalized_subtasks.append(subtask)
            continue
        semantic_request = task.request if len(plan.subtasks) == 1 else subtask.request
        expected = _expected_tool(semantic_request)
        if expected and command.tool_name != expected:
            raise SemanticPolicyViolation("SEMANTIC_TOOL_MISMATCH")
        arguments = normalize_command_arguments(
            command.tool_name,
            semantic_request,
            command.arguments,
            now=current,
        )

        specialist_catalog = constraints["capability_catalog"].get(subtask.specialist_id) or {}
        tool_contract = registry_tool(specialist_catalog, command.tool_name)
        errors = validate_tool_arguments(dict((tool_contract or {}).get("parameters") or {}), arguments)
        if errors:
            raise SemanticPolicyViolation("SEMANTIC_ARGUMENTS_INVALID")
        normalized_command = replace(command, arguments=arguments)
        normalized_subtasks.append(replace(subtask, structured_command=normalized_command))
    return replace(plan, subtasks=normalized_subtasks)


__all__ = [
    "DEFAULT_CALENDAR_HOUR",
    "DEFAULT_RESULT_LIMIT",
    "DEFAULT_TASK_DEADLINE_HOUR",
    "DEFAULT_WAREHOUSE_PRODUCT_LIMIT",
    "SemanticPolicyViolation",
    "normalize_command_arguments",
    "normalize_plan",
]
