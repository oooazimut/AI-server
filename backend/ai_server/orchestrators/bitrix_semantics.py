"""Orchestrator-owned Bitrix meaning, defaults and semantic validation.

The planning model may suggest a command, but this module is the authority for
what common user phrases mean.  Bitrix specialists receive an already formed
command and remain transport/access/schema executors only.
"""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import date, datetime, time, timedelta
from typing import Any

from ai_server.capability_registry import registry_tool, validate_tool_arguments
from ai_server.orchestrators.entity_catalog import find_entities_in_text, resolve_entity
from ai_server.orchestrators.orchestrator_policy import bitrix_policy_defaults, bitrix_policy_templates
from ai_server.utils import MOSCOW_TZ

_DEFAULTS = bitrix_policy_defaults()
_TEMPLATES = bitrix_policy_templates()
DEFAULT_RESULT_LIMIT = _DEFAULTS["result_limit"]
DEFAULT_WAREHOUSE_PRODUCT_LIMIT = _DEFAULTS["warehouse_page_size"]
DEFAULT_TASK_DEADLINE_WORKING_DAYS = _DEFAULTS["task_deadline_working_days"]
DEFAULT_TASK_DEADLINE_HOUR = _DEFAULTS["task_deadline_hour"]
DEFAULT_CALENDAR_WORKING_DAYS = _DEFAULTS["calendar_start_working_days"]
DEFAULT_CALENDAR_HOUR = _DEFAULTS["calendar_start_hour"]
DEFAULT_CALENDAR_DURATION_MINUTES = _DEFAULTS["calendar_duration_minutes"]

_TASK_CLOSE_COMMAND_SUMMARIES = frozenset(
    {
        "закрой",
        "закрой задачу",
        "закрой эту задачу",
        "закройте задачу",
        "закрыть",
        "закрыть задачу",
        "задачу закрыть",
        "закрываем задачу",
        "отметь задачу выполненной",
        "отметить задачу выполненной",
        "пометь задачу выполненной",
        "пометить задачу выполненной",
        "close",
        "close task",
        "complete task",
        "mark task complete",
        "mark task completed",
    }
)


class SemanticPolicyViolation(ValueError):
    """The proposed command does not preserve the meaning of the user request."""


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip()


def _task_close_is_command_summary(value: object, task_id: object) -> bool:
    text = _text(str(value or ""))
    if not text:
        return False
    try:
        normalized_task_id = int(task_id)
    except (TypeError, ValueError):
        normalized_task_id = None
    if normalized_task_id is not None:
        text = re.sub(rf"(?<!\d)#?{normalized_task_id}(?!\d)", " ", text)
    text = re.sub(r"[\"'`«»“”№#.,;:!?()\[\]\-]+", " ", text)
    ignored_tokens = {"bitrix", "битрикс", "в", "пожалуйста"}
    text = _text(" ".join(token for token in text.split() if token not in ignored_tokens))
    return text in _TASK_CLOSE_COMMAND_SUMMARIES


def _normalize_task_close_action(value: object) -> str:
    text = _text(str(value or "complete"))
    if text in {"approve", "accept", "принять", "утвердить"}:
        return "approve"
    return "complete"


def _normalize_task_close_status(value: object) -> str:
    text = _text(str(value or ""))
    aliases = {
        "completed": {
            "completed",
            "complete",
            "done",
            "выполнено",
            "готово",
            "полностью",
            "выполнена полностью",
        },
        "partial": {
            "partial",
            "partially_done",
            "partly_done",
            "частично",
            "выполнена частично",
        },
        "not_done": {
            "not_done",
            "not_completed",
            "not done",
            "не выполнено",
            "не сделано",
            "не выполнена",
        },
        "unconfirmed": {
            "unconfirmed",
            "unknown",
            "unclear",
            "не подтверждено",
            "неизвестно",
            "непонятно",
        },
    }
    for canonical, values in aliases.items():
        if text in values:
            return canonical
    return text


def _add_working_days(value: datetime, days: int) -> datetime:
    result = value
    remaining = days
    while remaining:
        result += timedelta(days=1)
        if result.weekday() < 5:
            remaining -= 1
    return result


def _default_task_deadline(now: datetime) -> str:
    day = _add_working_days(now.astimezone(MOSCOW_TZ), DEFAULT_TASK_DEADLINE_WORKING_DAYS)
    return datetime.combine(day.date(), time(DEFAULT_TASK_DEADLINE_HOUR), tzinfo=MOSCOW_TZ).isoformat()


def _default_calendar_start(now: datetime) -> datetime:
    day = _add_working_days(now.astimezone(MOSCOW_TZ), DEFAULT_CALENDAR_WORKING_DAYS)
    return datetime.combine(day.date(), time(DEFAULT_CALENDAR_HOUR), tzinfo=MOSCOW_TZ)


def _catalog_available(catalog: dict[str, Any] | None) -> bool:
    return bool(catalog) and str(catalog.get("status") or "") in {"ready", "stale"}


def _catalog_item_by_id(
    catalog: dict[str, Any] | None,
    entity_type: str,
    entity_id: object,
) -> dict[str, Any] | None:
    try:
        wanted = int(entity_id)
    except (TypeError, ValueError):
        return None
    for item in (catalog or {}).get(entity_type) or []:
        try:
            if int(item.get("id")) == wanted:
                return item
        except (TypeError, ValueError):
            continue
    return None


def _warehouse_semantics(
    request: str,
    arguments: dict[str, Any],
    entity_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = _text(request)
    result = dict(arguments)
    list_all = bool(re.search(r"\b(?:все|список)\s+(?:склад|склады|складов)\b", text))
    result["list_all"] = list_all
    product_match = re.search(
        r"\b(?:найди|найдите|покажи|покажите)\s+(.+?)\s+(?:на|в)\s+склад(?:е|у)?\s+(.+)$",
        text,
    )
    warehouse_match = re.search(r"\bсклад(?:е|у|а|ов|ы)?\s+(.+)$", text)
    if list_all:
        result.update(
            {
                "query": "все",
                "list_all": True,
                "include_products": False,
                "limit": DEFAULT_RESULT_LIMIT,
                "product_limit": DEFAULT_WAREHOUSE_PRODUCT_LIMIT,
                "product_offset": 0,
            }
        )
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
    elif any(
        marker in text
        for marker in ("покажи склад", "найди склад", "выведи склад", "ищи склад", "остат", "товар", "налич")
    ):
        result.update({"include_products": True, "product_limit": DEFAULT_WAREHOUSE_PRODUCT_LIMIT})

    page_match = re.search(r"\b(\d+)\s*(?:-?ю|страниц)", text)
    if page_match:
        result["product_offset"] = max(0, (int(page_match.group(1)) - 1) * DEFAULT_WAREHOUSE_PRODUCT_LIMIT)
        result["include_products"] = True
    result.setdefault("limit", DEFAULT_RESULT_LIMIT)
    result.setdefault("product_limit", DEFAULT_WAREHOUSE_PRODUCT_LIMIT)
    result.setdefault("product_offset", 0)
    result.setdefault("include_products", False)
    if not list_all:
        if not _catalog_available(entity_catalog):
            raise SemanticPolicyViolation("ENTITY_CATALOG_UNAVAILABLE")
        named = find_entities_in_text(entity_catalog, "warehouses", request)
        if len(named) > 1:
            raise SemanticPolicyViolation("ENTITY_AMBIGUOUS")
        supplied_store_id = result.get("store_id")
        matched = named[0] if named else (
            resolve_entity(entity_catalog, "warehouses", result.get("query"))[0]
            or _catalog_item_by_id(entity_catalog, "warehouses", supplied_store_id)
        )
        if matched is None:
            raise SemanticPolicyViolation("ENTITY_NOT_FOUND")
        if supplied_store_id not in (None, "") and int(supplied_store_id) != int(matched["id"]):
            raise SemanticPolicyViolation("ENTITY_ID_MISMATCH")
        result["store_id"] = int(matched["id"])
        result["query"] = str(matched["name"])
    return result


def _task_read_semantics(
    tool_name: str,
    request: str,
    arguments: dict[str, Any],
    *,
    task: Any | None = None,
    entity_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = _text(request)
    result = dict(arguments)
    if tool_name == "bitrix_my_tasks":
        result.setdefault("status", "closed" if "закрыт" in text else "open")
        result.setdefault("limit", DEFAULT_RESULT_LIMIT)
        result.setdefault("offset", 0)
        return result
    if tool_name == "bitrix_task_search":
        named_users = find_entities_in_text(entity_catalog or {}, "users", request)
        if len(named_users) > 1:
            raise SemanticPolicyViolation("ENTITY_AMBIGUOUS")
        supplied_user_id = result.get("target_user_id")
        supplied_user_name = str(result.get("target_user_name") or "").strip()
        if named_users or supplied_user_id not in (None, "") or supplied_user_name:
            if not _catalog_available(entity_catalog):
                raise SemanticPolicyViolation("ENTITY_CATALOG_UNAVAILABLE")
            matched_user = named_users[0] if named_users else (
                resolve_entity(entity_catalog or {}, "users", supplied_user_name)[0]
                or _catalog_item_by_id(entity_catalog, "users", supplied_user_id)
            )
            if matched_user is None:
                raise SemanticPolicyViolation("ENTITY_NOT_FOUND")
            target_user_id = int(matched_user["id"])
            if supplied_user_id not in (None, "") and int(supplied_user_id) != target_user_id:
                raise SemanticPolicyViolation("ENTITY_ID_MISMATCH")
            result["target_user_id"] = target_user_id
            result["target_user_name"] = str(matched_user["name"])
        project_name = str(result.get("project_name") or "").strip()
        if project_name and not result.get("project_id"):
            if not _catalog_available(entity_catalog):
                raise SemanticPolicyViolation("ENTITY_CATALOG_UNAVAILABLE")
            project, ambiguous = resolve_entity(entity_catalog, "projects", project_name)
            if ambiguous:
                raise SemanticPolicyViolation("ENTITY_AMBIGUOUS")
            if project is None:
                raise SemanticPolicyViolation("ENTITY_NOT_FOUND")
            result["project_id"] = int(project["id"])
            result["project_name"] = str(project["name"])
        if result.get("scope") == "all" and not re.search(r"\b(?:все|всех)\s+задач", text):
            result["scope"] = "my"
        if not any(key in result for key in ("task_id", "query", "project_id", "project_name", "comment_query")):
            result.setdefault("scope", "my")
        result.setdefault("status", "overdue" if "просроч" in text else "active")
        result.setdefault("include_closed", result["status"] in {"closed", "all"})
        result.setdefault("include_comments", False)
        result.setdefault("limit", DEFAULT_RESULT_LIMIT)
        result.setdefault("offset", 0)
    return result


def _task_create_semantics(
    arguments: dict[str, Any],
    now: datetime,
    *,
    task: Any | None = None,
    entity_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(arguments)
    task_user_id = None
    if task is not None:
        try:
            task_user_id = int(task.user.id) if task.user.id not in (None, "") else None
        except (TypeError, ValueError):
            task_user_id = None
    named_users = find_entities_in_text(entity_catalog or {}, "users", task.request) if task is not None else []
    if len(named_users) > 1:
        raise SemanticPolicyViolation("ENTITY_AMBIGUOUS")
    supplied_responsible_id = result.get("responsible_id")
    supplied_responsible_name = str(result.get("responsible_name") or "").strip()
    requester_name = str(task.user.display_name or "").strip() if task is not None else ""
    named_responsible_requested = bool(
        named_users
        or (
            supplied_responsible_id not in (None, "")
            and task_user_id is not None
            and int(supplied_responsible_id) != task_user_id
        )
        or (
            supplied_responsible_name
            and _text(supplied_responsible_name) != _text(requester_name)
        )
    )
    if named_responsible_requested:
        if not _catalog_available(entity_catalog):
            raise SemanticPolicyViolation("ENTITY_CATALOG_UNAVAILABLE")
        matched_user = named_users[0] if named_users else (
            resolve_entity(entity_catalog or {}, "users", supplied_responsible_name)[0]
            or _catalog_item_by_id(entity_catalog, "users", supplied_responsible_id)
        )
        if matched_user is None:
            raise SemanticPolicyViolation("ENTITY_NOT_FOUND")
        responsible_id = int(matched_user["id"])
        if supplied_responsible_id not in (None, "") and int(supplied_responsible_id) != responsible_id:
            raise SemanticPolicyViolation("ENTITY_ID_MISMATCH")
        result["responsible_id"] = responsible_id
        result["responsible_name"] = str(matched_user["name"])
        result.pop("responsible_self", None)
    elif result.get("responsible_id") in (None, ""):
        if task_user_id is None:
            result["responsible_self"] = True
        else:
            result["responsible_id"] = task_user_id
            result["responsible_name"] = str(task.user.display_name or "").strip()
            result.pop("responsible_self", None)
    if not result.get("deadline_iso") and not result.get("no_deadline"):
        result["deadline_iso"] = _default_task_deadline(now)
    if not str(result.get("description") or "").strip() and str(result.get("title") or "").strip():
        result["description"] = str(_TEMPLATES["task_description"]).format(
            title=str(result["title"]).strip()
        )
    explicit_project_name = str(result.get("project_name") or result.get("group_name") or "").strip()
    project_was_explicit = bool(
        task is not None and re.search(r"\b(?:проект|групп)\w*\b", _text(task.request))
    )
    if explicit_project_name and not project_was_explicit and not result.get("group_id"):
        result.pop("project_name", None)
        result.pop("group_name", None)
        explicit_project_name = ""
    if explicit_project_name and not result.get("group_id"):
        if not _catalog_available(entity_catalog):
            raise SemanticPolicyViolation("ENTITY_CATALOG_UNAVAILABLE")
        project, ambiguous = resolve_entity(entity_catalog, "projects", explicit_project_name)
        if ambiguous:
            raise SemanticPolicyViolation("ENTITY_AMBIGUOUS")
        if project is None:
            raise SemanticPolicyViolation("ENTITY_NOT_FOUND")
        result["group_id"] = int(project["id"])
        result["project_name"] = str(project["name"])
    if not any(result.get(key) for key in ("group_id", "project_name", "group_name")):
        responsible_name = str(result.get("responsible_name") or "").strip()
        if responsible_name:
            if not _catalog_available(entity_catalog):
                raise SemanticPolicyViolation("ENTITY_CATALOG_UNAVAILABLE")
            personal_project_name = " ".join(responsible_name.split()[:2])
            project, ambiguous = resolve_entity(entity_catalog or {}, "projects", personal_project_name)
            if ambiguous:
                raise SemanticPolicyViolation("ENTITY_AMBIGUOUS")
            result["project_name"] = personal_project_name
            result["_default_personal_project"] = True
            result["_default_personal_project_owner_id"] = int(result.get("responsible_id") or task_user_id or 0)
            result["_default_personal_project_missing"] = project is None
            if project is not None:
                result["group_id"] = int(project["id"])
    return result


def _calendar_semantics(arguments: dict[str, Any], now: datetime, *, task: Any | None = None) -> dict[str, Any]:
    result = dict(arguments)
    start: datetime | None = None
    raw_start = str(result.get("start_iso") or "").strip()
    raw_date = str(result.get("date_iso") or "").strip()
    if raw_start:
        try:
            start = datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SemanticPolicyViolation("CALENDAR_START_INVALID") from exc
        if start.tzinfo is None:
            start = start.replace(tzinfo=MOSCOW_TZ)
        start = start.astimezone(MOSCOW_TZ)
    elif raw_date:
        try:
            calendar_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise SemanticPolicyViolation("CALENDAR_DATE_INVALID") from exc
        start = datetime.combine(calendar_date, time(DEFAULT_CALENDAR_HOUR), tzinfo=MOSCOW_TZ)
    else:
        start = _default_calendar_start(now)
        if task is not None:
            time_match = re.search(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", _text(task.request))
            if time_match:
                start = datetime.combine(
                    start.date(),
                    time(int(time_match.group(1)), int(time_match.group(2))),
                    tzinfo=MOSCOW_TZ,
                )
    result["start_iso"] = start.isoformat()
    result.pop("date_iso", None)
    if not result.get("end_iso"):
        result["end_iso"] = (start + timedelta(minutes=DEFAULT_CALENDAR_DURATION_MINUTES)).isoformat()
    if task is not None and not result.get("owner_name") and not result.get("attendee_ids"):
        result["owner_name"] = str(task.user.display_name or "").strip()
    return result


def _task_close_semantics(request: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = dict(arguments)
    text = _text(request)
    if re.search(r"\b(?:закрой|закрыть|заверши|завершить)\s+задач", text):
        result["close_now"] = True
    task_id = result.get("task_id") or result.get("id") or result.get("ID")
    summary_fields = ("completion_summary", "result_text", "summary")
    for key in summary_fields:
        if _task_close_is_command_summary(result.get(key), task_id):
            result.pop(key, None)
    if "action" in result or "close_action" in result:
        result["action"] = _normalize_task_close_action(
            result.get("action") or result.get("close_action")
        )
        result.pop("close_action", None)
    if "overall_status" in result or "completion_status" in result:
        result["overall_status"] = _normalize_task_close_status(
            result.get("overall_status") or result.get("completion_status")
        )
        result.pop("completion_status", None)
    has_result = any(
        result.get(key)
        for key in (
            "completion_summary",
            "equipment_consumables",
            "overall_status",
            "not_done_items",
            "unconfirmed_items",
            "status_reasons",
        )
    )
    if not has_result:
        result["overall_status"] = "unconfirmed"
        result["unconfirmed_items"] = [str(_TEMPLATES["task_close_unconfirmed_item"])]
        result["missing_fields"] = list(_TEMPLATES["task_close_missing_fields"])
    return result


def _portal_search_semantics(request: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = dict(arguments)
    result.setdefault("scope", "documents")
    result.setdefault("limit", DEFAULT_RESULT_LIMIT)
    result.setdefault("offset", 0)
    result.setdefault("show_all", False)
    return result


def _project_search_semantics(
    request: str,
    arguments: dict[str, Any],
    entity_catalog: dict[str, Any] | None,
) -> dict[str, Any]:
    result = dict(arguments)
    if not _catalog_available(entity_catalog):
        raise SemanticPolicyViolation("ENTITY_CATALOG_UNAVAILABLE")
    named = find_entities_in_text(entity_catalog or {}, "projects", request)
    if len(named) > 1:
        raise SemanticPolicyViolation("ENTITY_AMBIGUOUS")
    project = named[0] if named else resolve_entity(
        entity_catalog or {}, "projects", result.get("query")
    )[0]
    if project is not None:
        result["project_id"] = int(project["id"])
        result["query"] = str(project["name"])
    else:
        raise SemanticPolicyViolation("ENTITY_NOT_FOUND")
    result.setdefault("limit", DEFAULT_RESULT_LIMIT)
    return result


def _project_create_semantics(arguments: dict[str, Any], *, task: Any | None = None) -> dict[str, Any]:
    result = dict(arguments)
    result.setdefault("opened", True)
    result.setdefault("visible", True)
    result.setdefault("project", True)
    result.setdefault("subject_id", 1)
    if task is not None:
        try:
            actor_id = int(task.user.id) if task.user.id not in (None, "") else None
        except (TypeError, ValueError):
            actor_id = None
        if actor_id is not None:
            result.setdefault("owner_id", actor_id)
        actor_name = str(task.user.display_name or "").strip()
        if actor_name and _text(str(result.get("name") or "")) == _text(" ".join(actor_name.split()[:2])):
            result.setdefault("personal_for_self", True)
    return result


def _task_close_control_semantics(
    request: str,
    arguments: dict[str, Any],
    entity_catalog: dict[str, Any] | None,
) -> dict[str, Any]:
    result = dict(arguments)
    result.setdefault("operation", "prepare")
    if result["operation"] != "prepare" or result.get("action") not in {
        "add_operator",
        "remove_operator",
        "add_controlled_user",
        "remove_controlled_user",
    }:
        return result
    if not _catalog_available(entity_catalog):
        raise SemanticPolicyViolation("ENTITY_CATALOG_UNAVAILABLE")
    named = find_entities_in_text(entity_catalog or {}, "users", request)
    supplied_id = result.get("target_user_id")
    supplied_name = str(result.get("target_user_name") or "").strip()
    if len(named) > 1:
        raise SemanticPolicyViolation("ENTITY_AMBIGUOUS")
    matched = named[0] if named else (
        resolve_entity(entity_catalog or {}, "users", supplied_name)[0]
        or _catalog_item_by_id(entity_catalog, "users", supplied_id)
    )
    if matched is None:
        raise SemanticPolicyViolation("ENTITY_NOT_FOUND")
    if supplied_id not in (None, "") and int(supplied_id) != int(matched["id"]):
        raise SemanticPolicyViolation("ENTITY_ID_MISMATCH")
    result["target_user_id"] = int(matched["id"])
    result["target_user_name"] = str(matched["name"])
    return result


def _task_close_report_incident_semantics(arguments: dict[str, Any]) -> dict[str, Any]:
    result = dict(arguments)
    action = _text(str(result.get("action") or ""))
    if action in {
        "restore",
        "1",
        "one",
        "first",
        "первый",
        "первый вариант",
        "восстановить",
        "вернуть",
    }:
        result["action"] = "restore"
    elif action in {
        "accept_missing",
        "accept",
        "2",
        "two",
        "second",
        "второй",
        "второй вариант",
        "удалить",
        "все в порядке",
    }:
        result["action"] = "accept_missing"
    return result


def _expected_tool(request: str, entity_catalog: dict[str, Any] | None = None) -> str | None:
    text = _text(request)
    if "склад" in text or "остат" in text:
        return "bitrix_warehouse_search"
    if any(marker in text for marker in ("напомни", "напоминани", "календар")):
        return "calendar_event_draft"
    if re.search(r"\bсозда(?:й|ть|йте)\s+задач", text):
        return "task_create_draft"
    if re.search(r"\b(?:закрой|закрыть|заверши|завершить)\s+задач", text):
        return "task_close_draft"
    if any(marker in text for marker in ("диск", "файл", "документ", "папк")):
        return "portal_search"
    if (
        "мои задач" in text
        or (
            text.startswith("покажи задачи")
            and not find_entities_in_text(entity_catalog or {}, "users", request)
        )
    ):
        return "bitrix_my_tasks"
    if "задач" in text:
        return "bitrix_task_search"
    return None


def normalize_command_arguments(
    tool_name: str,
    request: str,
    arguments: dict[str, Any],
    *,
    now: datetime | None = None,
    task: Any | None = None,
    entity_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply orchestrator defaults before JSON-schema validation."""

    current = (now or datetime.now(MOSCOW_TZ)).astimezone(MOSCOW_TZ)
    if tool_name == "bitrix_warehouse_search":
        return _warehouse_semantics(request, arguments, entity_catalog)
    if tool_name in {"bitrix_my_tasks", "bitrix_task_search"}:
        return _task_read_semantics(
            tool_name, request, arguments, task=task, entity_catalog=entity_catalog
        )
    if tool_name == "task_create_draft":
        return _task_create_semantics(
            arguments, current, task=task, entity_catalog=entity_catalog
        )
    if tool_name == "task_close_draft":
        result = _task_close_semantics(request, arguments)
        event = task.context.get("task_close_event") if task is not None else None
        if (
            task is not None
            and task.source == "task_close_direct_control"
            and task.context.get("orchestrator_internal_event") is True
            and isinstance(event, dict)
        ):
            result.update(
                {
                    "task_id": event.get("task_id"),
                    "task_title": event.get("task_title"),
                    "task_points": list(event.get("task_points") or []),
                    "source_task_description_empty": bool(event.get("source_task_description_empty")),
                    "already_closed": True,
                    "close_now": False,
                    "overall_status": "unconfirmed",
                }
            )
            if event.get("task_results"):
                result["completion_summary"] = "\n".join(str(item) for item in event["task_results"])
            result.setdefault("unconfirmed_items", ["Результат закрытия задачи не подтверждён пользователем."])
            result.setdefault("missing_fields", ["Подтвердите фактический результат выполнения задачи."])
        return result
    if tool_name == "task_close_confirm":
        result = dict(arguments)
        if (
            task is not None
            and task.source == "task_close_direct_control"
            and task.context.get("orchestrator_internal_event") is True
        ):
            result["mode"] = str(task.context.get("task_close_confirmation_mode") or "auto_unconfirmed")
        else:
            result.setdefault("mode", "user_confirm")
        return result
    if tool_name == "calendar_event_draft":
        return _calendar_semantics(arguments, current, task=task)
    if tool_name == "portal_search":
        return _portal_search_semantics(request, arguments)
    if tool_name == "bitrix_project_search":
        return _project_search_semantics(request, arguments, entity_catalog)
    if tool_name == "project_create_draft":
        return _project_create_semantics(arguments, task=task)
    if tool_name == "task_close_control_update":
        return _task_close_control_semantics(request, arguments, entity_catalog)
    if tool_name == "task_close_report_incident":
        return _task_close_report_incident_semantics(arguments)
    return dict(arguments)


def normalize_plan(
    plan: Any,
    *,
    task: Any,
    constraints: dict[str, Any],
    now: datetime | None = None,
    entity_catalog: dict[str, Any] | None = None,
) -> Any:
    """Normalize and validate structured commands before any specialist call."""

    if getattr(plan, "state", None) != "EXECUTE":
        return plan
    current = (now or datetime.now(MOSCOW_TZ)).astimezone(MOSCOW_TZ)
    if entity_catalog is None:
        entity_catalog = (
            task.context.get("orchestrator_entity_catalog")
            if isinstance(getattr(task, "context", None), dict)
            and isinstance(task.context.get("orchestrator_entity_catalog"), dict)
            else {}
        )
    required_tool = ""
    if (
        getattr(task, "source", "") == "task_close_direct_control"
        and isinstance(getattr(task, "context", None), dict)
        and task.context.get("orchestrator_internal_event") is True
    ):
        required_tool = str(task.context.get("orchestrator_required_tool") or "").strip()
        if not required_tool or len(plan.subtasks) != 1:
            raise SemanticPolicyViolation("INTERNAL_EVENT_COMMAND_INVALID")
    normalized_subtasks = []
    for subtask in plan.subtasks:
        command = subtask.structured_command
        if required_tool and (
            subtask.specialist_id != "bitrix24"
            or command is None
            or command.tool_name != required_tool
        ):
            raise SemanticPolicyViolation("INTERNAL_EVENT_COMMAND_INVALID")
        if subtask.specialist_id != "bitrix24" or command is None:
            normalized_subtasks.append(subtask)
            continue
        semantic_request = task.request if len(plan.subtasks) == 1 else subtask.request
        expected = required_tool or _expected_tool(semantic_request, entity_catalog)
        if expected and command.tool_name != expected:
            raise SemanticPolicyViolation("SEMANTIC_TOOL_MISMATCH")
        arguments = normalize_command_arguments(
            command.tool_name,
            semantic_request,
            command.arguments,
            now=current,
            task=task,
            entity_catalog=entity_catalog,
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
