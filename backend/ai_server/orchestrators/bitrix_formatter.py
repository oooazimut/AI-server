"""Deterministic user-facing rendering owned by the orchestrator.

This is intentionally independent from every specialist model. It is the sole
user-facing Bitrix formatter and is owned by the orchestrator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from ai_server.models import ModelUsageRecord, ToolResult
from ai_server.orchestrators.draft_confirmation import draft_confirmation_phrase
from ai_server.utils import compact_text


@dataclass(frozen=True)
class BitrixFormattedResult:
    status: str
    answer: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


WAREHOUSE_RESPONSE_ITEM_LIMIT = 50


def _direct_task_create_response(
    *,
    agent_id: str,
    tool_results: list[ToolResult],
    portal_base_url: str = "",
) -> BitrixFormattedResult | None:
    successful_results = [result for result in tool_results if result.status == "ok"]
    warehouse_results = [result for result in successful_results if result.tool == "bitrix_warehouse_search"]
    if len(warehouse_results) > 1 and len(warehouse_results) == len(successful_results):
        limited_results, omitted = _limit_multi_warehouse_results(
            warehouse_results,
            total_limit=WAREHOUSE_RESPONSE_ITEM_LIMIT,
        )
        parts = [
            _format_warehouse_answer(result.data, portal_base_url=portal_base_url)
            for result in limited_results
        ]
        if omitted:
            parts.append(
                f"Ещё {omitted} {_ru_warehouse_word(omitted)} не поместились в общий предел "
                f"{WAREHOUSE_RESPONSE_ITEM_LIMIT} позиций. Их можно запросить продолжением."
            )
        return BitrixFormattedResult(
            status="completed",
            answer="\n\n".join(parts),
            model_usage=_local_model_usage(agent_id, "warehouse_multi_response"),
        )
    for result in reversed(tool_results):
        if result.status == "denied" and str(result.data.get("reason") or "") in {
            "ACTIVE_DRAFT_CONFLICT",
            "ACTIVE_DRAFT_IN_PROGRESS",
        }:
            return BitrixFormattedResult(
                status="needs_clarification",
                answer=str(result.data.get("answer") or result.error or "Уточните действие с активным черновиком."),
                model_usage=_local_model_usage(agent_id, "active_draft_conflict_response"),
            )
        if result.status == "denied" and result.tool in {
            "bitrix_warehouse_search",
            "bitrix_my_tasks",
            "bitrix_task_search",
            "bitrix_project_search",
        }:
            return BitrixFormattedResult(
                status="completed",
                answer=_format_read_denied_answer(result.data, result.error),
                model_usage=_local_model_usage(agent_id, "read_authorization_response"),
            )
        if result.status != "ok":
            continue
        if result.tool == "bitrix_warehouse_search":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_warehouse_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "warehouse_response"),
            )
        if result.tool == "bitrix_my_tasks":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_my_tasks_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "my_tasks_response"),
            )
        if result.tool == "bitrix_task_search":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_task_search_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "task_search_response"),
            )
        if result.tool == "bitrix_project_search":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_project_search_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "project_search_response"),
            )
        if result.tool == "portal_search":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_portal_search_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "portal_search_response"),
            )
        if result.tool == "project_create_draft":
            return BitrixFormattedResult(
                status="needs_human",
                answer=_format_project_create_draft_answer(result.data),
                model_usage=_local_model_usage(agent_id, "project_create_draft_response"),
            )
        if result.tool == "project_create_confirm":
            confirm_data = result.data if isinstance(result.data, dict) else {}
            has_followup_task = isinstance(confirm_data.get("followup_task_draft"), dict)
            return BitrixFormattedResult(
                status="needs_human" if has_followup_task else "completed",
                answer=_format_project_create_confirm_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "project_create_confirm_response"),
            )
        if result.tool == "project_create_discard":
            data = result.data if isinstance(result.data, dict) else {}
            return BitrixFormattedResult(
                status="completed",
                answer=(
                    "Черновик проекта и связанной задачи удалён."
                    if data.get("linked_task")
                    else "Черновик проекта удалён."
                ),
                model_usage=_local_model_usage(agent_id, "project_create_discard_response"),
            )
        if result.tool == "task_create_draft":
            return BitrixFormattedResult(
                status="needs_human",
                answer=_format_task_create_draft_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_create_draft_response"),
            )
        if result.tool == "task_create_confirm":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_task_create_confirm_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "task_create_confirm_response"),
            )
        if result.tool == "task_draft_discard":
            return BitrixFormattedResult(
                status="completed",
                answer="Черновик задачи удалён.",
                model_usage=_local_model_usage(agent_id, "task_draft_discard_response"),
            )
        if result.tool == "task_close_draft":
            return BitrixFormattedResult(
                status="needs_human",
                answer=_format_task_close_draft_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_close_draft_response"),
            )
        if result.tool == "task_close_confirm":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_task_close_confirm_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "task_close_confirm_response"),
            )
        if result.tool == "task_close_discard":
            return BitrixFormattedResult(
                status="completed",
                answer="Черновик закрытия задачи удалён.",
                model_usage=_local_model_usage(agent_id, "task_close_discard_response"),
            )
        if result.tool == "task_close_report_incident":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_task_close_report_incident_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_close_report_incident_response"),
            )
        if result.tool == "task_close_control_get":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_task_close_control_get_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_close_control_get_response"),
            )
        if result.tool == "task_close_control_update":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_task_close_control_update_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_close_control_update_response"),
            )
        if result.tool == "calendar_event_draft":
            return BitrixFormattedResult(
                status="needs_human",
                answer=_format_calendar_event_draft_answer(result.data),
                model_usage=_local_model_usage(agent_id, "calendar_event_draft_response"),
            )
        if result.tool == "calendar_event_confirm":
            return BitrixFormattedResult(
                status="completed",
                answer=_format_calendar_event_confirm_answer(result.data),
                model_usage=_local_model_usage(agent_id, "calendar_event_confirm_response"),
            )
        if result.tool == "calendar_event_discard":
            return BitrixFormattedResult(
                status="completed",
                answer="Черновик события календаря удалён.",
                model_usage=_local_model_usage(agent_id, "calendar_event_discard_response"),
            )
    return None


def _limit_multi_warehouse_results(
    results: list[ToolResult],
    *,
    total_limit: int,
) -> tuple[list[ToolResult], int]:
    capacities = [
        len((result.data.get("products") or {}).get("items") or [])
        if isinstance(result.data, dict) and isinstance(result.data.get("products"), dict)
        else 0
        for result in results
    ]
    allocations = _balanced_allocations(capacities, max(1, total_limit))
    limited: list[ToolResult] = []
    omitted = 0
    for result, capacity, allowed in zip(results, capacities, allocations, strict=True):
        if capacity > 0 and allowed <= 0:
            omitted += 1
            continue
        if capacity <= allowed or not isinstance(result.data, dict):
            limited.append(result)
            continue
        data = dict(result.data)
        products = dict(data.get("products") or {})
        items = list(products.get("items") or [])[:allowed]
        offset = _int_value(products.get("offset")) or 0
        total = (
            _int_value(products.get("available_items_with_names"))
            or _int_value(products.get("available_items_seen"))
            or capacity
        )
        products.update(
            {
                "items": items,
                "limit": allowed,
                "has_more": offset + len(items) < total,
            }
        )
        data["products"] = products
        limited.append(result.model_copy(update={"data": data}))
    return limited, omitted


def _balanced_allocations(capacities: list[int], total_limit: int) -> list[int]:
    allocations = [0] * len(capacities)
    remaining = max(0, total_limit)
    active = [index for index, capacity in enumerate(capacities) if capacity > 0]
    while active and remaining > 0:
        share, extra = divmod(remaining, len(active))
        if share == 0:
            share = 1
            extra = 0
        spent = 0
        next_active: list[int] = []
        for position, index in enumerate(active):
            quota = share + (1 if position < extra else 0)
            take = min(capacities[index] - allocations[index], quota, remaining - spent)
            allocations[index] += max(0, take)
            spent += max(0, take)
            if allocations[index] < capacities[index]:
                next_active.append(index)
            if spent >= remaining:
                break
        if spent <= 0:
            break
        remaining -= spent
        active = next_active
    return allocations


def _ru_warehouse_word(value: int) -> str:
    remainder_100 = value % 100
    remainder_10 = value % 10
    if 11 <= remainder_100 <= 14:
        return "складов"
    if remainder_10 == 1:
        return "склад"
    if 2 <= remainder_10 <= 4:
        return "склада"
    return "складов"


def direct_tool_results_response(
    *,
    agent_id: str,
    tool_results: list[ToolResult],
    portal_base_url: str = "",
    command_arguments: dict[str, Any] | None = None,
) -> BitrixFormattedResult:
    """Render one exact orchestrator command without another Bitrix model call."""

    direct = _direct_task_create_response(
        agent_id=agent_id,
        tool_results=tool_results,
        portal_base_url=portal_base_url,
    )
    if direct is not None:
        return direct
    result = tool_results[-1] if tool_results else None
    if result is None:
        return BitrixFormattedResult(
            status="failed",
            answer="Bitrix не вернул результат выполнения команды.",
            model_usage=_local_model_usage(agent_id, "structured_command_empty_result"),
        )
    if result.status in {"error", "not_configured"}:
        status = "failed"
    elif result.status in {"invalid_tool_call", "contract_violation"}:
        status = "needs_clarification"
    else:
        status = "completed"
    return BitrixFormattedResult(
        status=status,
        answer=_format_generic_tool_answer(result, command_arguments=command_arguments or {}),
        model_usage=_local_model_usage(agent_id, "structured_command_generic_response"),
    )


def _format_generic_tool_answer(result: ToolResult, *, command_arguments: dict[str, Any]) -> str:
    data = result.data if isinstance(result.data, dict) else {}
    if result.status != "ok":
        return _format_read_denied_answer(data, result.error)
    summary = _text(data.get("summary")) or _text(command_arguments.get("summary"))
    if result.tool == "bitrix_api":
        method = _text(data.get("method")) or _text(command_arguments.get("method")) or "Bitrix API"
        payload = data.get("result")
        count = len(payload) if isinstance(payload, list) else None
        if count is not None:
            return f"Bitrix выполнил {method}. Получено записей: {count}."
        return f"Bitrix выполнил {method}."
    messages = {
    }
    return messages.get(result.tool) or summary or f"Команда {result.tool} выполнена."


def _format_read_denied_answer(data: dict[str, Any], error: str | None) -> str:
    authorization = data.get("authorization") if isinstance(data.get("authorization"), dict) else {}
    message = _text(authorization.get("message")) if authorization else ""
    if message:
        return message
    return _text(error) or "Для чтения данных Bitrix24 требуется авторизация."


def _format_warehouse_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    matches = data.get("matches") if isinstance(data.get("matches"), list) else []
    if not matches:
        query = _text(data.get("query"))
        return f"Склад по запросу «{query}» не найден." if query else "Склад не найден."
    store = matches[0] if isinstance(matches[0], dict) else {}
    store_title = _text(store.get("title")) or "склад"
    address = _text(store.get("address"))
    store_label = f"{store_title} ({address})" if address else store_title

    products = data.get("products") if isinstance(data.get("products"), dict) else {}
    if not products:
        return f"Найден склад: {store_label}."
    if products.get("status") != "ok":
        error = _text(products.get("error")) or _text(products.get("message")) or "остатки недоступны"
        return f"Склад найден: {store_label}. Не удалось получить остатки: {error}."

    items = products.get("items") if isinstance(products.get("items"), list) else []
    if not items:
        return f"На складе {store_label} положительных остатков не найдено."

    lines = [f"Остатки по складу {store_label}:"]
    offset = _int_value(products.get("offset")) or 0
    for index, item in enumerate(items, start=offset + 1):
        if not isinstance(item, dict):
            continue
        name = _text(item.get("product_name")) or "товар"
        title = _warehouse_product_link(name, _text(item.get("product_url")), portal_base_url=portal_base_url)
        amount = _format_stock_amount(item.get("amount"))
        suffix = f" — {amount} шт." if amount else ""
        lines.append(f"{index}. {title}{suffix}")

    total = _int_value(products.get("available_items_with_names")) or _int_value(products.get("available_items_seen"))
    shown = len(items)
    limit = _int_value(products.get("limit")) or shown
    if total:
        start = offset + 1
        end = offset + shown
        has_more = bool(products.get("has_more")) or end < total
        if offset == 0 and not has_more:
            lines.append(f"Показаны все {shown} {_ru_position_word(shown)} с положительным остатком.")
        elif offset == 0:
            lines.append(
                f"Показаны первые {shown} {_ru_position_word(shown)} из {total}. "
                "Остальные позиции есть; можно запросить следующие."
            )
        elif has_more:
            lines.append(
                f"Показаны позиции {start}-{end} из {total}. Остальные позиции есть; можно запросить следующие."
            )
        else:
            lines.append(f"Показаны позиции {start}-{end} из {total}. Остальных позиций нет.")
    elif offset == 0 and shown >= limit:
        lines.append(
            f"Показаны первые {shown} {_ru_position_word(shown)}. Если нужно, можно запросить следующие позиции."
        )
    return "\n".join(lines)


def _warehouse_product_link(name: str, url: str, *, portal_base_url: str = "") -> str:
    if not url:
        return name
    if url.startswith("http://") or url.startswith("https://"):
        resolved = url
    elif url.startswith("/") and portal_base_url:
        resolved = portal_base_url.rstrip("/") + url
    else:
        return name
    return f"[URL={resolved}]{name}[/URL]"


def _format_my_tasks_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    items = data.get("items") if isinstance(data.get("items"), list) else []
    status = _text(data.get("status")) or "open"
    title = {
        "open": "Мои открытые задачи:",
        "closed": "Мои завершённые задачи:",
        "all": "Мои задачи:",
    }.get(status, "Мои задачи:")
    if not items:
        return title.replace(":", " не найдены.")

    lines = [title]
    offset = _int_value(data.get("offset")) or 0
    for index, item in enumerate(items, start=offset + 1):
        if not isinstance(item, dict):
            continue
        task_title = _text(item.get("title")) or "задача"
        task_id = _text(item.get("id"))
        label = _task_link(task_title, task_id, portal_base_url=portal_base_url)
        deadline = _text(item.get("deadline_label")) or "без срока"
        status_label = _text(item.get("status_label"))
        details = [f"срок: {deadline}"]
        if status_label:
            details.append(status_label)
        responsible = _text(item.get("responsible_label"))
        creator = _text(item.get("creator_label"))
        if responsible:
            details.append(f"исполнитель: {responsible}")
        if creator:
            details.append(f"постановщик: {creator}")
        lines.append(f"{index}. {label} — {', '.join(details)}")

    shown = len(items)
    total = _int_value(data.get("total"))
    if total and total > offset + shown:
        if offset == 0:
            lines.append(f"Показаны первые {shown} задач из {total}. Если нужно, можно запросить следующие.")
        else:
            lines.append(f"Показаны задачи {offset + 1}-{offset + shown} из {total}.")
    return "\n".join(lines)


def _format_task_search_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    mode = _text(data.get("mode")) or "list"
    if mode == "detail":
        item = data.get("item") if isinstance(data.get("item"), dict) else None
        if not item:
            return "Задача не найдена."
        title = _text(item.get("title")) or "задача"
        label = _task_link(title, _text(item.get("id")), portal_base_url=portal_base_url)
        details = _task_details(item, include_roles=False)
        if not details:
            return f"Задача найдена: {label}."
        return f"Задача найдена: {label}.\n" + "\n".join(details)

    project_query = _text(data.get("project_query"))
    if data.get("project_not_found"):
        return f"Проект «{project_query}» не найден." if project_query else "Проект не найден."

    items = data.get("items") if isinstance(data.get("items"), list) else []
    title = _task_search_title(data)
    if not items:
        return title.replace(":", " не найдены.")

    lines = [title]
    offset = _int_value(data.get("offset")) or 0
    for index, item in enumerate(items, start=offset + 1):
        if not isinstance(item, dict):
            continue
        task_title = _text(item.get("title")) or "задача"
        label = _task_link(task_title, _text(item.get("id")), portal_base_url=portal_base_url)
        details = _task_details(item, include_roles=False)
        suffix = f" — {', '.join(details)}" if details else ""
        lines.append(f"{index}. {label}{suffix}")

    shown = len(items)
    total = _int_value(data.get("total"))
    if total and total > offset + shown:
        if offset == 0:
            lines.append(f"Показаны первые {shown} задач из {total}. Если нужно, можно запросить следующие.")
        else:
            lines.append(f"Показаны задачи {offset + 1}-{offset + shown} из {total}.")
    return "\n".join(lines)


def _format_project_search_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    query = _text(data.get("query"))
    items = data.get("items") if isinstance(data.get("items"), list) else []
    if not items:
        return f"Проект «{query}» не найден." if query else "Проект не найден."
    lines = ["Найденные проекты:" if len(items) > 1 else "Найден проект:"]
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name")) or "проект"
        lines.append(f"{index}. {_project_link(name, _text(item.get('id')), portal_base_url=portal_base_url)}")
    return "\n".join(lines)


def _format_portal_search_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    query = _text(data.get("query"))
    scope = _text(data.get("scope")) or "documents"
    items = data.get("results") if isinstance(data.get("results"), list) else []
    title = _portal_search_title(scope=scope, query=query)
    if not items:
        return title.replace(":", " не найдены.")

    offset = _int_value(data.get("offset")) or 0
    lines = [title]
    for index, item in enumerate(items, start=offset + 1):
        if not isinstance(item, dict):
            continue
        label = _portal_item_link(
            _text(item.get("title")) or "документ",
            _text(item.get("url")),
            portal_base_url=portal_base_url,
        )
        lines.append(f"{index}. {label}")
        snippet = _portal_snippet(_text(item.get("body")), query=query)
        if snippet:
            lines.append(f"   Фрагмент: {snippet}")

    shown = len(items)
    total = _int_value(data.get("total")) or shown
    range_start = _int_value(data.get("range_start")) or (offset + 1 if shown else 0)
    range_end = _int_value(data.get("range_end")) or (offset + shown)
    remaining = _int_value(data.get("remaining")) or max(0, total - range_end)
    pages = _int_value(data.get("pages")) or 1
    if total:
        lines.append(
            f"Показаны результаты {range_start}-{range_end} из {total}. Осталось: {remaining}. Страниц: {pages}."
        )
    if bool(data.get("has_more")):
        lines.append("Чтобы продолжить тот же поиск, попросите показать следующие результаты.")
    return "\n".join(lines)


def _draft_confirmation_suffix(data: dict[str, Any]) -> str:
    candidates: list[dict[str, Any]] = []
    for key in ("draft", "params", "pending_task_draft"):
        value = data.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    candidates.append(data)
    for item in candidates:
        draft_type = _text(item.get("_draft_type"))
        if draft_type:
            return f"\n\nДля подтверждения отправьте фразу: «{draft_confirmation_phrase(draft_type)}»."
    return f"\n\nДля подтверждения отправьте фразу: «{draft_confirmation_phrase('task_create')}»."


def _format_project_create_draft_answer(data: dict[str, Any]) -> str:
    preview = data.get("preview") if isinstance(data.get("preview"), dict) else {}
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    method_params = params.get("params") if isinstance(params.get("params"), dict) else {}
    fields = method_params.get("fields") if isinstance(method_params.get("fields"), dict) else {}
    name = _text(preview.get("name")) or _text(fields.get("NAME")) or "проект"
    project_type = _text(preview.get("type")) or "проект"
    visibility = _text(preview.get("visibility"))
    description = _text(preview.get("description")) or _text(fields.get("DESCRIPTION"))

    lines = [
        "Черновик проекта:",
        f"Название: {name}",
        f"Тип: {project_type}",
    ]
    if visibility:
        lines.append(f"Открытость: {visibility}")
    if description:
        lines.append(f"Описание: {description}")
    lines.extend(["", "Черновик действует 15 минут."])
    return "\n".join(lines) + _draft_confirmation_suffix({**data, "_draft_type": "project_create"})


def _format_project_create_confirm_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    draft = data.get("draft") if isinstance(data.get("draft"), dict) else {}
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    fields = params.get("fields") if isinstance(params.get("fields"), dict) else {}
    draft_params = draft.get("params") if isinstance(draft.get("params"), dict) else {}
    draft_fields = draft_params.get("fields") if isinstance(draft_params.get("fields"), dict) else {}
    name = _text(fields.get("NAME")) or _text(draft_fields.get("NAME")) or "проект"
    project_id = _created_project_id(data.get("result"))
    label = _project_link(name, project_id, portal_base_url=portal_base_url)
    answer = f"Проект создан: {label}."
    followup = data.get("followup_task_draft") if isinstance(data.get("followup_task_draft"), dict) else {}
    if followup:
        answer = f"{answer}\n\n{_format_task_create_draft_answer(followup)}"
    return answer


def _portal_search_title(*, scope: str, query: str) -> str:
    scope_label = {
        "documents": "Документы",
        "files": "Файлы",
        "projects": "Проекты",
        "catalog": "Каталог",
        "stores": "Склады",
        "products": "Товары",
        "stock": "Остатки",
    }.get(scope, "Результаты поиска")
    return f"{scope_label} по запросу «{query}»:"


def _task_search_title(data: dict[str, Any]) -> str:
    project = data.get("project") if isinstance(data.get("project"), dict) else {}
    project_name = _text(project.get("name")) if isinstance(project, dict) else ""
    scope_label = _text(data.get("scope_label")) or "задачи"
    status = _text(data.get("status")) or "active"
    query = _text(data.get("query"))
    comment_query = _text(data.get("comment_query"))
    ai_close_problem_type = _text(data.get("ai_close_problem_type"))
    if project_name:
        base = f"Задачи в проекте {project_name}"
    elif status == "overdue":
        base = f"Просроченные {scope_label}"
    elif status == "closed":
        base = f"Завершённые {scope_label}"
    elif status == "all":
        base = scope_label
    else:
        base = f"Активные {scope_label}"
    if ai_close_problem_type:
        base = f"{base} {_ai_close_problem_filter_label(ai_close_problem_type)}"
    if query:
        base = f"{base} по запросу «{query}»"
    if comment_query:
        base = f"{base} с комментарием «{comment_query}»"
    return _capitalize_first(base) + ":"


def _task_details(item: dict[str, Any], *, include_roles: bool) -> list[str]:
    details = [f"срок: {_text(item.get('deadline_label')) or 'без срока'}"]
    status_label = _text(item.get("status_label"))
    if status_label:
        details.append(status_label)
    responsible = _text(item.get("responsible_label"))
    creator = _text(item.get("creator_label"))
    if responsible:
        details.append(f"исполнитель: {responsible}")
    if creator:
        details.append(f"постановщик: {creator}")
    if item.get("ai_close_incomplete"):
        details.append(f"AI-закрытие: {_ai_close_problem_detail_label(item.get('ai_close_problem_types'))}")
    roles = item.get("roles") if isinstance(item.get("roles"), list) else []
    roles_label = ", ".join(_text(role) for role in roles if _text(role))
    if include_roles and roles_label:
        details.append(roles_label)
    snippets = item.get("comment_snippets") if isinstance(item.get("comment_snippets"), list) else []
    first_snippet = next((_text(snippet) for snippet in snippets if _text(snippet)), "")
    if first_snippet:
        details.append(f"комментарий: {first_snippet}")
    return details


def _ai_close_problem_filter_label(problem_type: str) -> str:
    return {
        "not_done": "с невыполненными пунктами",
        "unconfirmed": "с неподтверждённым результатом",
        "any": "с неполным AI-закрытием",
    }.get(problem_type, "с неполным AI-закрытием")


def _ai_close_problem_detail_label(value: object) -> str:
    problem_types = set(_text_items(value))
    labels = []
    if "not_done" in problem_types:
        labels.append("есть невыполненные пункты")
    if "unconfirmed" in problem_types:
        labels.append("результат не подтверждён")
    return ", ".join(labels) or "неполное/неподтверждённое"


def _capitalize_first(value: str) -> str:
    if not value:
        return value
    return value[:1].upper() + value[1:]


def _task_link(title: str, task_id: str, *, portal_base_url: str = "") -> str:
    if not portal_base_url or not task_id:
        return title
    url = f"{portal_base_url.rstrip('/')}/company/personal/user/0/tasks/task/view/{task_id}/"
    return f"[URL={url}]{title}[/URL]"


def _project_link(title: str, project_id: str, *, portal_base_url: str = "") -> str:
    if not portal_base_url or not project_id:
        return title
    url = f"{portal_base_url.rstrip('/')}/workgroups/group/{project_id}/"
    return f"[URL={url}]{title}[/URL]"


def _portal_item_link(title: str, url: str, *, portal_base_url: str = "") -> str:
    if not url:
        return title
    if url.startswith("http://") or url.startswith("https://"):
        resolved = url
    elif url.startswith("/") and portal_base_url:
        resolved = portal_base_url.rstrip("/") + url
    else:
        return title
    return f"[URL={resolved}]{title}[/URL]"


def _portal_snippet(body: str, *, query: str, max_length: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", body).strip()
    if not cleaned:
        return ""
    if len(cleaned) <= max_length:
        return cleaned
    terms = [term for term in re.findall(r"[\w#№.-]+", query.lower().replace("ё", "е")) if len(term) > 1]
    lowered = cleaned.lower().replace("ё", "е")
    positions = [lowered.find(term) for term in terms if term and lowered.find(term) >= 0]
    if positions:
        position = min(positions)
        start = max(0, position - max_length // 3)
        end = min(len(cleaned), start + max_length)
        start = max(0, end - max_length)
        snippet = cleaned[start:end].strip()
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(cleaned) else ""
        return f"{prefix}{snippet}{suffix}"
    return cleaned[: max_length - 3].rstrip() + "..."


def _format_stock_amount(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        amount = Decimal(text.replace(",", "."))
    except (InvalidOperation, ValueError):
        return text
    if amount == amount.to_integral_value():
        return str(int(amount))
    return format(amount.normalize(), "f").rstrip("0").rstrip(".")


def _ru_position_word(count: int) -> str:
    count = abs(count)
    if count % 100 in (11, 12, 13, 14):
        return "позиций"
    last = count % 10
    if last == 1:
        return "позиция"
    if last in (2, 3, 4):
        return "позиции"
    return "позиций"


def _format_task_create_draft_answer(data: dict[str, Any]) -> str:
    if data.get("requires_project_creation"):
        return _format_task_create_requires_project_answer(data)
    preview = data.get("preview") if isinstance(data.get("preview"), dict) else {}
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    fields = params.get("fields") if isinstance(params.get("fields"), dict) else {}
    title = _text(preview.get("title")) or _text(fields.get("TITLE")) or "задача"
    responsible = _text(preview.get("responsible")) or "указанный сотрудник"
    project = _text(preview.get("project"))
    deadline = _text(preview.get("deadline")) or _deadline_label_from_fields(fields)
    description = _clean_task_description(
        _text(preview.get("description")) or _text(fields.get("DESCRIPTION")) or f"Краткое содержание: {title}"
    )
    lines = [
        "Черновик задачи:",
        f"Название: {title}",
        f"Ответственный: {responsible}",
    ]
    if project:
        lines.append(f"Проект: {project}")
    lines.extend(
        [
            f"Срок: {deadline}",
            f"Описание: {description}",
            "",
            "Черновик действует 15 минут.",
        ]
    )
    return "\n".join(lines) + _draft_confirmation_suffix({**data, "_draft_type": "task_create"})


def _format_task_create_requires_project_answer(data: dict[str, Any]) -> str:
    project_name = _text(data.get("missing_project_name")) or "личный проект"
    project_draft = data.get("project_draft") if isinstance(data.get("project_draft"), dict) else {}
    project_preview = project_draft.get("preview") if isinstance(project_draft.get("preview"), dict) else {}
    task_draft = data.get("pending_task_draft") if isinstance(data.get("pending_task_draft"), dict) else {}
    task_preview = task_draft.get("preview") if isinstance(task_draft.get("preview"), dict) else {}
    task_params = task_draft.get("params") if isinstance(task_draft.get("params"), dict) else {}
    task_fields = task_params.get("fields") if isinstance(task_params.get("fields"), dict) else {}

    title = _text(task_preview.get("title")) or _text(task_fields.get("TITLE")) or "задача"
    responsible = _text(task_preview.get("responsible")) or "указанный сотрудник"
    deadline = _text(task_preview.get("deadline")) or _deadline_label_from_fields(task_fields)
    description = _clean_task_description(
        _text(task_preview.get("description"))
        or _text(task_fields.get("DESCRIPTION"))
        or f"Краткое содержание: {title}"
    )
    project_type = _text(project_preview.get("type")) or "личный проект"
    visibility = _text(project_preview.get("visibility"))
    project_description = _text(project_preview.get("description"))

    lines = [
        f"Личный проект «{project_name}» не найден.",
        "Сначала нужно создать проект, затем я подготовлю задачу в нём.",
        "",
        "Черновик проекта:",
        f"Название: {project_name}",
        f"Тип: {project_type}",
    ]
    if visibility:
        lines.append(f"Открытость: {visibility}")
    if project_description:
        lines.append(f"Описание: {project_description}")
    lines.extend(
        [
            "",
            "После создания проекта будет подготовлен черновик задачи:",
            f"Название: {title}",
            f"Ответственный: {responsible}",
            f"Срок: {deadline}",
            f"Описание: {description}",
            "",
            "Черновик действует 15 минут.",
        ]
    )
    return "\n".join(lines)


def _format_task_create_confirm_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    fields = params.get("fields") if isinstance(params.get("fields"), dict) else {}
    title = _text(fields.get("TITLE")) or "задача"
    task_id = _created_task_id(data.get("result"))
    if portal_base_url and task_id:
        url = f"{portal_base_url.rstrip('/')}/company/personal/user/0/tasks/task/view/{task_id}/"
        return f"Задача создана: [URL={url}]{title}[/URL]."
    return f"Задача создана: {title}."


def _format_task_close_draft_answer(data: dict[str, Any]) -> str:
    if data.get("status") == "active_draft_conflict":
        return _format_task_close_draft_conflict_answer(data)
    preview = data.get("preview") if isinstance(data.get("preview"), dict) else {}
    draft = data.get("draft") if isinstance(data.get("draft"), dict) else {}
    answer = format_task_close_report(draft, preview=preview, include_instruction=True)
    return answer + _draft_confirmation_suffix({**data, "_draft_type": "task_close"})


def format_task_close_report(
    draft: dict[str, Any],
    *,
    preview: dict[str, Any] | None = None,
    heading: str | None = None,
    include_instruction: bool = False,
) -> str:
    """Render the four-block task-close template owned by the orchestrator."""

    preview = preview if isinstance(preview, dict) else {}
    task_id = _text(draft.get("task_id"))
    task_title = _text(preview.get("task_title")) or _text(draft.get("task_title")) or _task_fallback_title(task_id)
    raw_result_text = (
        _text(preview.get("completion_summary"))
        or _text(draft.get("completion_summary"))
        or _text(draft.get("result_text"))
    )
    result_text = "" if _task_close_is_placeholder_summary(raw_result_text) else raw_result_text
    task_points = _text_items(preview.get("task_points") or draft.get("task_points"))
    source_task_description_empty = bool(
        preview.get("source_task_description_empty") or draft.get("source_task_description_empty")
    )
    equipment = _text(preview.get("equipment_consumables")) or _text(draft.get("equipment_consumables"))
    additional_info = _text(preview.get("additional_info")) or _text(draft.get("additional_info"))
    overall_status = _text(preview.get("overall_status")) or _text(draft.get("overall_status"))
    overall = _text(preview.get("overall_status_label")) or _text(draft.get("overall_status_label"))
    if overall.casefold() == "результат не подтверждён" and not result_text:
        overall = ""
    not_done = _text_items(preview.get("not_done_items") or draft.get("not_done_items"))
    unconfirmed = _text_items(preview.get("unconfirmed_items") or draft.get("unconfirmed_items"))
    status_reasons = _text_items(preview.get("status_reasons") or draft.get("status_reasons"))
    unresolved = _text_items(preview.get("unresolved_items") or draft.get("unresolved_items"))
    if not unconfirmed and unresolved:
        unconfirmed = unresolved
    unconfirmed = _task_close_visible_unconfirmed(unconfirmed, task_points=task_points, result_text=result_text)
    lines = [heading or f"Черновик #{task_id or '?'}: {task_title}"]
    lines.append("")
    lines.append("1. Выполняемые работы")
    if source_task_description_empty:
        result_items = _task_close_display_items(result_text)
        if result_items:
            lines.extend(f"1.{index} {item}" for index, item in enumerate(result_items, start=1))
            lines.append(f"1.{len(result_items) + 1} Еще работы - ... ???")
        else:
            lines.append("1.1 Еще работы - ... ???")
    else:
        if task_points:
            lines.extend(
                f"1.{index} {item} - {_task_close_point_status(item, result_text, not_done, unconfirmed)}"
                for index, item in enumerate(task_points, start=1)
            )
            lines.append(f"1.{len(task_points) + 1} Еще работы - ... ???")
        elif result_text:
            result_items = _task_close_display_items(result_text)
            lines.extend(f"1.{index} {item}" for index, item in enumerate(result_items, start=1))
            lines.append(f"1.{len(result_items) + 1} Еще работы - ... ???")
        else:
            lines.append("1.1 Еще работы - ... ???")
    lines.append("")
    lines.append("2. Использовано материалов, оборудование")
    if _task_close_no_equipment_value(equipment):
        lines.append("   (перечисли, что использовали, или укажи: [ВЫБРАНО: не использовалось])")
    else:
        lines.append("   (перечисли, что использовали, или укажи: не использовалось)")
        equipment_items = _task_close_display_items(equipment)
        if equipment_items:
            lines.extend(f"2.{index} {item}" for index, item in enumerate(equipment_items, start=1))
            lines.append(f"2.{len(equipment_items) + 1} Еще материалы - ... ???")
    lines.append("")
    lines.append("3. Статус выполнения работ")
    lines.append(f"   {_task_close_status_prompt(overall_status or overall)}")
    if status_reasons:
        for index, item in enumerate(status_reasons, start=1):
            lines.append(f"3.{index} причина: {item}")
        lines.append(f"3.{len(status_reasons) + 1} Еще причины невыполнения - ... ???")
    elif _task_close_status_needs_reason(overall_status or overall):
        lines.append("3.1 Еще причины невыполнения - ... ???")
    lines.append("")
    lines.append("4. Дополнительная информация")
    if _task_close_absent_value(additional_info):
        lines.append("   (если есть — укажи; если нет — укажи: [ВЫБРАНО: отсутствует])")
    else:
        lines.append("   (если есть — укажи; если нет — укажи: отсутствует)")
    additional_items = _task_close_display_items(additional_info)
    if additional_items and not _task_close_absent_value(additional_info):
        lines.extend(f"4.{index} {item}" for index, item in enumerate(additional_items, start=1))
        lines.append(f"4.{len(additional_items) + 1} Еще информация - ... ???")
    if include_instruction:
        lines.append("")
        lines.append("Внести изменения (укажите пункт или подпункт и нужную информацию) либо подтвердить закрытие.")
    return "\n".join(lines)


def format_task_close_result_text(
    *,
    completion_summary: str,
    task_points: list[str],
    equipment_consumables: str,
    additional_info: str,
    overall_status_label: str,
    not_done_items: list[str],
    unconfirmed_items: list[str],
    status_reasons: list[str],
) -> str:
    """Render the exact task result payload that Bitrix stores."""

    lines: list[str] = []
    if completion_summary:
        lines.append(completion_summary)
    if task_points:
        lines.extend(["", "Пункты задачи:", *(f"- {item}" for item in task_points)])
    if equipment_consumables:
        lines.extend(["", f"Оборудование, расходники: {equipment_consumables}"])
    if additional_info:
        lines.extend(["", f"Дополнительная информация: {additional_info}"])
    if overall_status_label:
        lines.extend(["", f"Общий итог: {overall_status_label}"])
    if status_reasons:
        lines.extend(["", "Причины неполного выполнения:", *(f"- {item}" for item in status_reasons)])
    if not_done_items or unconfirmed_items:
        if not_done_items and unconfirmed_items:
            close_status = "выполнена частично, неподтвержденная"
        elif not_done_items:
            close_status = "выполнена частично"
        else:
            close_status = "неподтвержденная"
        lines.extend(["", f"Статус AI-закрытия: {close_status}"])
    if not_done_items:
        lines.extend(["Невыполненные пункты:", *(f"- {item}" for item in not_done_items)])
    if unconfirmed_items:
        lines.extend(["Неподтверждённые пункты:", *(f"- {item}" for item in unconfirmed_items)])
    return "\n".join(lines).strip()


def _task_close_status_prompt(status: str) -> str:
    normalized = _task_close_normalized_status(status)
    options = [
        ("completed", "выполнено полностью"),
        ("partial", "выполнено частично"),
        ("not_done", "не выполнено"),
    ]
    rendered = [f"[ВЫБРАНО: {label}]" if normalized == key else label for key, label in options]
    return (
        f"({' / '.join(rendered)}; если выполнено частично или не выполнено — укажи причину неполного выполнения работ)"
    )


def _task_close_normalized_status(status: str) -> str:
    text = compact_text(status).casefold()
    if text in {"completed", "complete", "done"} or "полностью" in text:
        return "completed"
    if text in {"partial", "partially_done", "partly_done"} or "частич" in text:
        return "partial"
    if text in {"not_done", "not_completed", "not done"} or "не выполн" in text or "не сделан" in text:
        return "not_done"
    return ""


def _task_close_status_needs_reason(status: str) -> bool:
    return _task_close_normalized_status(status) in {"partial", "not_done"}


def _task_close_absent_value(value: str) -> bool:
    text = compact_text(value).casefold()
    return text in {"отсутствует", "нет", "не было", "нет дополнительной информации"}


def _task_close_no_equipment_value(value: str) -> bool:
    text = compact_text(value).casefold().strip(" .;:-")
    if not text:
        return False
    if text in {
        "не использовалось",
        "не использовались",
        "не использовали",
        "не применялось",
        "не применяли",
        "не было",
        "нет",
        "ничего",
        "без материалов",
        "без оборудования",
        "материалы не использовались",
        "оборудование не использовалось",
    }:
        return True
    return ("не использ" in text or "не примен" in text) and not any(char.isdigit() for char in text)


def _task_close_is_placeholder_summary(value: str) -> bool:
    text = compact_text(value)
    if not text:
        return False
    lowered = text.casefold()
    if "статус ai-закрытия" in lowered:
        return True
    if "результат выполнения не указан" in lowered:
        return True
    return "общий итог: результат не подтвержд" in lowered and (
        lowered.startswith("пункты задачи:") or "неподтверждённые пункты" in lowered
    )


def _task_close_visible_unconfirmed(
    values: list[str],
    *,
    task_points: list[str],
    result_text: str,
) -> list[str]:
    point_keys = {item.casefold() for item in task_points}
    visible: list[str] = []
    for value in values:
        key = value.casefold()
        if key in {"результат выполнения не указан", "result is not specified"}:
            continue
        if key in point_keys:
            if result_text and _task_close_result_explicitly_marks_unconfirmed(value, result_text):
                visible.append(value)
            continue
        visible.append(value)
    return visible


def _task_close_result_explicitly_marks_unconfirmed(point: str, result_text: str) -> bool:
    point_cf = point.casefold()
    for clause in re.split(r"(?:\r?\n)+|[.;]+|,(?=\s*\d+\.\d+)", result_text):
        clause_cf = clause.casefold()
        if not any(marker in clause_cf for marker in ("не подтверж", "неизвест", "не провер", "unconfirmed")):
            continue
        if _task_close_texts_overlap(point_cf, clause):
            return True
    return False


def _task_close_display_items(value: str) -> list[str]:
    text = compact_text(value)
    if not text:
        return []
    raw_parts = re.split(r"(?:\r?\n)+|(?:^|\s)(?:\d+[.)]|[-*•])\s+|[,;]+", text)
    parts = _unique_text_items(part.strip(" .;:-") for part in raw_parts)
    return parts[:12] if len(parts) > 1 else ([text] if text else [])


def _unique_text_items(values: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = compact_text(str(value or "")).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _task_close_point_status(
    point: str,
    result_text: str,
    not_done: list[str],
    unconfirmed: list[str],
) -> str:
    point_cf = point.casefold()
    for item in not_done:
        if _task_close_texts_overlap(point_cf, item):
            return "не выполнено"
    for item in unconfirmed:
        if _task_close_texts_overlap(point_cf, item):
            return "не подтверждено"
    completed_label = _task_close_point_completed_label(point, result_text)
    if completed_label:
        return completed_label
    if result_text:
        return "... ???"
    return "... ???"


def _task_close_point_completed_label(point: str, result_text: str) -> str:
    if not result_text:
        return ""
    point_cf = point.casefold()
    clauses = re.split(r"(?:\r?\n)+|[.;]+|,(?=\s*\d+\.\d+)", result_text)
    for clause in clauses:
        clause_cf = clause.casefold()
        if "не выполн" in clause_cf or "не подтверж" in clause_cf:
            continue
        if ("выполн" in clause_cf or "сделан" in clause_cf or "готов" in clause_cf) and _task_close_texts_overlap(
            point_cf, clause
        ):
            return "выполнено полностью" if "полностью" in clause_cf else "выполнено"
    return ""


def _task_close_texts_overlap(point_cf: str, value: str) -> bool:
    value_cf = value.casefold()
    if point_cf and point_cf in value_cf:
        return True
    words = [word for word in re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", point_cf) if len(word) >= 5]
    return bool(words and any(word in value_cf for word in words[:4]))


def _format_task_close_draft_conflict_answer(data: dict[str, Any]) -> str:
    draft = data.get("draft") if isinstance(data.get("draft"), dict) else {}
    incoming = data.get("incoming_draft") if isinstance(data.get("incoming_draft"), dict) else {}
    current_task_id = _text(draft.get("task_id"))
    requested_task_id = _text(incoming.get("task_id"))
    current_title = _text(draft.get("task_title")) or _task_fallback_title(current_task_id)
    requested_title = _text(incoming.get("task_title")) or _task_fallback_title(requested_task_id)
    return "\n".join(
        [
            "Уже есть активный черновик закрытия.",
            f"Текущий: {current_title}",
            f"Новая задача: {requested_title}",
            "",
            "Сначала выберите действие:",
            "1 — продолжить текущий черновик",
            "2 — закрыть текущую задачу как есть с пометкой",
            "3 — отменить текущий черновик и перейти к новой задаче",
        ]
    )


def _format_task_close_confirm_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    draft = data.get("draft") if isinstance(data.get("draft"), dict) else {}
    task_id = _text(data.get("task_id")) or _text(draft.get("task_id"))
    task_title = _text(data.get("task_title")) or _text(draft.get("task_title")) or _task_fallback_title(task_id)
    action = _text(data.get("action")) or _text(draft.get("action"))
    label = _task_link(task_title, task_id, portal_base_url=portal_base_url)
    not_done = _text_items(data.get("not_done_items") or draft.get("not_done_items"))
    unconfirmed = _text_items(data.get("unconfirmed_items") or draft.get("unconfirmed_items"))
    unresolved = _text_items(data.get("unresolved_items") or draft.get("unresolved_items"))
    if not unconfirmed and unresolved:
        unconfirmed = unresolved
    suffix = f" Статус: {_task_close_human_status(not_done, unconfirmed)}." if not_done or unconfirmed else ""
    if action == "approve":
        return f"Задача закрыта: {label}.{suffix}"
    return f"Задача отмечена выполненной: {label}.{suffix}"


def _format_task_close_report_incident_answer(data: dict[str, Any]) -> str:
    task_id = _text(data.get("task_id"))
    file_name = _text(data.get("file_name")) or "AI-close report"
    action = _text(data.get("action"))
    if action == "restore":
        return f"Файл отчёта AI-закрытия восстановлен в задаче #{task_id}: {file_name}."
    if action == "accept_missing":
        return (
            f"Принято: удаление отчёта AI-закрытия по задаче #{task_id} оставлено как есть, служебные данные очищены."
        )
    return f"Инцидент отчёта AI-закрытия по задаче #{task_id} обработан."


def _format_task_close_control_get_answer(data: dict[str, Any]) -> str:
    lines = [
        "Настройки контроля закрытия задач:",
        f"- Автозакрытие: {data.get('auto_close_time') or '20:00'}",
        f"- Запуск контроля: {data.get('control_enabled_from') or 'не задан'}",
    ]
    members = _task_close_control_member_lines(data.get("members"))
    if members:
        lines.append("- Сейчас включены:")
        lines.extend(members)
    else:
        lines.append("- Сейчас включены: нет")

    available = _task_close_control_available_lines(data)
    if available:
        lines.append("- Пользователи Bitrix:")
        lines.extend(available)
        if data.get("available_users_truncated"):
            lines.append(f"  Показаны первые {data.get('available_users_limit') or 100}; можно уточнить по имени.")
    elif data.get("user_lookup_status") == "not_configured":
        lines.append("- Пользователи Bitrix: список недоступен, Bitrix user client не настроен.")
    elif data.get("user_lookup_status") in {"failed", "partial"}:
        lines.append(f"- Пользователи Bitrix: не удалось получить полный список ({data.get('user_lookup_error')}).")
    return "\n".join(lines)


def _format_task_close_control_update_answer(data: dict[str, Any]) -> str:
    action = str(data.get("action") or "")
    action_labels = {
        "add_operator": "оператор добавлен",
        "remove_operator": "оператор удалён",
        "add_controlled_user": "контролируемый пользователь добавлен",
        "remove_controlled_user": "контролируемый пользователь удалён",
        "set_auto_close_time": "время автозакрытия обновлено",
        "set_control_enabled_from": "дата запуска контроля обновлена",
    }
    lines = [f"Готово: {action_labels.get(action, 'настройки обновлены')}."]
    lines.append(f"Сейчас включены: {_task_close_control_members_summary(data)}.")
    lines.append(f"Автозакрытие: {data.get('auto_close_time') or '20:00'}.")
    if data.get("control_enabled_from"):
        lines.append(f"Запуск контроля: {data['control_enabled_from']}.")
    return " ".join(lines)


def _task_close_control_member_lines(value: object, *, limit: int = 50) -> list[str]:
    if not isinstance(value, list):
        return []
    lines = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        lines.append(f"  {_task_close_control_user_label(item)}")
    if len(value) > limit:
        lines.append(f"  ... ещё {len(value) - limit}")
    return lines


def _task_close_control_available_lines(data: dict[str, Any], *, limit: int = 100) -> list[str]:
    users = data.get("available_users")
    if not isinstance(users, list):
        return []
    lines = []
    for item in users[:limit]:
        if not isinstance(item, dict):
            continue
        lines.append(f"  {_task_close_control_user_label(item)}")
    return lines


def _task_close_control_members_summary(data: dict[str, Any]) -> str:
    members = data.get("members")
    if isinstance(members, list) and members:
        labels = [_task_close_control_user_label(item) for item in members[:8] if isinstance(item, dict)]
        suffix = f"; ещё {len(members) - 8}" if len(members) > 8 else ""
        return "; ".join(labels) + suffix
    operator_ids = _ids_label(data.get("operator_user_ids"))
    controlled_ids = _ids_label(data.get("controlled_user_ids"))
    return f"операторы: {operator_ids}; контролируемые: {controlled_ids}"


def _task_close_control_user_label(item: dict[str, Any]) -> str:
    user_id = _text(item.get("user_id")) or "?"
    name = _text(item.get("name")) or f"Bitrix user #{user_id}"
    role = _task_close_control_role_label(item)
    return f"ID {user_id} — {name} — {role}"


def _task_close_control_role_label(item: dict[str, Any]) -> str:
    is_operator = bool(item.get("is_operator"))
    is_controlled = bool(item.get("is_controlled"))
    if is_operator and is_controlled:
        return "оператор, контролируемый"
    if is_operator:
        return "оператор"
    if is_controlled:
        return "контролируемый"
    return "можно добавить"


def _ids_label(value: object) -> str:
    if not isinstance(value, list) or not value:
        return "нет"
    return ", ".join(str(item) for item in value)


def _task_close_human_status(not_done_items: list[str], unconfirmed_items: list[str]) -> str:
    if not_done_items and unconfirmed_items:
        return "выполнена частично, неподтвержденная"
    if not_done_items:
        return "выполнена частично"
    return "неподтвержденная"


def _format_calendar_event_draft_answer(data: dict[str, Any]) -> str:
    preview = data.get("preview") if isinstance(data.get("preview"), dict) else {}
    draft = data.get("draft") if isinstance(data.get("draft"), dict) else {}
    params = draft.get("params") if isinstance(draft.get("params"), dict) else {}
    title = _text(preview.get("title")) or _text(draft.get("title")) or _text(params.get("name")) or "событие"
    start = _text(preview.get("start")) or _text(draft.get("start_iso")) or _text(params.get("from"))
    end = _text(preview.get("end")) or _text(draft.get("end_iso")) or _text(params.get("to"))
    description = (
        _text(preview.get("description")) or _text(draft.get("description")) or _text(params.get("description"))
    )
    participants = _text(preview.get("participants")) or "только текущий пользователь"
    reminder = _text(preview.get("reminder")) or "по настройкам календаря Bitrix"
    lines = [
        "Черновик события календаря:",
        f"Название: {title}",
        f"Начало: {start}",
        f"Окончание: {end}",
    ]
    if participants and not _is_default_calendar_participant_label(participants):
        lines.append(f"Участники: {participants}")
    if reminder and not _is_default_calendar_reminder_label(reminder):
        lines.append(f"Напоминание: {reminder}")
    if description and not _is_generic_calendar_description(description):
        lines.append(f"Описание: {description}")
    lines.append("")
    lines.append("Черновик действует 15 минут.")
    return "\n".join(lines) + _draft_confirmation_suffix({**data, "_draft_type": "calendar_event"})


def _format_calendar_event_confirm_answer(data: dict[str, Any]) -> str:
    title = _text(data.get("title"))
    draft = data.get("draft") if isinstance(data.get("draft"), dict) else {}
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    if not title:
        title = _text(draft.get("title")) or _text(params.get("name")) or "событие"
    start_label = _text(data.get("start_label"))
    if start_label:
        return f"Событие добавлено в календарь: {title}. Начало: {start_label}."
    return f"Событие добавлено в календарь: {title}."


def _task_fallback_title(task_id: str) -> str:
    return f"задача #{task_id}" if task_id else "указанная задача"


def _clean_task_description(value: str) -> str:
    text = _text(value)
    for prefix in ("Краткое содержание:", "Описание:"):
        if text.casefold().startswith(prefix.casefold()):
            return text[len(prefix) :].strip()
    return text


def _is_default_calendar_participant_label(value: str) -> bool:
    return _text(value).casefold() in {"только текущий пользователь", "текущий пользователь"}


def _is_default_calendar_reminder_label(value: str) -> bool:
    return _text(value).casefold() == "по настройкам календаря bitrix"


def _is_generic_calendar_description(value: str) -> bool:
    return _text(value).casefold() in {"напоминание", "событие", "calendar event", "reminder"}


def _created_task_id(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    candidates: list[object] = [value.get("id"), value.get("ID")]
    for key in ("task", "TASK", "result"):
        nested = value.get(key)
        if isinstance(nested, dict):
            candidates.extend([nested.get("id"), nested.get("ID")])
            nested_task = nested.get("task") or nested.get("TASK")
            if isinstance(nested_task, dict):
                candidates.extend([nested_task.get("id"), nested_task.get("ID")])
    for candidate in candidates:
        text = _text(candidate)
        if text:
            return text
    return ""


def _created_project_id(value: object) -> str:
    direct = _text(value)
    if direct and not isinstance(value, dict):
        return direct
    if not isinstance(value, dict):
        return ""
    candidates: list[object] = [value.get("id"), value.get("ID"), value.get("group_id"), value.get("GROUP_ID")]
    nested = value.get("result")
    if isinstance(nested, dict):
        candidates.extend([nested.get("id"), nested.get("ID"), nested.get("group_id"), nested.get("GROUP_ID")])
    else:
        candidates.append(nested)
    for candidate in candidates:
        text = _text(candidate)
        if text:
            return text
    return ""


def _deadline_label_from_fields(fields: dict[str, Any]) -> str:
    if fields.get("NO_DEADLINE"):
        return "без срока"
    deadline = _text(fields.get("DEADLINE"))
    if not deadline:
        return "не указан"
    try:
        parsed = datetime.fromisoformat(deadline.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return deadline
    return parsed.strftime("%d.%m.%Y %H:%M")


def _text(value: object) -> str:
    return str(value or "").strip()


def _text_items(value: object) -> list[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list | tuple | set):
        raw_items = list(value)
    else:
        raw_items = []
    items: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = _text(item).strip(" -•")
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def _int_value(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _local_model_usage(agent_id: str, note: str) -> ModelUsageRecord:
    return ModelUsageRecord(
        agent_id=agent_id,
        provider="local",
        model="bitrix_direct_response",
        status="skipped",
        notes=[note],
    )


__all__ = ["direct_tool_results_response"]
