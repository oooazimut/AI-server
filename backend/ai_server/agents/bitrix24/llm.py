from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from ai_server.agents.specialist_llm_shared import (
    DIALOG_HISTORY_PROMPT_FRAGMENT,
    SKILLS_PROMPT_FRAGMENT,
    allowed_tool_definitions,
    compact_tool_result,
    decision_status,
    load_instructions,
    result_status,
    retrieval_context,
    skills_context,
)
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.settings import Settings
from ai_server.utils import confidence, optional_int

logger = logging.getLogger(__name__)

ALLOWED_TOOL_NAMES = {
    "bitrix_warehouse_search",
    "bitrix_my_tasks",
    "bitrix_task_search",
    "bitrix_project_search",
    "bitrix_api",
    "task_create_draft",
    "task_create_confirm",
    "task_draft_discard",
    "task_close_draft",
    "task_close_confirm",
    "task_close_discard",
    "calendar_event_draft",
    "calendar_event_confirm",
    "calendar_event_discard",
    "project_create_draft",
    "project_create_confirm",
    "project_create_discard",
    "save_incomplete_proposal",
    "delete_incomplete_proposal",
    "save_responsible_response",
    "portal_search",
    "none",
}


class BitrixAgentLLM(Protocol):
    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        dialog_history: list[dict[str, str]] | None = None,
        available_skills: list | None = None,
    ) -> BitrixLLMDecisionResult:
        pass

    async def compose(
        self,
        *,
        manifest: AgentManifest | None = None,
        task: AgentTask,
        decision: BitrixLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]],
    ) -> BitrixLLMFinalResult:
        pass


@dataclass(frozen=True)
class BitrixLLMToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class BitrixLLMDecision:
    status: str
    answer: str
    tool_calls: list[BitrixLLMToolCall]
    confidence: float = 0.5


@dataclass(frozen=True)
class BitrixLLMDecisionResult:
    decision: BitrixLLMDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BitrixLLMFinalResult:
    status: str
    answer: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class BitrixLLMService:
    def __init__(self, client: LLMClient | None = None, *, settings: Settings) -> None:
        self.client = client or OpenAICompatibleLLMClient()
        self.settings = settings

    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        dialog_history: list[dict[str, str]] | None = None,
        available_skills: list | None = None,
    ) -> BitrixLLMDecisionResult:
        local_decision = _common_read_decision(task.request, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "common_read_route"),
                raw={"source": "common_read_route"},
            )

        instructions = load_instructions(manifest)
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _decision_system_prompt(instructions)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent": {
                                "id": manifest.id,
                                "name": manifest.name,
                                "handoff_description": manifest.handoff_description,
                            },
                            "request": task.request,
                            "user": task.user.model_dump(),
                            "files": task.files,
                            "current_datetime": datetime.now(UTC).astimezone().isoformat(),
                            "available_skills": skills_context(available_skills or []),
                            "dialog_history": dialog_history or [],
                            "permission_context": _permission_context(task, self.settings),
                            "retrieval_context": retrieval_context(retrieval_hits),
                            "tools": allowed_tool_definitions(tool_definitions, ALLOWED_TOOL_NAMES),
                            "tool_results": [compact_tool_result(result) for result in (tool_results or [])],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        decision = _normalize_common_read_decision(_parse_decision(completion.json_content()), task.request)
        return BitrixLLMDecisionResult(
            decision=decision,
            model_usage=completion.model_usage,
            raw=completion.raw,
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest | None = None,
        task: AgentTask,
        decision: BitrixLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]],
    ) -> BitrixLLMFinalResult:
        agent_id = manifest.id if manifest is not None else "bitrix24"
        portal_base_url = self.settings.bitrix_portal_base_url
        direct_result = _direct_task_create_response(
            agent_id=agent_id,
            tool_results=tool_results,
            portal_base_url=portal_base_url,
        )
        if direct_result is not None:
            return direct_result
        completion = await self.client.complete(
            agent_id=agent_id,
            messages=[
                {"role": "system", "content": _compose_system_prompt(portal_base_url)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "user": task.user.model_dump(),
                            "initial_decision": _decision_dict(decision),
                            "tool_results": [compact_tool_result(result) for result in tool_results],
                            "approval_actions": approval_actions,
                            "portal_base_url": portal_base_url,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        parsed = completion.json_content()
        return BitrixLLMFinalResult(
            status=result_status(parsed.get("status")),
            answer=str(parsed.get("answer") or "").strip() or "Готово.",
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def llm_failure_result(message: str, agent_id: str = "bitrix24") -> BitrixLLMFinalResult:
    return BitrixLLMFinalResult(
        status="failed",
        answer=f"Не смог обработать Bitrix-запрос через LLM-субагента: {message}",
        model_usage=ModelUsageRecord(
            agent_id=agent_id,
            provider="",
            model="",
            status="error",
            notes=[message],
        ),
    )


def _direct_task_create_response(
    *,
    agent_id: str,
    tool_results: list[ToolResult],
    portal_base_url: str = "",
) -> BitrixLLMFinalResult | None:
    for result in reversed(tool_results):
        if result.status != "ok":
            continue
        if result.tool == "bitrix_warehouse_search":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_warehouse_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "warehouse_response"),
            )
        if result.tool == "bitrix_my_tasks":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_my_tasks_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "my_tasks_response"),
            )
        if result.tool == "bitrix_task_search":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_task_search_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "task_search_response"),
            )
        if result.tool == "bitrix_project_search":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_project_search_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "project_search_response"),
            )
        if result.tool == "portal_search":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_portal_search_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "portal_search_response"),
            )
        if result.tool == "task_create_draft":
            return BitrixLLMFinalResult(
                status="needs_human",
                answer=_format_task_create_draft_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_create_draft_response"),
            )
        if result.tool == "task_create_confirm":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_task_create_confirm_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "task_create_confirm_response"),
            )
        if result.tool == "task_draft_discard":
            return BitrixLLMFinalResult(
                status="completed",
                answer="Черновик задачи удалён.",
                model_usage=_local_model_usage(agent_id, "task_draft_discard_response"),
            )
        if result.tool == "task_close_draft":
            return BitrixLLMFinalResult(
                status="needs_human",
                answer=_format_task_close_draft_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_close_draft_response"),
            )
        if result.tool == "task_close_confirm":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_task_close_confirm_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "task_close_confirm_response"),
            )
        if result.tool == "task_close_discard":
            return BitrixLLMFinalResult(
                status="completed",
                answer="Черновик закрытия задачи удалён.",
                model_usage=_local_model_usage(agent_id, "task_close_discard_response"),
            )
        if result.tool == "calendar_event_draft":
            return BitrixLLMFinalResult(
                status="needs_human",
                answer=_format_calendar_event_draft_answer(result.data),
                model_usage=_local_model_usage(agent_id, "calendar_event_draft_response"),
            )
        if result.tool == "calendar_event_confirm":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_calendar_event_confirm_answer(result.data),
                model_usage=_local_model_usage(agent_id, "calendar_event_confirm_response"),
            )
        if result.tool == "calendar_event_discard":
            return BitrixLLMFinalResult(
                status="completed",
                answer="Черновик события календаря удалён.",
                model_usage=_local_model_usage(agent_id, "calendar_event_discard_response"),
            )
    return None


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
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        name = _text(item.get("product_name")) or "товар"
        title = _warehouse_product_link(name, _text(item.get("product_url")), portal_base_url=portal_base_url)
        amount = _format_stock_amount(item.get("amount"))
        suffix = f" — {amount} шт." if amount else ""
        lines.append(f"{index}. {title}{suffix}")

    total = _int_value(products.get("available_items_with_names")) or _int_value(products.get("available_items_seen"))
    offset = _int_value(products.get("offset")) or 0
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

    lines = [title]
    for index, item in enumerate(items, start=1):
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

    limit = _int_value(data.get("limit")) or len(items)
    if len(items) >= limit:
        lines.append(f"Показаны первые {len(items)} результатов. Если нужно, можно уточнить запрос.")
    return "\n".join(lines)


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
    roles = item.get("roles") if isinstance(item.get("roles"), list) else []
    roles_label = ", ".join(_text(role) for role in roles if _text(role))
    if include_roles and roles_label:
        details.append(roles_label)
    snippets = item.get("comment_snippets") if isinstance(item.get("comment_snippets"), list) else []
    first_snippet = next((_text(snippet) for snippet in snippets if _text(snippet)), "")
    if first_snippet:
        details.append(f"комментарий: {first_snippet}")
    return details


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
            "Если всё верно, напишите: да, создай.",
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
    preview = data.get("preview") if isinstance(data.get("preview"), dict) else {}
    draft = data.get("draft") if isinstance(data.get("draft"), dict) else {}
    task_id = _text(draft.get("task_id"))
    task_title = _text(preview.get("task_title")) or _text(draft.get("task_title")) or _task_fallback_title(task_id)
    action_label = _text(preview.get("action_label")) or "отметить задачу выполненной"
    result_text = _text(preview.get("completion_summary")) or _text(draft.get("completion_summary"))
    unresolved = preview.get("unresolved_items") if isinstance(preview.get("unresolved_items"), list) else []
    unresolved = [_text(item) for item in unresolved if _text(item)]
    lines = [
        "Черновик закрытия задачи:",
        f"Задача: {task_title}",
        f"Действие: {action_label}",
    ]
    if result_text:
        lines.append(f"Результат: {result_text}")
    if unresolved:
        lines.append("Непроверенные пункты:")
        lines.extend(f"- {item}" for item in unresolved)
        lines.append("При подтверждении будет добавлена метка AI_SERVER_TASK_CLOSE_INCOMPLETE.")
    lines.append("")
    lines.append("Если всё верно, напишите: да, закрывай.")
    return "\n".join(lines)


def _format_task_close_confirm_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    draft = data.get("draft") if isinstance(data.get("draft"), dict) else {}
    task_id = _text(data.get("task_id")) or _text(draft.get("task_id"))
    task_title = _text(data.get("task_title")) or _text(draft.get("task_title")) or _task_fallback_title(task_id)
    action = _text(data.get("action")) or _text(draft.get("action"))
    label = _task_link(task_title, task_id, portal_base_url=portal_base_url)
    unresolved = data.get("unresolved_items") if isinstance(data.get("unresolved_items"), list) else []
    suffix = " С непроверенными пунктами добавлена метка AI_SERVER_TASK_CLOSE_INCOMPLETE." if unresolved else ""
    if action == "approve":
        return f"Задача закрыта: {label}.{suffix}"
    return f"Задача отмечена выполненной: {label}.{suffix}"


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
    if description:
        lines.append(f"Описание: {description}")
    lines.append("")
    lines.append("Если всё верно, напишите: да, добавь в календарь.")
    return "\n".join(lines)


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


def _decision_system_prompt(instructions: str = "") -> str:
    extra = f"\n\nДополнительные инструкции:\n{instructions}" if instructions else ""
    return (
        "Ты LLM-субагент Bitrix24 внутри корпоративного AI-server. "
        "Оркестратор уже передал тебе запрос человека. "
        "Ты не выполняешь действия сам: выбираешь один или несколько tools. "
        "Backend выполнит tools и применит policy/OAuth/подтверждения. "
        "Запрещено отвечать так, будто действие уже выполнено, если нужен write-tool. "
        "В payload есть permission_context: именно ты обязан прочитать его до write-tool "
        "и решить, имеет ли текущий пользователь право просить такое действие. "
        "permission_context содержит Bitrix-факты о текущем пользователе и RAG-выдержки из политики прав. "
        f"{SKILLS_PROMPT_FRAGMENT}"
        f"{DIALOG_HISTORY_PROMPT_FRAGMENT}"
        "Если Bitrix-факты отсутствуют или противоречат политике, не угадывай права. "
        "Если permission_context не разрешает write-действие, не вызывай write-tool; "
        "верни needs_human или needs_clarification с коротким объяснением. "
        "Backend guardrails только страхуют выполнение, но не должны думать вместо тебя. "
        "Верни только JSON-объект без markdown. Формат: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"короткий предварительный ответ",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"bitrix_warehouse_search|bitrix_my_tasks|bitrix_task_search|bitrix_project_search|bitrix_api|task_create_draft|task_create_confirm|task_draft_discard|task_close_draft|task_close_confirm|task_close_discard|calendar_event_draft|calendar_event_confirm|calendar_event_discard|project_create_draft|project_create_confirm|project_create_discard|save_incomplete_proposal|delete_incomplete_proposal|save_responsible_response|portal_search|none","args":{},"summary":""}]}. '
        "Перед каждым tool_call сам проверь, хватает ли данных для его корректного вызова. "
        "Нельзя вызывать tool с надеждой, что backend или tool сам разберётся с недостающими данными. "
        'Если данных не хватает, не вызывай tool: верни status=needs_clarification, tool_calls=[{"name":"none"}], '
        "а в answer задай короткий уточняющий вопрос. "
        "Данные о текущем пользователе уже есть в permission_context.bitrix_current_user_profile. "
        "Для общей фразы 'мои задачи' или 'мои открытые задачи' используй bitrix_my_tasks: он включает задачи, "
        "где текущий пользователь исполнитель, постановщик, соисполнитель или другой участник Bitrix, "
        "возвращает 10 по умолчанию, берёт только активные статусы 1-4 и сортирует по сроку. "
        "Для 'задачи на мне', 'я исполнитель' используй bitrix_task_search со scope=responsible. "
        "Для 'задачи, поставленные мной' используй bitrix_task_search со scope=created_by. "
        "Для поиска задачи по ID, названию, сроку, просрочке или проекту используй bitrix_task_search, а не свободный bitrix_api. "
        "Для сложного поиска задач по нескольким критериям тоже используй bitrix_task_search: "
        "deadline_from/deadline_to для срока, created_from/created_to для даты создания, "
        "closed_from/closed_to для даты закрытия, comment_query для текста в комментариях, "
        "include_comments=true если общий query нужно искать и в комментариях. "
        "Если пользователь просит закрытые задачи или дату закрытия, передай status=closed и include_closed=true. "
        "Для поиска сотрудника по имени — bitrix_api с user.search, получи numeric ID. "
        "Для чтения/поиска проекта по названию используй bitrix_project_search. "
        "Для создания проекта используй только project_create_draft, не прямой bitrix_api sonet_group.create. "
        "Вызывай project_create_draft только после bitrix_project_search, если подходящий проект не найден. "
        "Обычный пользователь может подготовить только свой личный проект: personal_for_self=true, name бери как 'Фамилия Имя' из permission_context.bitrix_current_user_profile.data.profile.label, без отчества. "
        "Произвольное создание проектов доступно только Bitrix-администратору. Личные проекты по умолчанию открытые и видимые. "
        "Для поиска складов, остатков и запросов вида 'найди склад Борисов' используй bitrix_warehouse_search, "
        "а не свободный bitrix_api. Если пользователь просит что есть на складе/остатки, передай include_products=true "
        "и product_limit=10, если пользователь не попросил другое количество. Для следующих позиций используй product_offset. "
        "Не вызывай search.search: этот метод в текущем Bitrix недоступен. "
        "Для создания задачи используй task_create_draft. "
        "Для task_create_draft именно ты распознаёшь title, responsible_id/responsible_self, "
        "group_id, deadline_iso или no_deadline. Если ответственный — текущий пользователь, возьми имя из "
        "permission_context.bitrix_current_user_profile.data.profile.label и передай его в responsible_name. "
        "Если знаешь имя ответственного после поиска, тоже передай его в responsible_name "
        "только для человекочитаемого черновика. "
        "Если ответственный указан по имени — сначала вызови bitrix_api(user.search), получи ID, затем task_create_draft. "
        "Если проект указан по названию — сначала вызови bitrix_project_search, получи ID, затем task_create_draft; "
        "в task_create_draft передай group_id и project_name для человекочитаемого черновика. "
        "Если пользователь сказал относительный срок, вычисли deadline_iso сам по current_datetime. "
        "Если срок не указан, не спрашивай уточнение: backend поставит срок по умолчанию три рабочих дня, 19:00 МСК. "
        "Передавай no_deadline=true только если пользователь явно сказал, что задача должна быть без срока. "
        "Не вызывай task_create_draft без title и одного из responsible_id/responsible_self. "
        "If permission_context.pending_task_draft._draft_type is absent/task_create and the current user explicitly confirms creation, call task_create_confirm. "
        "If permission_context.pending_task_draft._draft_type is task_close and the current user explicitly confirms closing, call task_close_confirm. "
        "If permission_context.pending_task_draft._draft_type is calendar_event and the current user explicitly confirms calendar creation, call calendar_event_confirm. "
        "If permission_context.pending_task_draft._draft_type is project_create and the current user explicitly confirms project creation, call project_create_confirm. "
        "If the current user explicitly cancels or rejects a task creation draft, call task_draft_discard. "
        "If the current user explicitly cancels or rejects a task closing draft, call task_close_discard. "
        "If the current user explicitly cancels or rejects a calendar event draft, call calendar_event_discard. "
        "If the current user explicitly cancels or rejects a project creation draft, call project_create_discard. "
        "Do not call confirm tools for ambiguous replies; ask a short clarification instead. "
        "Для фраз вида 'напомни мне завтра позвонить Борисову' используй календарь: подготовь calendar_event_draft, а не задачу и не прямой bitrix_api. "
        "Если время напоминания не указано, передай date_iso или start_iso с датой без времени: backend поставит 12:00 МСК. "
        "Если участники прямо не указаны, не передавай attendee_ids: событие будет только для текущего пользователя. "
        "Метод calendar.event.add не вызывай через bitrix_api: для него есть calendar_event_* tools. "
        "Для закрытия задачи не вызывай tasks.task.result.add/tasks.task.complete/tasks.task.approve через bitrix_api напрямую. "
        "Сначала найди задачу через bitrix_task_search, собери результат выполнения, затем вызови task_close_draft. "
        "Если после уточнений пользователь явно разрешает закрыть с непроверенными пунктами, передай их в unresolved_items: "
        "backend добавит метку AI_SERVER_TASK_CLOSE_INCOMPLETE для будущей индексации. "
        "Для уведомления пользователя используй bitrix_api с im.notify.system.add. "
        "Методы закрытия задач complete/approve/result.add не вызывай через bitrix_api: для них есть task_close_* tools. "
        "Остальные разрешённые Bitrix-методы вызывай через bitrix_api. "
        "Для поиска документов/файлов используй portal_search. Если данных не хватает, status=needs_clarification."
        f"{extra}"
    )


def _compose_system_prompt(portal_base_url: str = "") -> str:
    links_rule = (
        "Если в ответе перечисляешь сущности портала (проекты/рабочие группы, задачи, "
        "лиды, сделки) — каждую сущность оформляй как Bitrix-ссылку [URL=...]название[/URL] на её страницу в портале, "
        "используя поле portal_base_url из payload и ID сущности из tool_results. "
        "Не оформляй сотрудников/пользователей ссылками на профиль и не показывай ID пользователя в тексте.\n"
        "- задача: {base}/company/personal/user/0/tasks/task/view/{ID}/\n"
        "- проект/рабочая группа: {base}/workgroups/group/{ID}/\n"
        "- лид CRM: {base}/crm/lead/details/{ID}/\n"
        "- сделка CRM: {base}/crm/deal/details/{ID}/\n"
        "Текст ссылки — название/ФИО сущности (или название+ID, если имени нет). "
        "Для файлов диска используй готовую ссылку из tool_results (DOWNLOAD_URL/поле ссылки), не строй её сам. "
        "Если portal_base_url пустой или ID неизвестен, выводи обычный текст без ссылки."
        if portal_base_url
        else "Если portal_base_url не передан, не пытайся строить ссылки на портал — выводи обычный текст."
    )
    return (
        "Ты тот же LLM-субагент Bitrix24. Сформируй итоговый ответ человеку по результатам tools. "
        "Не выдумывай данные, которых нет в tool_results. "
        "Для результата task_create_draft черновик должен быть обычным текстом без ссылок: название, ответственный, проект если указан, срок, описание, запрос подтверждения. "
        "Для результата task_create_confirm дай ссылку только на созданную задачу; ссылки на профиль сотрудника запрещены. "
        "Для результата task_close_draft покажи обычный текст без ссылок: задача, действие, результат, непроверенные пункты, запрос подтверждения. "
        "Для результата task_close_confirm дай ссылку только на задачу; ссылки на профиль сотрудника запрещены. "
        "Для результата calendar_event_draft покажи обычный текст без ссылок: название, начало, окончание, участники, описание, запрос подтверждения. "
        "Для результата calendar_event_confirm не выдумывай ссылку на календарь; если готовой ссылки нет, дай обычный текст. "
        "Для результата project_create_draft покажи обычный текст без ссылок: название проекта, тип, открытость, описание, запрос подтверждения. "
        "Для результата project_create_confirm дай ссылку только на созданный проект/рабочую группу; ссылки на профиль сотрудника запрещены. "
        "Если есть approval_actions, скажи, что действие подготовлено и требуется подтверждение. "
        f"{links_rule} "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human|failed","answer":"ответ человеку"}.'
    )


def _parse_decision(data: dict[str, Any]) -> BitrixLLMDecision:
    raw_tool_calls = data.get("tool_calls")
    tool_calls: list[BitrixLLMToolCall] = []
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if name not in ALLOWED_TOOL_NAMES:
                continue
            args = raw_call.get("args") if isinstance(raw_call.get("args"), dict) else {}
            tool_calls.append(
                BitrixLLMToolCall(
                    name=name,
                    args=args,
                    summary=str(raw_call.get("summary") or "").strip(),
                )
            )
    if not tool_calls:
        tool_calls = [BitrixLLMToolCall(name="none")]
    return BitrixLLMDecision(
        status=decision_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        confidence=confidence(data.get("confidence")),
        tool_calls=tool_calls,
    )


_WRITE_TOOL_NAMES = frozenset(
    {
        "task_create_draft",
        "task_create_confirm",
        "task_draft_discard",
        "task_close_draft",
        "task_close_confirm",
        "task_close_discard",
        "calendar_event_draft",
        "calendar_event_confirm",
        "calendar_event_discard",
        "save_incomplete_proposal",
        "delete_incomplete_proposal",
        "save_responsible_response",
    }
)
_READ_MARKERS = ("покажи", "найди", "найти", "выведи", "список", "какие", "ищи")
_WRITE_MARKERS = (
    "создай",
    "создать",
    "закрой",
    "закрыть",
    "измени",
    "изменить",
    "удали",
    "удалить",
    "напомни",
)


def _normalize_common_read_decision(decision: BitrixLLMDecision, request: str) -> BitrixLLMDecision:
    if any(call.name in _WRITE_TOOL_NAMES for call in decision.tool_calls):
        return decision
    local_decision = _common_read_decision(request, None)
    if local_decision is not None:
        return local_decision
    return decision


def _common_read_decision(request: str, tool_definitions: list[dict[str, Any]] | None) -> BitrixLLMDecision | None:
    available_tools = {str(tool.get("name") or "") for tool in tool_definitions or []}
    require_declared_tool = tool_definitions is not None
    clean_request = _strip_command_prefix(request)
    lowered = clean_request.casefold()
    if not any(marker in lowered for marker in _READ_MARKERS):
        return None
    if any(marker in lowered for marker in _WRITE_MARKERS):
        return None

    task_args = _common_task_read_args(clean_request)
    if task_args is not None:
        if require_declared_tool and "bitrix_task_search" not in available_tools:
            return None
        return _local_decision_tool("bitrix_task_search", task_args)

    project_query = _common_project_query(clean_request)
    if project_query:
        if require_declared_tool and "bitrix_project_search" not in available_tools:
            return None
        return _local_decision_tool("bitrix_project_search", {"query": project_query, "limit": 10})

    return None


def _replace_decision_tool(decision: BitrixLLMDecision, name: str, args: dict[str, Any]) -> BitrixLLMDecision:
    return BitrixLLMDecision(
        status="completed",
        answer=decision.answer,
        confidence=max(decision.confidence, 0.8),
        tool_calls=[BitrixLLMToolCall(name=name, args=args, summary="deterministic common Bitrix read routing")],
    )


def _local_decision_tool(name: str, args: dict[str, Any]) -> BitrixLLMDecision:
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.9,
        tool_calls=[BitrixLLMToolCall(name=name, args=args, summary="deterministic common Bitrix read routing")],
    )


def _common_task_read_args(request: str) -> dict[str, Any] | None:
    lowered = request.casefold()
    if "задач" not in lowered:
        return None

    task_id = _extract_task_id(request)
    if task_id is not None:
        return {"task_id": task_id}

    query = _extract_task_query(request)
    scope = _task_scope_from_text(lowered)
    if scope == "my" and not _has_explicit_user_task_scope(lowered):
        scope = "all"
    status = _task_status_from_text(lowered)
    args: dict[str, Any] = {
        "scope": scope,
        "status": status,
        "limit": _extract_task_limit(lowered) or 10,
    }
    if status in {"closed", "deferred", "declined", "all"}:
        args["include_closed"] = True
    project_name = _extract_project_name_from_task_request(request)
    if project_name:
        args["project_name"] = project_name
    if query:
        args["query"] = query
    comment_query = _extract_comment_query(request)
    if comment_query:
        args["comment_query"] = comment_query
        args["include_comments"] = True
        args["comment_lookup_task_limit"] = 200
    closed_from, closed_to = _extract_closed_date_range(request)
    if closed_from:
        args["closed_from"] = closed_from
        args["status"] = "closed"
        args["include_closed"] = True
    if closed_to:
        args["closed_to"] = closed_to
        args["status"] = "closed"
        args["include_closed"] = True
    return args


def _extract_task_limit(lowered: str) -> int | None:
    match = re.search(r"(?:не\s+больше|до|покажи)\s+(\d{1,2})\b", lowered, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        limit = int(match.group(1))
    except ValueError:
        return None
    return max(1, min(limit, 50))


def _extract_comment_query(request: str) -> str:
    patterns = (
        r"комментари[а-яё]*[^.?!]*(?:слово|фраз[ау]|текст)\s+(.+?)(?:[.?!]|$)",
        r"комментари[а-яё]*[^.?!]*\bесть\s+(.+?)(?:[.?!]|$)",
        r"с\s+комментари[а-яё]*\s+(.+?)(?:[.?!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, request, flags=re.IGNORECASE)
        if not match:
            continue
        value = _clean_comment_query(match.group(1))
        if value:
            return value
    return ""


def _clean_comment_query(value: str) -> str:
    text = str(value or "").strip(" \t\r\n\"'«».,!?")
    text = re.split(r"\s+покажи\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.sub(r"^(?:слово|фраз[ау]|текст)\s+", "", text, flags=re.IGNORECASE)
    return text.strip(" \t\r\n\"'«».,!?")


def _extract_closed_date_range(request: str) -> tuple[str, str]:
    lowered = request.casefold()
    if not any(marker in lowered for marker in ("закрыт", "заверш")):
        return "", ""
    match = re.search(
        r"\bс\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})\s+по\s+(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4})",
        request,
        flags=re.IGNORECASE,
    )
    if not match:
        return "", ""
    return _normalize_task_date(match.group(1)), _normalize_task_date(match.group(2))


def _normalize_task_date(value: str) -> str:
    match = re.fullmatch(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})", value.strip())
    if not match:
        return ""
    day, month, year = match.groups()
    if len(year) == 2:
        year = "20" + year
    try:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    except ValueError:
        return ""


def _common_project_query(request: str) -> str:
    lowered = request.casefold()
    if "проект" not in lowered or "задач" in lowered:
        return ""
    match = re.search(r"\bпроект[а-яё]*\s+(.+?)(?:[.?!]|$)", request, flags=re.IGNORECASE)
    if not match:
        return ""
    return _clean_read_query(match.group(1))


def _task_scope_from_text(lowered: str) -> str:
    if "поставлен" in lowered and "мной" in lowered:
        return "created_by"
    if "на мне" in lowered or "я исполнитель" in lowered or "где я исполнитель" in lowered:
        return "responsible"
    return "my"


def _has_explicit_user_task_scope(lowered: str) -> bool:
    return any(
        marker in lowered
        for marker in (
            "мои",
            "мой",
            "моя",
            "мою",
            "мной",
            "на мне",
            "я исполнитель",
            "где я исполнитель",
            "поставлен",
        )
    )


def _task_status_from_text(lowered: str) -> str:
    if "просроч" in lowered:
        return "overdue"
    if "отлож" in lowered:
        return "deferred"
    if "отклон" in lowered:
        return "declined"
    if "закрыт" in lowered or "заверш" in lowered:
        return "closed"
    if "включая закрыт" in lowered or "включая заверш" in lowered:
        return "all"
    return "active"


def _extract_task_id(request: str) -> int | None:
    match = re.search(r"\bзадач[ауы]?\s+#?(\d{1,10})\b", request, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _extract_project_name_from_task_request(request: str) -> str:
    match = re.search(r"\bв\s+проекте\s+(.+?)(?:[.?!]|$)", request, flags=re.IGNORECASE)
    if not match:
        return ""
    return _clean_read_query(match.group(1))


def _extract_task_query(request: str) -> str:
    match = re.search(r"\bнайд[иите]*\s+задач[ауы]?\s+(.+?)(?:[.?!]|$)", request, flags=re.IGNORECASE)
    if not match:
        return ""
    query = _clean_read_query(match.group(1))
    if not query or query.isdigit():
        return ""
    return query


def _strip_command_prefix(request: str) -> str:
    text = re.sub(r"^\[[^\]]+\]\s*", "", str(request or "")).strip()
    return re.sub(r"^битрикс[:,]?\s*", "", text, flags=re.IGNORECASE).strip()


def _clean_read_query(value: str) -> str:
    text = str(value or "").strip(" \t\r\n\"'«».,!?")
    text = re.split(r"\s+(?:со|с)\s+(?:статусом|сроком)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"\s+в\s+проекте\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip(" \t\r\n\"'«».,!?")


def _permission_context(task: AgentTask, settings: Settings) -> dict[str, Any]:
    user_id = optional_int(task.user.id)
    dialog_id = str(task.context.get("dialog_id") or (task.user.raw or {}).get("dialog_id") or "")
    profile_result = task.context.get("bitrix_current_user_profile")
    write_profile = _write_profile_from_bitrix_profile(profile_result)
    return {
        "current_user_id": user_id,
        "current_dialog_id": dialog_id,
        "current_user_write_profile": write_profile,
        "oauth_required_for_writes": settings.bitrix_oauth_required_for_writes,
        "pending_task_draft": task.context.get("pending_task_draft"),
        "bitrix_current_user_profile": profile_result,
        "permission_policy_context": task.context.get("permission_policy_context", []),
        "rules": [
            (
                "All Bitrix business writes must be prepared as drafts first and executed only after explicit"
                " chat confirmation."
            ),
            (
                "When oauth_required_for_writes is true, confirmed writes execute only through OAuth of"
                " current_user_id; if current_user_id or dialog_id is missing, do not prepare/execute writes."
            ),
            "Bitrix itself decides final permissions for the OAuth user; return the Bitrix denial if it refuses.",
            "Regular users must not create arbitrary projects; only admins may prepare arbitrary project creation.",
            (
                "Regular users may prepare task writes only for existing allowed projects or their own personal"
                " project according to the agreed naming rules."
            ),
            (
                "Use sonet_group.user.get and Bitrix read methods for context, but do not replace Bitrix"
                " permission checks with local guesses."
            ),
            "task closure must use task_close_draft first, then task_close_confirm after explicit chat confirmation.",
            "calendar reminders/events must use calendar_event_draft first, then calendar_event_confirm after explicit chat confirmation.",
        ],
    }


_MANAGER_POSITION_KEYWORDS = ("руковод", "начальник", "директор", "президент", "chairman")


def _write_profile_from_bitrix_profile(profile_result: object) -> str:
    if not isinstance(profile_result, dict) or profile_result.get("status") != "ok":
        return "read_only"
    profile = (profile_result.get("data") or {}).get("profile") or {}
    if not isinstance(profile, dict):
        return "read_only"
    if not profile.get("active", True):
        return "read_only"
    if profile.get("is_admin"):
        return "full_bitrix_write"
    position = str(profile.get("work_position") or "").lower()
    if any(kw in position for kw in _MANAGER_POSITION_KEYWORDS):
        return "full_bitrix_write"
    # External/extranet users get read-only; all internal employees get member_write
    user_type = str(profile.get("user_type") or "").lower()
    if user_type in ("", "employee"):
        return "member_write"
    return "read_only"


def _decision_dict(decision: BitrixLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [{"name": call.name, "args": call.args, "summary": call.summary} for call in decision.tool_calls],
    }
