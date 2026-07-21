from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
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
from ai_server.agents.bitrix24.draft_confirmation import draft_confirmation_phrase, matches_draft_confirmation
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ, compact_text, confidence, optional_int

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
    "task_close_report_incident",
    "task_close_control_get",
    "task_close_control_update",
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
        self.read_client = self.client
        if client is None and settings.llm_routing_enabled:
            fallback_model = settings.llm_pro_model if settings.llm_flash_fallback_to_pro else None
            self.read_client = OpenAICompatibleLLMClient(
                settings,
                model=settings.llm_flash_model or settings.llm_model,
                fallback_model=fallback_model,
            )
        self.settings = settings

    def _decision_client_for_request(self, request: str) -> LLMClient:
        if self.read_client is self.client:
            return self.client
        if _looks_like_bitrix_read_only_request(request):
            return self.read_client
        return self.client

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
        local_decision = _common_draft_discard_decision(task.request, task.context, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "draft_discard_route"),
                raw={"source": "draft_discard_route"},
            )

        local_decision = _common_draft_confirm_decision(task.request, task.context, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "draft_confirm_route"),
                raw={"source": "draft_confirm_route"},
            )

        local_decision = _common_task_close_draft_conflict_decision(task.request, task.context, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "task_close_draft_conflict_route"),
                raw={"source": "task_close_draft_conflict_route"},
            )

        local_decision = _common_task_close_active_update_decision(task.request, task.context, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "task_close_active_update_route"),
                raw={"source": "task_close_active_update_route"},
            )

        local_decision = _common_task_close_report_incident_decision(task.request, dialog_history, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "task_close_report_incident_route"),
                raw={"source": "task_close_report_incident_route"},
            )

        local_decision = _common_unsupported_vehicle_usage_decision(task.request, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "unsupported_vehicle_usage_route"),
                raw={"source": "unsupported_vehicle_usage_route"},
            )

        local_decision = _common_admin_panel_clarification_decision(task.request, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "admin_panel_clarification_route"),
                raw={"source": "admin_panel_clarification_route"},
            )

        local_decision = _common_task_close_control_decision(task.request, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "task_close_control_route"),
                raw={"source": "task_close_control_route"},
            )

        local_decision = _common_task_close_start_decision(task.request, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "task_close_start_route"),
                raw={"source": "task_close_start_route"},
            )

        local_decision = _common_task_create_draft_decision(task.request, task.context, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "task_create_draft_route"),
                raw={"source": "task_create_draft_route"},
            )

        local_decision = _common_calendar_event_draft_decision(task.request, dialog_history, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "calendar_reminder_route"),
                raw={"source": "calendar_reminder_route"},
            )

        local_decision = _common_warehouse_continuation_decision(task.request, dialog_history, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "warehouse_continuation_route"),
                raw={"source": "warehouse_continuation_route"},
            )

        local_decision = _common_read_decision(task.request, tool_definitions)
        if local_decision is not None:
            return BitrixLLMDecisionResult(
                decision=local_decision,
                model_usage=_local_model_usage(manifest.id, "common_read_route"),
                raw={"source": "common_read_route"},
            )

        instructions = load_instructions(manifest)
        completion = await self._decision_client_for_request(task.request).complete(
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
        if result.status == "denied" and result.tool in {
            "bitrix_warehouse_search",
            "bitrix_my_tasks",
            "bitrix_task_search",
            "bitrix_project_search",
        }:
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_read_denied_answer(result.data, result.error),
                model_usage=_local_model_usage(agent_id, "read_authorization_response"),
            )
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
        if result.tool == "project_create_draft":
            return BitrixLLMFinalResult(
                status="needs_human",
                answer=_format_project_create_draft_answer(result.data),
                model_usage=_local_model_usage(agent_id, "project_create_draft_response"),
            )
        if result.tool == "project_create_confirm":
            confirm_data = result.data if isinstance(result.data, dict) else {}
            has_followup_task = isinstance(confirm_data.get("followup_task_draft"), dict)
            return BitrixLLMFinalResult(
                status="needs_human" if has_followup_task else "completed",
                answer=_format_project_create_confirm_answer(result.data, portal_base_url=portal_base_url),
                model_usage=_local_model_usage(agent_id, "project_create_confirm_response"),
            )
        if result.tool == "project_create_discard":
            data = result.data if isinstance(result.data, dict) else {}
            return BitrixLLMFinalResult(
                status="completed",
                answer=(
                    "Черновик проекта и связанной задачи удалён."
                    if data.get("linked_task")
                    else "Черновик проекта удалён."
                ),
                model_usage=_local_model_usage(agent_id, "project_create_discard_response"),
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
        if result.tool == "task_close_report_incident":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_task_close_report_incident_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_close_report_incident_response"),
            )
        if result.tool == "task_close_control_get":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_task_close_control_get_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_close_control_get_response"),
            )
        if result.tool == "task_close_control_update":
            return BitrixLLMFinalResult(
                status="completed",
                answer=_format_task_close_control_update_answer(result.data),
                model_usage=_local_model_usage(agent_id, "task_close_control_update_response"),
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
    lines = [f"Черновик #{task_id or '?'}: {task_title}"]
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
    lines.append("")
    lines.append("Внести изменения (укажите пункт или подпункт и нужную информацию) либо подтвердить закрытие.")
    return "\n".join(lines) + _draft_confirmation_suffix({**data, "_draft_type": "task_close"})


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
        '"tool_calls":[{"name":"bitrix_warehouse_search|bitrix_my_tasks|bitrix_task_search|bitrix_project_search|bitrix_api|task_create_draft|task_create_confirm|task_draft_discard|task_close_draft|task_close_confirm|task_close_discard|task_close_report_incident|task_close_control_get|task_close_control_update|calendar_event_draft|calendar_event_confirm|calendar_event_discard|project_create_draft|project_create_confirm|project_create_discard|save_incomplete_proposal|delete_incomplete_proposal|save_responsible_response|portal_search|none","args":{},"summary":""}]}. '
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
        "include_comments=true если общий query нужно искать и в комментариях, "
        "ai_close_problem_type=not_done для AI-закрытий с невыполненными пунктами, "
        "ai_close_problem_type=unconfirmed для AI-закрытий с неизвестным/неподтверждённым результатом. "
        "Если пользователь просит закрытые задачи или дату закрытия, передай status=closed и include_closed=true. "
        "Для поиска сотрудника по имени — bitrix_api с user.search, получи numeric ID. "
        "Для чтения/поиска проекта по названию используй bitrix_project_search. "
        "Для создания проекта используй только project_create_draft, не прямой bitrix_api sonet_group.create. "
        "Вызывай project_create_draft только после bitrix_project_search, если подходящий проект не найден. "
        "Обычный пользователь может подготовить только свой личный проект: personal_for_self=true, name бери как 'Фамилия Имя' из permission_context.bitrix_current_user_profile.data.profile.label, без отчества. "
        "Произвольное создание проектов доступно только Bitrix-администратору. Личные проекты по умолчанию открытые и видимые. "
        "Для поиска складов, остатков и запросов вида 'найди склад Борисов' используй bitrix_warehouse_search, "
        "а не свободный bitrix_api. Если пользователь просит что есть на складе/остатки или говорит 'покажи склад <название>', передай include_products=true "
        "и product_limit=50, если пользователь не попросил другое количество. Для следующих позиций используй product_offset. "
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
        "If permission_context.pending_task_draft._draft_type is task_close and the user adds details for the same task, call task_close_draft with the same task_id and the updated full draft fields; backend will merge without resetting old fields. "
        "If a user starts closing another task while task_close draft is active, call task_close_draft for the requested task so backend returns the active-draft choice instead of overwriting the current draft. "
        "For task closing, missing completion details are not a reason to return needs_clarification after the task is identified: call task_close_draft and put unknown fields into missing_fields/unconfirmed_items so the user always sees the full draft. "
        "For task closing draft updates, keep the four blocks isolated: block 1/task points only records each work item status (completed/not done/unconfirmed), block 2 only records used materials/equipment, block 3 only records the user's overall status and reasons, and block 4 only records additional information. Ignore extra text inside the wrong block and never move it to another block. "
        "If permission_context.pending_task_draft._draft_type is calendar_event and the current user explicitly confirms calendar creation, call calendar_event_confirm. "
        "If permission_context.pending_task_draft._draft_type is project_create and the current user explicitly confirms project creation, call project_create_confirm. "
        "If permission_context.pending_task_draft._draft_type is admin_change, confirm with task_close_control_update operation=confirm and cancel with operation=discard. "
        "If the current user explicitly cancels or rejects a task creation draft, call task_draft_discard. "
        "If the current user explicitly cancels or rejects a task closing draft, call task_close_discard. "
        "If the current user explicitly cancels or rejects a calendar event draft, call calendar_event_discard. "
        "If the current user explicitly cancels or rejects a project creation draft, call project_create_discard. "
        "If dialog_history contains an AI-close report missing alert and the admin replies with option 1/restore, call task_close_report_incident with action=restore. "
        "If the admin replies with option 2/accept/delete for that alert, call task_close_report_incident with action=accept_missing. Do not add Bitrix buttons. "
        "Для просмотра настроек контроля закрытия задач используй task_close_control_get. "
        "Для изменения операторов, контролируемых пользователей, времени автозакрытия и даты запуска контроля используй task_close_control_update. "
        "Не используй bitrix_api для этих настроек. Действия task_close_control_update: add_operator/remove_operator/add_controlled_user/remove_controlled_user/set_auto_close_time/set_control_enabled_from. "
        "Статус оператора и статус контролируемого пользователя независимы: если оператор должен попадать под правила закрытия задач, отдельно добавь его как controlled user. "
        "Операторы могут менять только controlled users; операторов, время автозакрытия и дату запуска меняет только Bitrix-админ. "
        "Do not call confirm tools for ambiguous replies; ask a short clarification instead. "
        "Для фраз вида 'напомни мне завтра позвонить Борисову' используй календарь: подготовь calendar_event_draft, а не задачу и не прямой bitrix_api. "
        "Если время напоминания не указано, передай date_iso или start_iso с датой без времени: backend поставит 12:00 МСК. "
        "Если участники прямо не указаны, не передавай attendee_ids: событие будет только для текущего пользователя. "
        "Метод calendar.event.add не вызывай через bitrix_api: для него есть calendar_event_* tools. "
        "Для закрытия задачи не вызывай tasks.task.result.add/tasks.task.complete/tasks.task.approve через bitrix_api напрямую. "
        "Сначала найди задачу через bitrix_task_search, собери результат выполнения, затем вызови task_close_draft. "
        "В task_close_draft передавай короткий полный черновик: task_points, equipment_consumables, overall_status, "
        "not_done_items для невыполненных пунктов и unconfirmed_items для неизвестного/неподтверждённого результата. "
        "Не перетаскивай данные между блоками черновика закрытия: в блоке 1 по пунктам задачи сохраняй только статус пункта, без причин; материалы только в блок 2; общий статус и причины только в блок 3; дополнительную информацию только в блок 4. "
        "Если задача уже закрыта в Bitrix, всё равно используй task_close_draft/task_close_confirm, но передай already_closed=true: "
        "backend сохранит AI-отчёт и не будет повторно вызывать tasks.task.complete/tasks.task.approve. "
        "Если описание/чеклист исходной задачи пустые, не выдумывай task_points: передай source_task_description_empty=true, "
        "а результат пользователя положи в completion_summary как свободное описание выполненной работы. "
        "Если после уточнений пользователь явно разрешает закрыть с такими проблемами, backend сохранит признак неполного AI-закрытия для будущей индексации. "
        "Не показывай пользователю технический код AI_SERVER_TASK_CLOSE_INCOMPLETE; в видимом тексте пиши коротко: неподтвержденная или выполнена частично. "
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
        "Для результата task_close_draft покажи обычный текст без ссылок: задача, пункты задачи, оборудование/расходники, итог, невыполненное, неподтверждённое, что нужно дописать, действия. "
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
        "task_close_report_incident",
        "task_close_control_update",
        "calendar_event_draft",
        "calendar_event_confirm",
        "calendar_event_discard",
        "save_incomplete_proposal",
        "delete_incomplete_proposal",
        "save_responsible_response",
    }
)
_READ_MARKERS = ("покажи", "найди", "найти", "выведи", "список", "какие", "ищи")
_DOCUMENT_SEARCH_MARKERS = (
    "договор",
    "документ",
    "файл",
    "папк",
    "скан",
    "scan",
    "диск",
    "счет",
    "счёт",
)
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
_CALENDAR_REMINDER_TITLE_CUTOFF_RE = re.compile(
    r"\s+(?:не\s+добавляй|только\s+покажи|покажи\s+черновик|создай\s+черновик|для\s+подтверждения)\b.*",
    re.IGNORECASE | re.DOTALL,
)
_TASK_CREATE_TITLE_CUTOFF_RE = re.compile(
    r"(?:[.?!]\s*|\s+)(?:не\s+создавай|не\s+добавляй|только\s+покажи|покажи\s+черновик|"
    r"создай\s+черновик|для\s+подтверждения)\b.*",
    re.IGNORECASE | re.DOTALL,
)


def _common_draft_discard_decision(
    request: str,
    context: dict[str, Any],
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    lowered = _strip_command_prefix(request).casefold()
    if not _is_draft_discard_request(lowered):
        return None

    draft = context.get("pending_task_draft") if isinstance(context, dict) else None
    if isinstance(draft, dict) and draft:
        draft_type = _text(draft.get("_draft_type")) or "task_create"
        tool_name = {
            "task_create": "task_draft_discard",
            "task_close": "task_close_discard",
            "calendar_event": "calendar_event_discard",
            "project_create": "project_create_discard",
            "admin_change": "task_close_control_update",
        }.get(draft_type)
        if tool_name and _draft_discard_tool_available(tool_definitions, tool_name):
            args = {"operation": "discard"} if draft_type == "admin_change" else None
            return _discard_decision_tool(
                tool_name,
                f"deterministic Bitrix {draft_type} draft discard routing",
                args=args,
            )
        return None

    fallback_tool = _draft_discard_tool_from_text(lowered)
    if fallback_tool and _draft_discard_tool_available(tool_definitions, fallback_tool):
        return _discard_decision_tool(fallback_tool, f"deterministic Bitrix {fallback_tool} routing")
    return None


def _draft_discard_tool_from_text(lowered_request: str) -> str:
    if "календар" in lowered_request or "событ" in lowered_request:
        return "calendar_event_discard"
    if "проект" in lowered_request:
        return "project_create_discard"
    if "закрыт" in lowered_request and "задач" in lowered_request:
        return "task_close_discard"
    if "задач" in lowered_request:
        return "task_draft_discard"
    return ""


def _draft_discard_tool_available(tool_definitions: list[dict[str, Any]] | None, tool_name: str) -> bool:
    available_tools = {str(tool.get("name") or "") for tool in tool_definitions or []}
    return tool_definitions is None or tool_name in available_tools


def _is_draft_discard_request(lowered_request: str) -> bool:
    if "черновик" not in lowered_request:
        return False
    return any(marker in lowered_request for marker in ("отмени", "отменить", "удали", "удалить"))


def _discard_decision_tool(
    name: str,
    summary: str,
    *,
    args: dict[str, Any] | None = None,
) -> BitrixLLMDecision:
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.92,
        tool_calls=[
            BitrixLLMToolCall(
                name=name,
                args=args or {},
                summary=summary,
            )
        ],
    )


def _common_task_create_draft_decision(
    request: str,
    context: dict[str, Any],
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    """Route an unambiguous self-assigned task draft without model variance.

    Named assignees remain with the model/tool loop.  The task draft tool already
    resolves an exact project name through its bounded Bitrix lookup, so an
    unambiguous current-user project task is deterministic too.
    """

    if not _draft_discard_tool_available(tool_definitions, "task_create_draft"):
        return None
    if _write_profile_from_bitrix_profile(context.get("bitrix_current_user_profile")) not in {
        "member_write",
        "full_bitrix_write",
    }:
        return None
    args = _simple_self_task_draft_args(_strip_command_prefix(request))
    if args is None:
        return None
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.95,
        tool_calls=[
            BitrixLLMToolCall(
                name="task_create_draft",
                args=args,
                summary="deterministic Bitrix current-user task draft routing",
            )
        ],
    )


def _simple_self_task_draft_args(request: str) -> dict[str, Any] | None:
    lowered = request.casefold()
    if "задач" not in lowered:
        return None
    if not re.search(r"\b(?:создай|создайте|создать|поставь|поставьте|поставить)\s+задач", lowered):
        return None
    assignment_markers = ("назнач", "поруч", "исполнител", "ответственн")
    if any(marker in lowered for marker in assignment_markers):
        return None

    match = re.search(
        r"\bзадач[ауи]?\b(?P<between>[^:]{0,80}):\s*(?P<title>.+)", request, flags=re.IGNORECASE | re.DOTALL
    )
    if match is None:
        return None
    between = match.group("between")
    project_match = re.fullmatch(
        r"\s*в\s+проекте\s+(?P<project>.+?)\s+(?:на|для)\s+меня\s*",
        between,
        flags=re.IGNORECASE,
    )
    if project_match is None and "проект" in lowered:
        return None
    if (
        project_match is None
        and between.strip()
        and not re.fullmatch(r"\s*(?:на|для)\s+меня\s*", between, flags=re.IGNORECASE)
    ):
        return None
    title = _TASK_CREATE_TITLE_CUTOFF_RE.split(match.group("title"), maxsplit=1)[0]
    title = title.strip(" \t\r\n\"'«».,!?;:-")
    if not title:
        return None
    args: dict[str, Any] = {"title": title, "responsible_self": True}
    if project_match is not None:
        project_name = project_match.group("project").strip(" \t\r\n\"'«».,!?;:-")
        if not project_name:
            return None
        args["project_name"] = project_name
    return args


def _common_draft_confirm_decision(
    request: str,
    context: dict[str, Any],
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    draft = context.get("pending_task_draft") if isinstance(context, dict) else None
    if not isinstance(draft, dict) or not draft:
        return None
    if not matches_draft_confirmation(_strip_command_prefix(request), draft):
        return None

    draft_type = _text(draft.get("_draft_type")) or "task_create"
    tool_name = {
        "task_create": "task_create_confirm",
        "task_close": "task_close_confirm",
        "calendar_event": "calendar_event_confirm",
        "project_create": "project_create_confirm",
        "admin_change": "task_close_control_update",
    }.get(draft_type)
    if not tool_name:
        return None
    if not _draft_discard_tool_available(tool_definitions, tool_name):
        return None
    args = {"operation": "confirm"} if draft_type == "admin_change" else None
    return _confirm_decision_tool(
        tool_name,
        f"deterministic Bitrix {draft_type} draft confirm routing",
        args=args,
    )


def _is_draft_confirm_request(lowered_request: str) -> bool:
    if any(marker in lowered_request for marker in ("не создавай", "отмени", "отменить", "удали", "удалить")):
        return False
    return bool(
        re.search(r"\bда\b", lowered_request)
        or "подтверждаю" in lowered_request
        or "всё верно" in lowered_request
        or "все верно" in lowered_request
        or "создавай" in lowered_request
        or "закрывай" in lowered_request
        or "добавь в календар" in lowered_request
        or "создай проект" in lowered_request
    )


def _confirm_decision_tool(
    name: str,
    summary: str,
    *,
    args: dict[str, Any] | None = None,
) -> BitrixLLMDecision:
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.92,
        tool_calls=[
            BitrixLLMToolCall(
                name=name,
                args=args or {},
                summary=summary,
            )
        ],
    )


def _common_task_close_start_decision(
    request: str,
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    if not _draft_discard_tool_available(tool_definitions, "task_close_draft"):
        return None
    clean_request = _strip_command_prefix(request)
    lowered = clean_request.casefold()
    if not _is_task_close_start_request(lowered):
        return None
    task_id = _extract_task_id(clean_request)
    if task_id is None:
        return None
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.92,
        tool_calls=[
            BitrixLLMToolCall(
                name="task_close_draft",
                args={
                    "task_id": task_id,
                    "close_now": True,
                    "overall_status": "unconfirmed",
                    "unconfirmed_items": ["результат выполнения не указан"],
                    "missing_fields": [
                        "что сделано по задаче",
                        "оборудование/расходники, если использовались",
                        "итог: выполнена полностью, частично или не выполнена",
                    ],
                },
                summary="deterministic task close draft start routing",
            )
        ],
    )


def _is_task_close_start_request(lowered_request: str) -> bool:
    if "задач" not in lowered_request:
        return False
    if any(marker in lowered_request for marker in ("найди", "найти", "покажи", "выведи", "список", "какие", "ищи")):
        return False
    return any(marker in lowered_request for marker in ("закрой", "закрыть", "закроем", "закрываем", "заверши"))


def _common_task_close_draft_conflict_decision(
    request: str,
    context: dict[str, Any],
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    draft = context.get("pending_task_draft") if isinstance(context, dict) else None
    if not isinstance(draft, dict) or draft.get("_draft_type") != "task_close":
        return None
    pending = draft.get("_task_close_conflict_pending")
    if not isinstance(pending, dict) or not pending:
        return None

    action = _task_close_draft_conflict_action(request)
    if not action:
        return None
    if action == "continue_current":
        if not _draft_discard_tool_available(tool_definitions, "task_close_draft"):
            return None
        return BitrixLLMDecision(
            status="completed",
            answer="",
            confidence=0.92,
            tool_calls=[
                BitrixLLMToolCall(
                    name="task_close_draft",
                    args=_task_close_draft_replay_args(draft),
                    summary="deterministic task close active draft continue routing",
                )
            ],
        )
    if action == "close_current":
        if not _draft_discard_tool_available(tool_definitions, "task_close_confirm"):
            return None
        return _confirm_decision_tool(
            "task_close_confirm", "deterministic task close active draft close-current routing"
        )

    if not (
        _draft_discard_tool_available(tool_definitions, "task_close_discard")
        and _draft_discard_tool_available(tool_definitions, "task_close_draft")
    ):
        return None
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.92,
        tool_calls=[
            BitrixLLMToolCall(
                name="task_close_discard",
                args={},
                summary="deterministic task close active draft discard-current routing",
            ),
            BitrixLLMToolCall(
                name="task_close_draft",
                args=_task_close_draft_replay_args(pending),
                summary="deterministic task close active draft start-requested routing",
            ),
        ],
    )


def _task_close_draft_conflict_action(request: str) -> str:
    text = _strip_command_prefix(request).strip().casefold()
    if text in {"1", "1.", "первый", "первый вариант"}:
        return "continue_current"
    if text in {"2", "2.", "второй", "второй вариант"}:
        return "close_current"
    if text in {"3", "3.", "третий", "третий вариант"}:
        return "start_requested"
    if "продолж" in text and "текущ" in text:
        return "continue_current"
    if "закр" in text and ("как есть" in text or "помет" in text):
        return "close_current"
    if any(marker in text for marker in ("перейти к нов", "к новой задач", "отменить текущ", "отмени текущ")):
        return "start_requested"
    return ""


def _task_close_draft_replay_args(draft: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": optional_int(draft.get("task_id")),
        "task_title": _text(draft.get("task_title")),
        "action": _text(draft.get("action")) or "complete",
        "completion_summary": _text(draft.get("completion_summary")),
        "task_points": _text_items(draft.get("task_points")),
        "source_task_description_empty": bool(draft.get("source_task_description_empty")),
        "equipment_consumables": _text(draft.get("equipment_consumables")),
        "additional_info": _text(draft.get("additional_info")),
        "overall_status": _text(draft.get("overall_status")),
        "not_done_items": _text_items(draft.get("not_done_items")),
        "unconfirmed_items": _text_items(draft.get("unconfirmed_items")),
        "unresolved_items": _text_items(draft.get("unresolved_items")),
        "missing_fields": _text_items(draft.get("missing_fields")),
    }


def _common_task_close_active_update_decision(
    request: str,
    context: dict[str, Any],
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    if not _draft_discard_tool_available(tool_definitions, "task_close_draft"):
        return None
    draft = context.get("pending_task_draft") if isinstance(context, dict) else None
    if not isinstance(draft, dict) or draft.get("_draft_type") != "task_close":
        return None
    if isinstance(draft.get("_task_close_conflict_pending"), dict):
        return None

    clean_request = _strip_command_prefix(request)
    lowered = clean_request.casefold()
    if _is_draft_discard_request(lowered) or _is_draft_confirm_request(lowered):
        return None
    if _task_close_report_incident_action(clean_request) and _has_task_close_report_context(clean_request):
        return None

    current_task_id = optional_int(draft.get("task_id"))
    requested_task_id = _extract_task_id(clean_request)
    task_id = requested_task_id or current_task_id
    if task_id is None:
        return None

    args = _task_close_active_update_args(clean_request, draft=draft, task_id=task_id)
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.9,
        tool_calls=[
            BitrixLLMToolCall(
                name="task_close_draft",
                args=args,
                summary="deterministic task close active draft update routing",
            )
        ],
    )


def _task_close_active_update_args(request: str, *, draft: dict[str, Any], task_id: int) -> dict[str, Any]:
    summary = _task_close_update_summary(request, task_id=task_id)
    lowered = summary.casefold()
    task_points = _text_items(draft.get("task_points"))
    work_summary = _task_close_work_summary_from_text(summary)
    work_summary = _task_close_expand_numbered_work_summary(work_summary, task_points=task_points)
    work_lowered = work_summary.casefold()
    status_section = _task_close_section_text(summary, 3)
    work_targets_points = _task_close_summary_targets_task_points(work_summary or summary, task_points)
    status_text = status_section or (
        summary if not _task_close_has_section_markers(summary) and not work_targets_points else ""
    )
    status_lowered = status_text.casefold()
    args = {
        "task_id": task_id,
        "task_title": _text(draft.get("task_title")),
        "completion_summary": work_summary,
    }
    equipment = _task_close_equipment_from_text(summary, lowered, task_points=task_points)
    if equipment:
        args["equipment_consumables"] = equipment
    additional_info = _task_close_additional_info_from_text(summary)
    if additional_info:
        args["additional_info"] = additional_info
    overall_status = _task_close_overall_status_from_text(status_lowered)
    if overall_status:
        args["overall_status"] = overall_status
    status_reasons = (
        _task_close_status_reasons_from_text(status_text)
        if status_section or re.search(r"\bпричин[аы]?\b", status_text, flags=re.IGNORECASE)
        else []
    )
    if status_reasons:
        args["status_reasons"] = status_reasons
    not_done_items = _task_close_not_done_items_from_text(work_summary, work_lowered, task_points=task_points)
    if not_done_items:
        args["not_done_items"] = not_done_items
    unconfirmed_items = _task_close_unconfirmed_items_from_text(work_summary, work_lowered, task_points=task_points)
    if unconfirmed_items:
        args["unconfirmed_items"] = unconfirmed_items
    if equipment or additional_info or overall_status or status_reasons:
        args["missing_fields"] = []
    return args


def _task_close_update_summary(request: str, *, task_id: int) -> str:
    text = re.sub(
        rf"^\s*(?:по\s+)?задач[аеуы]?\s+#?{task_id}\s*:?\s*",
        "",
        request,
        flags=re.IGNORECASE,
    )
    return compact_text(text) or compact_text(request)


def _task_close_work_summary_from_text(summary: str) -> str:
    text = _task_close_section_text(summary, 1) or _task_close_before_any_block(summary, (2, 3, 4))
    text = re.sub(r"^\s*(?:по\s+)?пункт[ау]?\s+1\s*:?", "", text, flags=re.IGNORECASE)
    if re.search(r"(?<!\d)1\.\d+\s+", text):
        return compact_text(text).strip(" .;:-")
    return _task_close_clean_referenced_text(text)


def _task_close_expand_numbered_work_summary(summary: str, *, task_points: list[str]) -> str:
    if not summary or not task_points:
        return summary
    items: list[str] = []
    for match in re.finditer(r"(?<!\d)1\.(\d+)\s+(.+?)(?=(?:\s+1\.\d+\s+)|$)", summary, flags=re.DOTALL):
        point_index = optional_int(match.group(1))
        if point_index is None or point_index < 1 or point_index > len(task_points):
            continue
        value = _task_close_clean_referenced_text(match.group(2)).strip(" ,.;:-")
        if not value:
            continue
        point = task_points[point_index - 1]
        status_label = _task_close_work_status_label(value)
        if status_label:
            items.append(f"{point} {status_label}")
        else:
            items.append(value if _task_close_texts_overlap(point.casefold(), value) else f"{point} {value}")
    return ". ".join(items) if items else summary


def _task_close_equipment_from_text(summary: str, lowered: str, *, task_points: list[str]) -> str:
    section = _task_close_section_text(summary, 2)
    if not section:
        if _task_close_has_section_markers(summary):
            return ""
        if _task_close_summary_targets_task_points(summary, task_points):
            return ""
        has_equipment_intent = re.search(
            r"\b(?:использ|примен|израсход|расход|материал|оборудовани[ея]|кабель|камер[ауы])",
            lowered,
            flags=re.IGNORECASE,
        )
        if not has_equipment_intent:
            return ""
    section = section or summary
    section_lowered = section.casefold()
    if "не использ" in section_lowered or "не примен" in section_lowered:
        return "не использовалось"
    section = re.sub(
        r"^\s*(?:по\s+)?пункт[ау]?\s+2\s*(?:оборудование|расходники)?\s*:?",
        "",
        section,
        flags=re.IGNORECASE,
    )
    section = re.sub(r"^\s*2\s+", "", section)
    section = re.sub(r"^\s*(?:оборудование|расходники)\s*:?", "", section, flags=re.IGNORECASE)
    section = re.split(
        r"\b(?:задач[ауы]?|работ[аы]?)\s+(?:выполн|не\s+выполн|частич)|\bпричин[аы]?\b",
        section,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _task_close_clean_referenced_text(section)


def _task_close_additional_info_from_text(summary: str) -> str:
    section = _task_close_section_text(summary, 4)
    if not section:
        return ""
    section = re.sub(
        r"^\s*(?:по\s+)?пункт[ау]?\s+4\s*(?:дополнительно|дополнительная информация)?\s*:?",
        "",
        section,
        flags=re.IGNORECASE,
    )
    section = re.sub(r"^\s*4\s+", "", section)
    return _task_close_filter_additional_info_text(_task_close_clean_referenced_text(section))


def _task_close_overall_status_from_text(lowered: str) -> str:
    if "частич" in lowered or "не все" in lowered:
        return "partial"
    if "не выполн" in lowered or "не сделан" in lowered:
        return "not_done"
    if "не подтверж" in lowered or "неизвест" in lowered or "не провер" in lowered:
        return "unconfirmed"
    if "выполн" in lowered or "готов" in lowered or "сделан" in lowered:
        return "completed"
    return ""


def _task_close_not_done_items_from_text(summary: str, lowered: str, *, task_points: list[str]) -> list[str]:
    values = _task_close_status_items_from_text(
        summary,
        task_points=task_points,
        markers=("не выполн", "не сделан", "не сделал", "не поставил", "не установил", "не забрал", "не починил"),
    )
    if values:
        return _unique_text_items(values)
    if _task_close_work_status_label(lowered) == "не выполнено":
        return [summary]
    return []


def _task_close_unconfirmed_items_from_text(summary: str, lowered: str, *, task_points: list[str]) -> list[str]:
    values = _task_close_status_items_from_text(
        summary,
        task_points=task_points,
        markers=("не подтверж", "неизвест", "не провер"),
    )
    if values:
        return values
    if "не подтверж" in lowered or "неизвест" in lowered or "не провер" in lowered:
        return [summary]
    return []


def _task_close_status_reasons_from_text(summary: str) -> list[str]:
    summary = re.sub(r"^\s*(?:по\s+)?пункт[ау]?\s+3\s*:?", "", summary, flags=re.IGNORECASE)
    summary = re.sub(r"^\s*3\s+", "", summary)
    text = _task_close_clean_referenced_text(summary)
    if not text:
        return []
    reason = _task_close_reason_from_text(text)
    if reason:
        return _task_close_display_items(reason)
    text = re.sub(
        r"\b(?:выполнено|выполнена)\s+(?:полностью|частично)\b|\bне\s+выполнено\b|\bне\s+выполнена\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^\s*[,.;:-]*\s*(?:причин[аы]?\s*:?)?", "", text, flags=re.IGNORECASE)
    return _task_close_display_items(text)


def _task_close_status_items_from_text(summary: str, *, task_points: list[str], markers: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    clauses = re.split(r"(?:\r?\n)+|[.;]+|,(?=\s*(?:\d+\.\d+|по\s+пункт))", summary)
    for clause in clauses:
        cleaned = _task_close_clean_referenced_text(clause)
        lowered = cleaned.casefold()
        if not cleaned or not any(marker in lowered for marker in markers):
            continue
        matched = [point for point in task_points if _task_close_texts_overlap(point.casefold(), cleaned)]
        result.extend(matched or [cleaned])
    return _unique_text_items(result)


def _task_close_reason_from_text(summary: str) -> str:
    match = re.search(
        r"\bпричин[аы]?\s*:?\s*(.+?)(?=(?:\.\s*(?:(?:по\s+)?пункт[ау]?\s+4\b|4\b))|$)",
        summary,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""
    return _task_close_clean_referenced_text(match.group(1))


def _task_close_section_text(summary: str, section_number: int) -> str:
    match = re.search(_task_close_section_marker_pattern(section_number), summary, flags=re.IGNORECASE)
    if not match:
        return ""
    next_match = _task_close_next_section_match(summary[match.end() :], section_number)
    end = match.end() + next_match.start() if next_match else len(summary)
    return compact_text(summary[match.start() : end]).strip(" .;")


def _task_close_has_section_markers(summary: str) -> bool:
    if re.search(r"(?<!\d)1\.\d+\s+", summary):
        return True
    return any(
        re.search(_task_close_section_marker_pattern(number), summary, flags=re.IGNORECASE) for number in range(1, 5)
    )


def _task_close_before_any_block(summary: str, section_numbers: tuple[int, ...]) -> str:
    matches = [
        match
        for number in section_numbers
        if (match := re.search(_task_close_section_marker_pattern(number), summary, flags=re.IGNORECASE))
    ]
    if not matches:
        return compact_text(summary).strip(" .;")
    first = min(matches, key=lambda item: item.start())
    return compact_text(summary[: first.start()]).strip(" .;")


def _task_close_before_block(summary: str, section_number: int) -> str:
    match = re.search(_task_close_section_marker_pattern(section_number), summary, flags=re.IGNORECASE)
    return compact_text(summary[: match.start()] if match else summary).strip(" .;")


def _task_close_section_marker_pattern(section_number: int) -> str:
    explicit = rf"(?:по\s+)?пункт[ау]?\s+{section_number}\b"
    numeric_markers = {
        2: r"(?<![\d.])2(?![\d.])(?=\s+(?:использ|не\s+использ|примен|не\s+примен|оборуд|материал|расход))",
        3: r"(?<![\d.])3(?![\d.])(?=\s+(?:статус|выполн|не\s+выполн|частич|причин))",
        4: r"(?<![\d.])4(?![\d.])(?=\s+(?:отсутств|нет|доп|информац|нужно|надо|требуется|вернуться))",
    }
    numeric = numeric_markers.get(section_number)
    return rf"(?:{explicit}|{numeric})" if numeric else explicit


def _task_close_next_section_match(text: str, section_number: int) -> re.Match[str] | None:
    matches = [
        match
        for number in range(section_number + 1, 5)
        if (match := re.search(_task_close_section_marker_pattern(number), text, flags=re.IGNORECASE))
    ]
    return min(matches, key=lambda item: item.start()) if matches else None


def _task_close_clean_referenced_text(value: str) -> str:
    text = compact_text(value).strip(" .;:-")
    text = re.sub(r"^(?:\d+\.\d+|\d+[.)])\s*", "", text)
    text = re.sub(r"^\s*(?:по\s+)?пункт[ау]?\s+\d+(?:\.\d+)?\s*:?", "", text, flags=re.IGNORECASE)
    return compact_text(text).strip(" .;:-")


def _task_close_filter_additional_info_text(value: str) -> str:
    text = compact_text(value)
    if not text or _task_close_absent_value(text):
        return text
    kept: list[str] = []
    for item in _task_close_display_items(text):
        if _task_close_additional_info_item_allowed(item):
            kept.append(item)
    return ", ".join(kept)


def _task_close_additional_info_item_allowed(value: str) -> bool:
    text = compact_text(value)
    lowered = text.casefold()
    if not text:
        return False
    future_markers = (
        "нужно",
        "надо",
        "требуется",
        "следующ",
        "взять с собой",
        "вернуться",
        "потом",
        "завтра",
        "на будущее",
    )
    if any(marker in lowered for marker in future_markers):
        return True
    wrong_block_patterns = (
        r"\bиспользова",
        r"\bизрасходова",
        r"\bприменил",
        r"\bприменили",
        r"\bпоставил",
        r"\bпоставили",
        r"\bустановил",
        r"\bустановили",
        r"\bподключил",
        r"\bподключили",
        r"\bсмонтировал",
        r"\bсмонтировали",
        r"\bзаменил",
        r"\bзаменили",
        r"\bработ[аы]\s+(?:выполн|не\s+выполн|частич)",
        r"\bзадач[ауы]?\s+(?:выполн|не\s+выполн|частич)",
        r"\b(?:выполнено|выполнена)\s+(?:полностью|частично)",
        r"\bне\s+выполнено\b",
        r"\bне\s+выполнена\b",
        r"\bпричин[аы]?\b",
    )
    return not any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in wrong_block_patterns)


def _task_close_summary_targets_task_points(summary: str, task_points: list[str]) -> bool:
    if not summary or not task_points:
        return False
    if re.search(r"(?<!\d)1\.\d+\s+", summary):
        return True
    summary_cf = summary.casefold()
    return any(_task_close_texts_overlap(point.casefold(), summary_cf) for point in task_points)


def _task_close_work_status_label(text: str) -> str:
    lowered = compact_text(text).casefold()
    if not lowered:
        return ""
    if "не подтверж" in lowered or "неизвест" in lowered or "не провер" in lowered:
        return "не подтверждено"
    not_done_patterns = (
        r"\bне\s+выполн",
        r"\bне\s+сдела",
        r"\bне\s+постав",
        r"\bне\s+установ",
        r"\bне\s+забра",
        r"\bне\s+почин",
        r"\bне\s+переда",
        r"\bне\s+провер",
    )
    if any(re.search(pattern, lowered) for pattern in not_done_patterns):
        return "не выполнено"
    if "выполнено полностью" in lowered or "выполнена полностью" in lowered:
        return "выполнено полностью"
    completed_patterns = (
        r"\bвыполн",
        r"\bсдела",
        r"\bготов",
        r"\bпостав",
        r"\bустанов",
        r"\bзабра",
        r"\bпочин",
        r"\bпереда",
        r"\bпровер",
    )
    if any(re.search(pattern, lowered) for pattern in completed_patterns):
        return "выполнено"
    return ""


def _common_task_close_report_incident_decision(
    request: str,
    dialog_history: list[dict[str, str]] | None,
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    if not _draft_discard_tool_available(tool_definitions, "task_close_report_incident"):
        return None
    action = _task_close_report_incident_action(request)
    if not action:
        return None
    context_text = _task_close_report_incident_context(request, dialog_history)
    if not _has_task_close_report_context(context_text):
        return None
    task_id = _task_close_report_task_id(context_text)
    if task_id is None:
        return None
    args: dict[str, Any] = {"task_id": task_id, "action": action}
    file_name = _task_close_report_file_name(context_text)
    if file_name:
        args["file_name"] = file_name
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.95,
        tool_calls=[
            BitrixLLMToolCall(
                name="task_close_report_incident",
                args=args,
                summary="deterministic task close report incident routing",
            )
        ],
    )


def _task_close_report_incident_action(request: str) -> str:
    text = _strip_command_prefix(request).strip().casefold()
    if text in {"1", "1.", "первый", "первый вариант"}:
        return "restore"
    if text in {"2", "2.", "второй", "второй вариант"}:
        return "accept_missing"
    if any(marker in text for marker in ("восстанов", "верни", "вернуть")):
        return "restore"
    if any(marker in text for marker in ("удали", "удалить", "все в порядке", "всё в порядке", "оставь как есть")):
        return "accept_missing"
    return ""


def _task_close_report_incident_context(
    request: str,
    dialog_history: list[dict[str, str]] | None,
) -> str:
    parts = [request]
    for item in reversed(dialog_history or []):
        content = str(item.get("content") or "")
        if _has_task_close_report_context(content):
            parts.append(content)
            break
    return "\n".join(parts)


def _has_task_close_report_context(text: str) -> bool:
    lowered = text.casefold()
    return "ai-close-" in lowered or "файл отчёта" in lowered or "отчет ai-закрытия" in lowered


def _task_close_report_task_id(text: str) -> int | None:
    for pattern in (r"задач[аеиу]?\s*#?\s*(\d+)", r"task\s*#?\s*(\d+)", r"#(\d+)"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return optional_int(match.group(1))
    return None


def _task_close_report_file_name(text: str) -> str:
    match = re.search(
        r"\bAI-close-\d+(?:-(?:ok|partial|unconfirmed|failed))?(?: \(\d+\))?\.txt\b",
        text,
        flags=re.IGNORECASE,
    )
    return match.group(0) if match else ""


def _common_admin_panel_clarification_decision(
    request: str,
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    clean_request = _strip_command_prefix(request)
    lowered = clean_request.casefold()
    if _looks_like_task_close_control_request(lowered) or _looks_like_vehicle_usage_admin_panel_request(lowered):
        return None
    if not _looks_like_ambiguous_admin_panel_request(lowered):
        return None

    return BitrixLLMDecision(
        status="needs_clarification",
        answer="Уточните, какую панель показать: закрытие задач или отчет по машинам и людям.",
        confidence=0.86,
        tool_calls=[
            BitrixLLMToolCall(
                name="none",
                args={},
                summary="ambiguous admin panel clarification",
            )
        ],
    )


def _common_unsupported_vehicle_usage_decision(
    request: str,
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    clean_request = _strip_command_prefix(request)
    lowered = clean_request.casefold()
    if not _looks_like_vehicle_usage_admin_panel_request(lowered):
        return None
    if _looks_like_force_bitrix_request(lowered):
        return None
    return BitrixLLMDecision(
        status="needs_clarification",
        answer=(
            "В Bitrix не нашёл такой раздел или отчёт. "
            "Похожий сценарий найден у Логиста: отчёт по машинам и людям. "
            "Если нужен он, напишите: Логист, покажи отчёт по машинам и людям. "
            "Если вы точно хотите проверить именно Bitrix, повторите запрос с уточнением: "
            "Битрикс, всё равно обработай этот запрос."
        ),
        confidence=0.86,
        tool_calls=[
            BitrixLLMToolCall(
                name="none",
                args={},
                summary="unsupported vehicle usage domain for bitrix",
            )
        ],
    )


def _looks_like_force_bitrix_request(lowered: str) -> bool:
    force_terms = (
        "всё равно",
        "все равно",
        "точно bitrix",
        "точно битрикс",
        "именно bitrix",
        "именно битрикс",
        "обработай этот запрос",
        "передай в bitrix",
        "передай в битрикс",
    )
    return any(term in lowered for term in force_terms)


def _looks_like_ambiguous_admin_panel_request(lowered: str) -> bool:
    return (
        ("админ" in lowered and "панел" in lowered)
        or ("спис" in lowered and ("оператор" in lowered or "пользовател" in lowered))
        or ("кто" in lowered and "оператор" in lowered)
    )


def _looks_like_vehicle_usage_admin_panel_request(lowered: str) -> bool:
    vehicle_domain = (
        "машин" in lowered or "автомоб" in lowered or "люд" in lowered or "сотрудник" in lowered or "логист" in lowered
    )
    return vehicle_domain and ("отчет" in lowered or "оператор" in lowered or "пользовател" in lowered)


def _common_task_close_control_decision(
    request: str,
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    available_tools = {str(tool.get("name") or "") for tool in tool_definitions or []}
    require_declared_tool = tool_definitions is not None
    clean_request = _strip_command_prefix(request)
    lowered = clean_request.casefold()
    if not _looks_like_task_close_control_request(lowered):
        return None

    update_args = _task_close_control_update_args(clean_request)
    if update_args is not None:
        if require_declared_tool and "task_close_control_update" not in available_tools:
            return None
        return BitrixLLMDecision(
            status="completed",
            answer="",
            confidence=0.9,
            tool_calls=[
                BitrixLLMToolCall(
                    name="task_close_control_update",
                    args=update_args,
                    summary="deterministic task close control update routing",
                )
            ],
        )

    if require_declared_tool and "task_close_control_get" not in available_tools:
        return None
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.88,
        tool_calls=[
            BitrixLLMToolCall(
                name="task_close_control_get",
                args={},
                summary="deterministic task close control settings routing",
            )
        ],
    )


def _looks_like_task_close_control_request(lowered: str) -> bool:
    task_close_domain = (
        "контролируем" in lowered
        or ("автозакрыт" in lowered and "задач" in lowered)
        or ("контрол" in lowered and ("закрыт" in lowered or "задач" in lowered))
        or ("закрыт" in lowered and "задач" in lowered)
    )
    if not task_close_domain:
        return False

    settings_intent = (
        "настрой" in lowered
        or "панел" in lowered
        or "спис" in lowered
        or "кто" in lowered
        or "оператор" in lowered
        or "пользовател" in lowered
        or "автозакрыт" in lowered
        or any(
            marker in lowered
            for marker in (
                "добав",
                "включ",
                "назнач",
                "убери",
                "убрать",
                "удали",
                "удалить",
                "исключ",
                "сними",
                "снять",
            )
        )
    )
    return settings_intent


def _task_close_control_update_args(request: str) -> dict[str, Any] | None:
    lowered = request.casefold()
    time_match = re.search(r"\b([01]?\d|2[0-3])[:.ч]\s*([0-5]\d)\b", request)
    if "автозакрыт" in lowered and time_match is not None:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        return {"action": "set_auto_close_time", "auto_close_time": f"{hour:02d}:{minute:02d}"}

    user_id = _first_int(request)
    if user_id is None:
        return None
    add = any(marker in lowered for marker in ("добав", "включ", "назнач"))
    remove = any(marker in lowered for marker in ("убери", "убрать", "удали", "удалить", "исключ", "сними", "снять"))
    if not add and not remove:
        return None
    if "оператор" in lowered:
        return {"action": "add_operator" if add else "remove_operator", "target_user_id": user_id}
    if "контрол" in lowered:
        return {"action": "add_controlled_user" if add else "remove_controlled_user", "target_user_id": user_id}
    return None


def _first_int(text: str) -> int | None:
    match = re.search(r"\b(\d+)\b", text)
    return optional_int(match.group(1)) if match else None


def _common_calendar_event_draft_decision(
    request: str,
    dialog_history: list[dict[str, str]] | None,
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    available_tools = {str(tool.get("name") or "") for tool in tool_definitions or []}
    if tool_definitions is not None and "calendar_event_draft" not in available_tools:
        return None

    args = _simple_calendar_reminder_args(_calendar_request_with_history(request, dialog_history))
    if args is None:
        return None
    return BitrixLLMDecision(
        status="completed",
        answer="",
        confidence=0.92,
        tool_calls=[
            BitrixLLMToolCall(
                name="calendar_event_draft",
                args=args,
                summary="deterministic Bitrix calendar reminder draft routing",
            )
        ],
    )


def _calendar_request_with_history(request: str, dialog_history: list[dict[str, str]] | None) -> str:
    """Join a short time/title reply to its still-active calendar request."""
    current = _strip_command_prefix(request)
    if "напомни" in current.casefold():
        return current
    for item in reversed(dialog_history or []):
        if not isinstance(item, dict) or str(item.get("role") or "") != "user":
            continue
        previous = str(item.get("content") or "").strip()
        if "напомни" in previous.casefold():
            return f"{previous}. {current}"
    return current


def _simple_calendar_reminder_args(request: str) -> dict[str, Any] | None:
    lowered = request.casefold()
    if "напомни" not in lowered:
        return None
    if any(marker in lowered for marker in ("задач", "проект")):
        return None

    match = re.search(
        r"\bнапомни(?:\s+мне)?\s+(?P<date>сегодня|завтра|послезавтра)\s+(?P<title>.+)",
        request,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None

    target_date = _relative_calendar_date(match.group("date"))
    title = _calendar_reminder_title(match.group("title"))
    if not target_date or not title:
        return None
    return {
        "title": title,
        "date_iso": target_date.isoformat(),
        "description": title,
    }


def _relative_calendar_date(value: str) -> date | None:
    today = _moscow_today()
    match value.casefold():
        case "сегодня":
            return today
        case "завтра":
            return today + timedelta(days=1)
        case "послезавтра":
            return today + timedelta(days=2)
    return None


def _moscow_today() -> date:
    return datetime.now(MOSCOW_TZ).date()


def _calendar_reminder_title(value: str) -> str:
    title = _CALENDAR_REMINDER_TITLE_CUTOFF_RE.sub("", value)
    title = re.split(r"[.!?]\s+", title, maxsplit=1)[0]
    title = re.sub(r"^(?:что|чтобы|о\s+том,?\s+что)\s+", "", title.strip(), flags=re.IGNORECASE)
    title = title.strip(" .,:;-")
    if not title:
        return ""
    return title[:1].upper() + title[1:]


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

    document_args = _common_document_read_args(clean_request)
    if document_args is not None:
        if require_declared_tool and "portal_search" not in available_tools:
            return None
        return _local_decision_tool("portal_search", document_args)

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


def _common_warehouse_continuation_decision(
    request: str,
    dialog_history: list[dict[str, str]] | None,
    tool_definitions: list[dict[str, Any]] | None,
) -> BitrixLLMDecision | None:
    """Route an explicitly named next warehouse page without another Bitrix LLM loop.

    The route is deliberately narrow: it uses only a page that is already
    visible in the same dialog history and only when the user repeats the
    warehouse name.  Any uncertain case falls back to the existing model path.
    """
    available = {str(item.get("name") or "") for item in tool_definitions or []}
    if "bitrix_warehouse_search" not in available:
        return None
    lowered = request.casefold()
    if not any(marker in lowered for marker in ("следующ", "дальше", "продолж")) or "склад" not in lowered:
        return None
    requested_name = _warehouse_name_from_continuation(request)
    if not requested_name:
        return None
    for item in reversed(dialog_history or []):
        if str(item.get("role") or "") != "assistant":
            continue
        content = str(item.get("content") or "")
        prior_name = _warehouse_name_from_answer(content)
        page_end = _warehouse_page_end(content)
        if prior_name and page_end is not None and _warehouse_names_match(requested_name, prior_name):
            return _local_decision_tool(
                "bitrix_warehouse_search",
                {
                    "query": prior_name,
                    "include_products": True,
                    "product_limit": 50,
                    "product_offset": page_end,
                },
            )
    return None


def _warehouse_name_from_continuation(request: str) -> str:
    match = re.search(r"\bсклад(?:а|е|у|ом)?\s+(?P<name>.+)$", request, flags=re.IGNORECASE)
    if not match:
        return ""
    value = re.sub(r"\b\d{3,}\b", " ", match.group("name"))
    value = re.sub(r"\b(?:покажи|следующ\w*|страниц\w*|позици\w*)\b", " ", value, flags=re.IGNORECASE)
    return compact_text(value).strip(" .,:;!?«»")


def _warehouse_name_from_answer(answer: str) -> str:
    match = re.search(r"(?:остатки\s+по|на)\s+складу\s+(?P<name>[^\n(]+)", answer, flags=re.IGNORECASE)
    return compact_text(match.group("name")) if match else ""


def _warehouse_page_end(answer: str) -> int | None:
    match = re.search(r"показаны\s+(?:позиции|результаты)\s+\d+\s*[-–]\s*(\d+)\s+из\s+\d+", answer, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"показаны\s+первые\s+(\d+)\s+позиц", answer, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _warehouse_names_match(requested: str, prior: str) -> bool:
    requested_words = [word for word in re.findall(r"[\wё]+", requested.casefold()) if len(word) >= 4]
    prior_words = [word for word in re.findall(r"[\wё]+", prior.casefold()) if len(word) >= 4]
    return bool(requested_words and prior_words and any(a[:5] == b[:5] for a in requested_words for b in prior_words))


def _common_document_read_args(request: str) -> dict[str, Any] | None:
    lowered = request.casefold()
    if _looks_like_document_continuation(lowered):
        return {"continuation": "next"}
    if not _looks_like_document_search_request(lowered):
        return None
    query = _clean_document_query(request)
    if not query:
        return None
    scope = "files" if _looks_like_file_or_disk_scope(lowered) else "documents"
    show_all = _looks_like_document_show_all(lowered)
    return {
        "query": query,
        "scope": scope,
        "limit": 50 if show_all else (_extract_document_limit(lowered) or 10),
        **({"show_all": True} if show_all else {}),
    }


def _looks_like_document_continuation(lowered: str) -> bool:
    if any(marker in lowered for marker in ("задач", "проект", "склад", "товар")):
        return False
    has_continuation = any(marker in lowered for marker in ("следующ", "дальше", "продолж"))
    has_target = any(marker in lowered for marker in ("результат", "документ", "файл"))
    return has_continuation and has_target


def _looks_like_document_show_all(lowered: str) -> bool:
    return any(
        marker in lowered
        for marker in (
            "покажи все",
            "покажи всё",
            "выведи все",
            "выведи всё",
            "все документы",
            "все файлы",
        )
    )


def _looks_like_document_search_request(lowered: str) -> bool:
    return any(marker in lowered for marker in _DOCUMENT_SEARCH_MARKERS)


def _looks_like_file_or_disk_scope(lowered: str) -> bool:
    return any(marker in lowered for marker in ("файл", "папк", "диск", "скан", "scan"))


def _extract_document_limit(lowered: str) -> int | None:
    match = re.search(r"\b(?:последн(?:ие|их)?|первые|покажи|найди|выведи)\s+(\d{1,2})\b", lowered)
    if not match:
        return None
    try:
        return max(1, min(int(match.group(1)), 50))
    except ValueError:
        return None


def _clean_document_query(request: str) -> str:
    text = _clean_read_query(request)
    text = re.sub(
        r"\b(?:покажи|найди|найти|выведи|ищи|список|последн(?:ие|их)?|первые|все|всё)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{1,2}\b", " ", text)
    text = re.sub(r"\b(?:по|на|в|из|для|мне|пожалуйста)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b(?:диске|диск|папке|папка|файлы|файл|документы|документ|сканы|скан)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n\"'«».,!?")
    replacements = {
        "азимуту": "Азимут",
        "договоры": "договор",
        "договора": "договор",
        "договоров": "договор",
        "договору": "договор",
        "счета": "счет",
        "счёта": "счет",
        "счетов": "счет",
        "счётов": "счет",
    }
    words = [replacements.get(word.casefold(), word) for word in text.split()]
    return " ".join(words).strip()


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
    ai_close_problem_type = _extract_ai_close_problem_type(lowered)
    if ai_close_problem_type:
        args["ai_close_problem_type"] = ai_close_problem_type
        args["status"] = "closed"
        args["include_closed"] = True
        if args["scope"] == "all":
            args["scope"] = "member"
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


def _extract_ai_close_problem_type(lowered: str) -> str:
    if "невыполн" in lowered or "не выполн" in lowered or "не сделан" in lowered:
        return "not_done"
    if (
        "неподтверж" in lowered
        or "не подтверж" in lowered
        or "неизвест" in lowered
        or "непровер" in lowered
        or "не смогли подтверд" in lowered
    ):
        return "unconfirmed"
    if "неполностью" in lowered or "не полностью" in lowered:
        return "any"
    return ""


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
    return _clean_project_query(match.group(1))


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


def _looks_like_bitrix_read_only_request(request: str) -> bool:
    lowered = _strip_command_prefix(request).casefold()
    write_terms = (
        "созда",
        "сдела",
        "добав",
        "измени",
        "обнов",
        "назнач",
        "закрой",
        "закрыть",
        "удали",
        "отмени",
        "подтверд",
        "сохрани",
    )
    if any(term in lowered for term in write_terms):
        return False
    read_terms = (
        "покажи",
        "найди",
        "список",
        "какие",
        "какая",
        "какой",
        "мои задач",
        "остат",
        "проект",
        "задач",
        "склад",
    )
    return any(term in lowered for term in read_terms)


def _clean_read_query(value: str) -> str:
    text = str(value or "").strip(" \t\r\n\"'«».,!?")
    text = re.split(r"\s+(?:со|с)\s+(?:статусом|сроком)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    text = re.split(r"\s+в\s+проекте\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip(" \t\r\n\"'«».,!?")


def _clean_project_query(value: str) -> str:
    text = _clean_read_query(value)
    text = re.split(
        r"\s+(?:ответь|покажи|дай|напиши|выведи)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    text = re.split(
        r"\s+(?:кратко|ссылку|ссылкой|название|id)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
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
