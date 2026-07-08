from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    "bitrix_api",
    "task_create_draft",
    "task_create_confirm",
    "task_draft_discard",
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
        return BitrixLLMDecisionResult(
            decision=_parse_decision(completion.json_content()),
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
    return None


def _format_task_create_draft_answer(data: dict[str, Any]) -> str:
    preview = data.get("preview") if isinstance(data.get("preview"), dict) else {}
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    fields = params.get("fields") if isinstance(params.get("fields"), dict) else {}
    title = _text(preview.get("title")) or _text(fields.get("TITLE")) or "задача"
    responsible = _text(preview.get("responsible")) or "указанный сотрудник"
    deadline = _text(preview.get("deadline")) or _deadline_label_from_fields(fields)
    description = _text(preview.get("description")) or _text(fields.get("DESCRIPTION")) or f"Краткое содержание: {title}"
    notes = data.get("notes") if isinstance(data.get("notes"), list) else []
    if deadline != "без срока" and any("Срок по умолчанию" in str(note) for note in notes):
        deadline = f"{deadline} (по умолчанию: 3 рабочих дня)"
    return (
        "Черновик задачи:\n"
        f"Название: {title}\n"
        f"Ответственный: {responsible}\n"
        f"Срок: {deadline}\n"
        f"Описание: {description}\n\n"
        "Если всё верно, напишите: да, создай."
    )


def _format_task_create_confirm_answer(data: dict[str, Any], *, portal_base_url: str = "") -> str:
    params = data.get("params") if isinstance(data.get("params"), dict) else {}
    fields = params.get("fields") if isinstance(params.get("fields"), dict) else {}
    title = _text(fields.get("TITLE")) or "задача"
    task_id = _created_task_id(data.get("result"))
    if portal_base_url and task_id:
        url = f"{portal_base_url.rstrip('/')}/company/personal/user/0/tasks/task/view/{task_id}/"
        return f"Задача создана: [URL={url}]{title}[/URL]."
    return f"Задача создана: {title}."


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


def _local_model_usage(agent_id: str, note: str) -> ModelUsageRecord:
    return ModelUsageRecord(
        agent_id=agent_id,
        provider="local",
        model="bitrix_task_create_response",
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
        '"tool_calls":[{"name":"bitrix_warehouse_search|bitrix_api|task_create_draft|task_create_confirm|task_draft_discard|save_incomplete_proposal|delete_incomplete_proposal|save_responsible_response|portal_search|none","args":{},"summary":""}]}. '
        "Перед каждым tool_call сам проверь, хватает ли данных для его корректного вызова. "
        "Нельзя вызывать tool с надеждой, что backend или tool сам разберётся с недостающими данными. "
        'Если данных не хватает, не вызывай tool: верни status=needs_clarification, tool_calls=[{"name":"none"}], '
        "а в answer задай короткий уточняющий вопрос. "
        "Данные о текущем пользователе уже есть в permission_context.bitrix_current_user_profile. "
        "Для поиска задач используй bitrix_api с tasks.task.list/tasks.task.get. "
        "Для поиска сотрудника по имени — bitrix_api с user.search, получи numeric ID. "
        "Для поиска проекта по названию — bitrix_api с sonet_group.get, получи numeric ID. "
        "Для поиска складов, остатков и запросов вида 'найди склад Борисов' используй bitrix_warehouse_search, "
        "а не свободный bitrix_api. Если пользователь просит что есть на складе/остатки, передай include_products=true. "
        "Не вызывай search.search: этот метод в текущем Bitrix недоступен. "
        "Для создания задачи используй task_create_draft. "
        "Для task_create_draft именно ты распознаёшь title, responsible_id/responsible_self, "
        "group_id, deadline_iso или no_deadline. Если знаешь имя ответственного после поиска, передай его в responsible_name "
        "только для человекочитаемого черновика. "
        "Если ответственный указан по имени — сначала вызови bitrix_api(user.search), получи ID, затем task_create_draft. "
        "Если проект указан по названию — сначала вызови bitrix_api(sonet_group.get), получи ID, затем task_create_draft. "
        "Если пользователь сказал относительный срок, вычисли deadline_iso сам по current_datetime. "
        "Если срок не указан, не спрашивай уточнение: backend поставит срок по умолчанию три рабочих дня, 19:00 МСК. "
        "Передавай no_deadline=true только если пользователь явно сказал, что задача должна быть без срока. "
        "Не вызывай task_create_draft без title и одного из responsible_id/responsible_self. "
        "If permission_context.pending_task_draft exists and the current user explicitly confirms creation, call task_create_confirm. "
        "If the current user explicitly cancels or rejects the pending draft, call task_draft_discard. "
        "Do not call task_create_confirm for ambiguous replies; ask a short clarification instead. "
        "Закрытие задачи исполнителем: вызови tasks.task.result.add (добавить результат) + tasks.task.complete "
        "(завершить). Если TASK_CONTROL=Y, задача перейдёт в STATUS=4 (ждёт контроля). "
        "Закрытие задачи постановщиком: вызови tasks.task.approve напрямую — STATUS=5, без проверки результата. "
        "Для уведомления пользователя используй bitrix_api с im.notify.system.add. "
        "Все Bitrix-методы (approve/disapprove/complete/result.add/commentitem.add) вызывай через bitrix_api. "
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
        "Для результата task_create_draft черновик должен быть обычным текстом без ссылок: название, ответственный, срок, описание, запрос подтверждения. "
        "Для результата task_create_confirm дай ссылку только на созданную задачу; ссылки на профиль сотрудника запрещены. "
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
            "task closure via bitrix_api: responsible uses tasks.task.complete, creator uses tasks.task.approve.",
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
