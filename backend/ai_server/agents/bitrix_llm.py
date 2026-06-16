from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ai_server.agents.specialist_llm_shared import (
    DIALOG_HISTORY_PROMPT_FRAGMENT,
    allowed_tool_definitions,
    compact_tool_result,
    decision_status,
    load_instructions,
    result_status,
    retrieval_context,
)
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.settings import get_settings
from ai_server.utils import confidence, optional_int

logger = logging.getLogger(__name__)

ALLOWED_TOOL_NAMES = {
    "bitrix_api",
    "current_user_profile",
    "task_create_draft",
    "task_closure",
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
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        dialog_history: list[dict[str, str]] | None = None,
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
                            "dialog_history": dialog_history or [],
                            "permission_context": _permission_context(task),
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
        portal_base_url = get_settings().bitrix_portal_base_url
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
        f"{DIALOG_HISTORY_PROMPT_FRAGMENT}"
        "Если Bitrix-факты отсутствуют или противоречат политике, не угадывай права. "
        "Если permission_context не разрешает write-действие, не вызывай write-tool; "
        "верни needs_human или needs_clarification с коротким объяснением. "
        "Backend guardrails только страхуют выполнение, но не должны думать вместо тебя. "
        "Верни только JSON-объект без markdown. Формат: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"короткий предварительный ответ",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"current_user_profile|bitrix_api|task_create_draft|task_closure|portal_search|none","args":{},"summary":""}]}. '
        "Перед каждым tool_call сам проверь, хватает ли данных для его корректного вызова. "
        "Нельзя вызывать tool с надеждой, что backend или tool сам разберётся с недостающими данными. "
        'Если данных не хватает, не вызывай tool: верни status=needs_clarification, tool_calls=[{"name":"none"}], '
        "а в answer задай короткий уточняющий вопрос. "
        "Для проверки фактов о текущем пользователе можно использовать current_user_profile, но перед write-tool "
        "обычно уже есть permission_context.bitrix_current_user_profile. "
        "Для поиска задач используй bitrix_api с tasks.task.list/tasks.task.get. "
        "Для создания задачи используй task_create_draft. "
        "Для task_create_draft именно ты распознаёшь title, responsible_id/responsible_query/responsible_self, "
        "group_id/project_query, deadline_iso или no_deadline. "
        "Если пользователь сказал относительный срок, вычисли deadline_iso сам по current_datetime. "
        "Если срок не указан, применяй правила из retrieval_context; если правило неясно, спроси уточнение. "
        "Не вызывай task_create_draft без title, одного из responsible_id/responsible_query/responsible_self, "
        "и одного из deadline_iso/no_deadline=true. "
        "Для закрытия/завершения задачи из чата используй task_closure, если пользователь сообщил результат работы. "
        "Для task_closure именно ты выделяешь task_id или task_query и result_text. "
        "В result_text передавай только результат выполнения, без команды закрыть задачу. "
        "Не вызывай task_closure без result_text и одного из task_id/task_query. "
        "Для поиска документов/файлов используй portal_search. Если данных не хватает, status=needs_clarification."
        f"{extra}"
    )


def _compose_system_prompt(portal_base_url: str = "") -> str:
    links_rule = (
        "Если в ответе перечисляешь сущности портала (сотрудников, проекты/рабочие группы, задачи, "
        "лиды, сделки) — каждую сущность оформляй как markdown-ссылку на её страницу в портале, "
        "используя поле portal_base_url из payload и ID сущности из tool_results:\n"
        "- задача: {base}/company/personal/user/{RESPONSIBLE_ID}/tasks/task/view/{ID}/\n"
        "- проект/рабочая группа: {base}/workgroups/group/{ID}/\n"
        "- сотрудник: {base}/company/personal/user/{ID}/\n"
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


def _permission_context(task: AgentTask) -> dict[str, Any]:
    settings = get_settings()
    user_id = optional_int(task.user.id)
    full_write_user_ids = settings.resolved_agent_write_allowed_user_ids
    limited_user_ids = settings.resolved_agent_limited_task_create_user_ids
    limited_project_id = settings.agent_limited_task_create_project_id
    full_write = user_id is not None and user_id in full_write_user_ids
    limited_task_create = user_id is not None and limited_project_id is not None and user_id in limited_user_ids
    if full_write:
        profile = "full_bitrix_write"
    elif limited_task_create:
        profile = "limited_task_create"
    else:
        profile = "read_only"
    return {
        "current_user_id": user_id,
        "current_user_write_profile": profile,
        "full_write_user_ids": full_write_user_ids,
        "limited_task_create_user_ids": limited_user_ids,
        "limited_task_create_project_id": limited_project_id,
        "oauth_required_for_writes": settings.bitrix_oauth_required_for_writes,
        "bitrix_current_user_profile": task.context.get("bitrix_current_user_profile"),
        "permission_policy_context": task.context.get("permission_policy_context", []),
        "rules": [
            "full_bitrix_write users may prepare Bitrix write-tools, still requiring chat confirmation.",
            "limited_task_create users may prepare task_create_draft only for the configured limited project.",
            "read_only users should not prepare write-tools; ask for an authorized user or human handoff.",
            "task_closure should only be prepared when the user is acting on their own task result or has full write rights.",
        ],
    }


def _decision_dict(decision: BitrixLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [{"name": call.name, "args": call.args, "summary": call.summary} for call in decision.tool_calls],
    }
