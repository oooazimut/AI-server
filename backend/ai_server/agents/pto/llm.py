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
from ai_server.utils import confidence

logger = logging.getLogger(__name__)

ALLOWED_TOOL_NAMES = {
    "portal_document_search",
    "document_read",
    "spreadsheet_preview",
    "spreadsheet_compare",
    "document_draft_create",
    "document_draft_list",
    "none",
}


class PtoAgentLLM(Protocol):
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
    ) -> PtoLLMDecisionResult:
        pass

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: PtoLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> PtoLLMFinalResult:
        pass


@dataclass(frozen=True)
class PtoLLMToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class PtoLLMDecision:
    status: str
    answer: str
    tool_calls: list[PtoLLMToolCall]
    confidence: float = 0.5


@dataclass(frozen=True)
class PtoLLMDecisionResult:
    decision: PtoLLMDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PtoLLMFinalResult:
    status: str
    answer: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class PtoLLMService:
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
        available_skills: list | None = None,
    ) -> PtoLLMDecisionResult:
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
        return PtoLLMDecisionResult(
            decision=_parse_decision(completion.json_content()),
            model_usage=completion.model_usage,
            raw=completion.raw,
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: PtoLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> PtoLLMFinalResult:
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _compose_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "initial_decision": _decision_dict(decision),
                            "tool_results": [compact_tool_result(result) for result in tool_results],
                            "approval_actions": approval_actions or [],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        parsed = completion.json_content()
        return PtoLLMFinalResult(
            status=result_status(parsed.get("status")),
            answer=str(parsed.get("answer") or "").strip() or "Готово.",
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def pto_llm_failure_result(message: str, agent_id: str = "pto") -> PtoLLMFinalResult:
    return PtoLLMFinalResult(
        status="failed",
        answer=f"Не смог обработать ПТО-запрос через LLM-специалиста: {message}",
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
        "Ты LLM-специалист ПТО внутри корпоративного AI-server. "
        "Твоя зона: исполнительная и проектная документация, акты, письма по объектам, "
        "сметы/ведомости как технические документы, проверка комплектности и сравнение версий. "
        "Ты не работаешь напрямую с Bitrix API: выбирай document tools. Backend только скачивает/читает/сравнивает файлы "
        "и применяет guardrails доступа. "
        "В tool_results могут прийти результаты твоих предыдущих tool_calls; используй их как наблюдения "
        "для следующего шага. "
        f"{SKILLS_PROMPT_FRAGMENT}"
        f"{DIALOG_HISTORY_PROMPT_FRAGMENT}"
        "Если документ по смыслу бухгалтерский, складской, сетевой или программный, не притворяйся профильным специалистом: "
        "верни needs_human/needs_clarification и объясни, кому лучше передать. "
        "Перед каждым tool_call сам проверь, хватает ли данных. Если не хватает, не вызывай tool, задай уточняющий вопрос. "
        "Для сравнения таблиц сначала вызови spreadsheet_preview по каждому документу или по найденным entity_id. "
        "По preview сам выбери sheet, header_row_number, key_column и value_columns; только после этого вызывай "
        "spreadsheet_compare с явной схемой. spreadsheet_compare делает точный механический diff и не должен "
        "использоваться без выбранной тобой схемой. "
        "Если нужно подготовить текстовый черновик документа, сам сформируй его содержание и вызови "
        "document_draft_create. Этот tool только создаёт локальный черновик; отправка/загрузка в Bitrix требует "
        "отдельного подтверждаемого write-контура и сейчас не выполняется этим tool. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"короткий предварительный ответ",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"portal_document_search|document_read|spreadsheet_preview|spreadsheet_compare|document_draft_create|document_draft_list|none","args":{},"summary":""}]}.'
        f"{extra}"
    )


def _compose_system_prompt() -> str:
    return (
        "Ты тот же ПТО-специалист. Сформируй итоговый ответ человеку по результатам document tools. "
        "Не выдумывай данные, которых нет в tool_results. Если сравнение вернуло механические отличия, "
        "объясни их как ПТО: что изменилось, на что обратить внимание, где нужен человек. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human|failed","answer":"ответ человеку"}.'
    )


def _parse_decision(data: dict[str, Any]) -> PtoLLMDecision:
    raw_tool_calls = data.get("tool_calls")
    tool_calls: list[PtoLLMToolCall] = []
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if name not in ALLOWED_TOOL_NAMES:
                continue
            args = raw_call.get("args") if isinstance(raw_call.get("args"), dict) else {}
            tool_calls.append(
                PtoLLMToolCall(
                    name=name,
                    args=args,
                    summary=str(raw_call.get("summary") or "").strip(),
                )
            )
    if not tool_calls:
        tool_calls = [PtoLLMToolCall(name="none")]
    return PtoLLMDecision(
        status=decision_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        confidence=confidence(data.get("confidence")),
        tool_calls=tool_calls,
    )


def _decision_dict(decision: PtoLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [{"name": call.name, "args": call.args, "summary": call.summary} for call in decision.tool_calls],
    }
