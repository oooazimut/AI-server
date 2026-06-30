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
    safe_context,
    skills_context,
)
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.utils import confidence

logger = logging.getLogger(__name__)

ALLOWED_TOOL_NAMES = {
    "kartoteka_search",
    "kartoteka_context",
    "kartoteka_file_add",
    "kartoteka_file_delete",
    "kartoteka_file_move",
    "none",
}


class KartotekaAgentLLM(Protocol):
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
    ) -> KartotekaLLMDecisionResult: ...

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: KartotekaLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> KartotekaLLMFinalResult: ...


@dataclass(frozen=True)
class KartotekaLLMToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class KartotekaLLMDecision:
    status: str
    answer: str
    tool_calls: list[KartotekaLLMToolCall]
    confidence: float = 0.5


@dataclass(frozen=True)
class KartotekaLLMDecisionResult:
    decision: KartotekaLLMDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KartotekaLLMFinalResult:
    status: str
    answer: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class KartotekaLLMService:
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
    ) -> KartotekaLLMDecisionResult:
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
                            "context": safe_context(task.context),
                            "current_datetime": datetime.now(UTC).astimezone().isoformat(),
                            "available_skills": skills_context(available_skills or []),
                            "dialog_history": dialog_history or [],
                            "retrieval_context": retrieval_context(retrieval_hits),
                            "tools": allowed_tool_definitions(tool_definitions, ALLOWED_TOOL_NAMES),
                            "tool_results": [compact_tool_result(r) for r in (tool_results or [])],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return KartotekaLLMDecisionResult(
            decision=_parse_decision(completion.json_content()),
            model_usage=completion.model_usage,
            raw=completion.raw,
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: KartotekaLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> KartotekaLLMFinalResult:
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
                            "tool_results": [compact_tool_result(r) for r in tool_results],
                            "approval_actions": approval_actions or [],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        parsed = completion.json_content()
        return KartotekaLLMFinalResult(
            status=result_status(parsed.get("status")),
            answer=str(parsed.get("answer") or "").strip() or "Готово.",
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def kartoteka_llm_failure_result(message: str, agent_id: str = "kartoteka") -> KartotekaLLMFinalResult:
    return KartotekaLLMFinalResult(
        status="failed",
        answer=f"Не смог обработать запрос Картотеки через LLM: {message}",
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
        "Ты LLM-специалист ИИ-Картотека внутри корпоративного AI-server. "
        "Твоя зона: поиск файлов и документов в локальном файловом индексе организации. "
        "Ты не работаешь с Bitrix-диском — только с локальным файловым сервером. "
        "Операции изменения каталога (добавление, удаление, перемещение) сейчас недоступны. "
        f"{SKILLS_PROMPT_FRAGMENT}"
        f"{DIALOG_HISTORY_PROMPT_FRAGMENT}"
        "Сначала вызови kartoteka_context, если в tool_results ещё нет статистики. "
        "Для поиска используй kartoteka_search с чётким поисковым запросом. "
        "Если файл не найден — честно сообщи об этом, не придумывай результаты. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"предварительный ответ",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"kartoteka_search|kartoteka_context|none","args":{},"summary":""}]}.'
        f"{extra}"
    )


def _compose_system_prompt() -> str:
    return (
        "Ты та же ИИ-Картотека. Сформируй итоговый ответ для Переговорщика по результатам поиска в каталоге. "
        "Если файлы найдены — перечисли их названия и пути. "
        "Если ничего не найдено — сообщи честно и предложи уточнить запрос. "
        "Не выдумывай файлы, которых нет в tool_results. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human|failed","answer":"ответ человеку"}.'
    )


def _parse_decision(data: dict[str, Any]) -> KartotekaLLMDecision:
    raw_tool_calls = data.get("tool_calls")
    tool_calls: list[KartotekaLLMToolCall] = []
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if name not in ALLOWED_TOOL_NAMES:
                continue
            raw_args = raw_call.get("args")
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
            tool_calls.append(
                KartotekaLLMToolCall(
                    name=name,
                    args=args,
                    summary=str(raw_call.get("summary") or "").strip(),
                )
            )
    if not tool_calls:
        tool_calls = [KartotekaLLMToolCall(name="none")]
    return KartotekaLLMDecision(
        status=decision_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        confidence=confidence(data.get("confidence")),
        tool_calls=tool_calls,
    )


def _decision_dict(decision: KartotekaLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [{"name": c.name, "args": c.args, "summary": c.summary} for c in decision.tool_calls],
    }
