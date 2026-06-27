from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ai_server.agents.specialist_llm_shared import (
    compact_tool_result,
    decision_status,
    load_instructions,
    result_status,
    retrieval_context,
)
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.rule_loader import format_loaded_rules, load_rules_for_task
from ai_server.skill_loader import format_loaded_skills, load_skills_for_task
from ai_server.utils import confidence

logger = logging.getLogger(__name__)

ALLOWED_TOOL_NAMES = {"none"}


class DiagnosticAgentLLM(Protocol):
    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        dialog_history: list[dict[str, str]] | None = None,
    ) -> DiagnosticLLMDecisionResult:
        pass

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: DiagnosticLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> DiagnosticLLMFinalResult:
        pass


@dataclass(frozen=True)
class DiagnosticLLMToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class DiagnosticLLMDecision:
    status: str
    answer: str
    tool_calls: list[DiagnosticLLMToolCall]
    confidence: float = 0.5


@dataclass(frozen=True)
class DiagnosticLLMDecisionResult:
    decision: DiagnosticLLMDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticLLMFinalResult:
    status: str
    answer: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class DiagnosticLLMService:
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
    ) -> DiagnosticLLMDecisionResult:
        instructions = load_instructions(manifest)
        loaded_rules = load_rules_for_task(manifest, request=task.request, context=task.context)
        rules_prompt = format_loaded_rules(loaded_rules)
        loaded_skills = load_skills_for_task(manifest, request=task.request, context=task.context)
        skills_prompt = format_loaded_skills(loaded_skills)
        loaded_rules_meta = [
            {
                "id": rule.id,
                "title": rule.title,
                "file": rule.file,
                "reason": rule.reason,
                "matched_context_keys": rule.matched_context_keys,
                "matched_keywords": rule.matched_keywords,
                "matched_statuses": rule.matched_statuses,
                "match_reasons": rule.match_reasons,
                "priority": rule.priority,
            }
            for rule in loaded_rules
        ]
        loaded_skills_meta = [
            {
                "id": skill.id,
                "title": skill.title,
                "file": skill.file,
                "reason": skill.reason,
                "matched_context_keys": skill.matched_context_keys,
                "matched_keywords": skill.matched_keywords,
                "matched_statuses": skill.matched_statuses,
                "match_reasons": skill.match_reasons,
                "priority": skill.priority,
            }
            for skill in loaded_skills
        ]
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _decision_system_prompt(instructions, rules_prompt, skills_prompt)},
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
                            "context": task.context,
                            "current_datetime": datetime.now(UTC).astimezone().isoformat(),
                            "dialog_history": dialog_history or [],
                            "retrieval_context": retrieval_context(retrieval_hits),
                            "loaded_rules": loaded_rules_meta,
                            "loaded_skills": loaded_skills_meta,
                            "tools": [],
                            "tool_results": [compact_tool_result(result) for result in (tool_results or [])],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return DiagnosticLLMDecisionResult(
            decision=_parse_decision(completion.json_content()),
            model_usage=completion.model_usage,
            raw={**completion.raw, "loaded_rules": loaded_rules_meta, "loaded_skills": loaded_skills_meta},
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: DiagnosticLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> DiagnosticLLMFinalResult:
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _compose_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "context": task.context,
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
        return DiagnosticLLMFinalResult(
            status=result_status(parsed.get("status")),
            answer=str(parsed.get("answer") or "").strip() or decision.answer or "Диагностика завершена.",
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def diagnostic_llm_failure_result(message: str, agent_id: str = "diagnostic_agent") -> DiagnosticLLMFinalResult:
    return DiagnosticLLMFinalResult(
        status="failed",
        answer=f"Не смог выполнить диагностику через LLM-агента: {message}",
        model_usage=ModelUsageRecord(
            agent_id=agent_id,
            provider="",
            model="",
            status="error",
            notes=[message],
        ),
    )


def _decision_system_prompt(instructions: str = "", loaded_rules: str = "", loaded_skills: str = "") -> str:
    extra = f"\n\nДополнительные инструкции:\n{instructions}" if instructions else ""
    rules_extra = f"\n\n{loaded_rules}" if loaded_rules else ""
    skills_extra = f"\n\n{loaded_skills}" if loaded_skills else ""
    return (
        "Ты LLM-агент диагностики внутри корпоративного AI-server. "
        "Ты анализируешь feedback, learning events, trace, actions_taken, tool_calls, tool_results, "
        "загруженные rules/skills и финальный ответ. "
        "Ты не выполняешь бизнес-задачи и не вызываешь бизнес-системы. "
        "Если данных не хватает, явно укажи пробелы. "
        "Твоя цель: локализовать вероятный этап сбоя и предложить исправление. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"краткий диагностический вывод",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"none","args":{},"summary":""}]}.'
        f"{extra}{rules_extra}{skills_extra}"
    )


def _compose_system_prompt() -> str:
    return (
        "Ты тот же Агент диагностики. Сформируй итоговый диагностический отчет. "
        "Отчет должен кратко указать: что пошло не так, где вероятный сбой, почему это вероятно, "
        "что исправить и какой regression test добавить. "
        "Не выдумывай факты, которых нет в контексте или initial_decision. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human|failed","answer":"диагностический отчет"}'
    )


def _parse_decision(data: dict[str, Any]) -> DiagnosticLLMDecision:
    raw_tool_calls = data.get("tool_calls")
    tool_calls: list[DiagnosticLLMToolCall] = []
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if name not in ALLOWED_TOOL_NAMES:
                continue
            args = raw_call.get("args") if isinstance(raw_call.get("args"), dict) else {}
            tool_calls.append(
                DiagnosticLLMToolCall(
                    name=name,
                    args=args,
                    summary=str(raw_call.get("summary") or "").strip(),
                )
            )
    if not tool_calls:
        tool_calls = [DiagnosticLLMToolCall(name="none")]
    return DiagnosticLLMDecision(
        status=decision_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        confidence=confidence(data.get("confidence")),
        tool_calls=tool_calls,
    )


def _decision_dict(decision: DiagnosticLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [{"name": call.name, "args": call.args, "summary": call.summary} for call in decision.tool_calls],
    }
