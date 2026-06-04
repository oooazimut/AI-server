from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from ai_server.llm import LLMClient, LLMError, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit


ALLOWED_TOOL_NAMES = {"task_search", "task_create_draft", "portal_search", "none"}
RESULT_STATUSES = {"completed", "needs_clarification", "needs_human", "failed"}


class BitrixAgentLLM(Protocol):
    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
    ) -> "BitrixLLMDecisionResult":
        pass

    async def compose(
        self,
        *,
        task: AgentTask,
        decision: "BitrixLLMDecision",
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]],
    ) -> "BitrixLLMFinalResult":
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
    ) -> BitrixLLMDecisionResult:
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _decision_system_prompt()},
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
                            "current_datetime": datetime.now(timezone.utc).astimezone().isoformat(),
                            "retrieval_context": _retrieval_context(retrieval_hits),
                            "tools": _allowed_tool_definitions(tool_definitions),
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
        task: AgentTask,
        decision: BitrixLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]],
    ) -> BitrixLLMFinalResult:
        completion = await self.client.complete(
            agent_id="bitrix24",
            messages=[
                {"role": "system", "content": _compose_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "user": task.user.model_dump(),
                            "initial_decision": _decision_dict(decision),
                            "tool_results": [_compact_tool_result(result) for result in tool_results],
                            "approval_actions": approval_actions,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        parsed = completion.json_content()
        return BitrixLLMFinalResult(
            status=_result_status(parsed.get("status")),
            answer=str(parsed.get("answer") or "").strip() or "Готово.",
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def llm_failure_result(message: str) -> BitrixLLMFinalResult:
    return BitrixLLMFinalResult(
        status="failed",
        answer=f"Не смог обработать Bitrix-запрос через LLM-субагента: {message}",
        model_usage=ModelUsageRecord(
            agent_id="bitrix24",
            provider="",
            model="",
            status="error",
            notes=[message],
        ),
    )


def _decision_system_prompt() -> str:
    return (
        "Ты LLM-субагент Bitrix24 внутри корпоративного AI-server. "
        "Оркестратор уже передал тебе запрос человека. "
        "Ты не выполняешь действия сам: выбираешь один или несколько tools. "
        "Backend выполнит tools и применит policy/OAuth/подтверждения. "
        "Запрещено отвечать так, будто действие уже выполнено, если нужен write-tool. "
        "Верни только JSON-объект без markdown. Формат: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"короткий предварительный ответ",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"task_search|task_create_draft|portal_search|none","args":{},"summary":""}]}. '
        "Перед каждым tool_call сам проверь, хватает ли данных для его корректного вызова. "
        "Если данных не хватает, не вызывай tool: верни status=needs_clarification, tool_calls=[{\"name\":\"none\"}], "
        "а в answer задай короткий уточняющий вопрос. "
        "Для поиска задач используй task_search. Для создания задачи используй task_create_draft. "
        "Для task_create_draft именно ты распознаёшь title, responsible_id/responsible_query/responsible_self, "
        "group_id/project_query, deadline_iso или no_deadline. "
        "Если пользователь сказал относительный срок, вычисли deadline_iso сам по current_datetime. "
        "Если срок не указан, применяй правила из retrieval_context; если правило неясно, спроси уточнение. "
        "Не вызывай task_create_draft без title и одного из responsible_id/responsible_query/responsible_self. "
        "Для поиска документов/файлов используй portal_search. Если данных не хватает, status=needs_clarification."
    )


def _compose_system_prompt() -> str:
    return (
        "Ты тот же LLM-субагент Bitrix24. Сформируй итоговый ответ человеку по результатам tools. "
        "Не выдумывай данные, которых нет в tool_results. "
        "Если есть approval_actions, скажи, что действие подготовлено и требуется подтверждение. "
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
        status=_decision_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        confidence=_confidence(data.get("confidence")),
        tool_calls=tool_calls,
    )


def _decision_status(value: object) -> str:
    status = str(value or "completed").strip()
    return status if status in {"completed", "needs_clarification", "needs_human"} else "completed"


def _result_status(value: object) -> str:
    status = str(value or "completed").strip()
    return status if status in RESULT_STATUSES else "completed"


def _confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(number, 0.0), 1.0)


def _retrieval_context(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    context = []
    for hit in hits[:5]:
        context.append(
            {
                "topic": hit.chunk.topic,
                "section": hit.chunk.section,
                "score": hit.score,
                "text": hit.chunk.text[:1200],
            }
        )
    return context


def _allowed_tool_definitions(definitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [definition for definition in definitions if definition.get("name") in ALLOWED_TOOL_NAMES]


def _decision_dict(decision: BitrixLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [
            {"name": call.name, "args": call.args, "summary": call.summary}
            for call in decision.tool_calls
        ],
    }


def _compact_tool_result(result: ToolResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "tool": result.tool,
        "data": result.data,
        "error": result.error,
    }
