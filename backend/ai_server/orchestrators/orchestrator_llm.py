from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ai_server.agents.specialist_llm_shared import (
    compact_tool_result,
    load_instructions,
    result_status,
    retrieval_context,
)
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.utils import confidence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class OrchestratorDecision:
    status: str
    answer: str
    tool_calls: list[OrchestratorToolCall] = field(default_factory=list)
    confidence: float = 0.5


@dataclass(frozen=True)
class OrchestratorDecisionResult:
    decision: OrchestratorDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorFinalResult:
    answer: str
    status: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class OrchestratorLLM(Protocol):
    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        dialog_history: list[dict[str, str]] | None = None,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        available_skills: list | None = None,
        **kwargs: Any,
    ) -> OrchestratorDecisionResult:
        pass

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: OrchestratorDecision,
        tool_results: list[ToolResult],
        **kwargs: Any,
    ) -> OrchestratorFinalResult:
        pass


# ---------------------------------------------------------------------------
# Production service
# ---------------------------------------------------------------------------


class OrchestratorLLMService:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        dialog_history: list[dict[str, str]] | None = None,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        available_skills: list | None = None,
        **kwargs: Any,
    ) -> OrchestratorDecisionResult:
        local_decision = _explicit_agent_direct_decision(task.request, tool_definitions)
        if local_decision is not None:
            return OrchestratorDecisionResult(
                decision=local_decision,
                model_usage=ModelUsageRecord(
                    agent_id=manifest.id,
                    provider="",
                    model="",
                    status="skipped",
                    notes=["explicit_agent_direct_route"],
                ),
                raw={"source": "explicit_agent_direct_route"},
            )

        local_decision = _vehicle_usage_direct_decision(task.request, tool_definitions)
        if local_decision is not None:
            return OrchestratorDecisionResult(
                decision=local_decision,
                model_usage=ModelUsageRecord(
                    agent_id=manifest.id,
                    provider="",
                    model="",
                    status="skipped",
                    notes=["vehicle_usage_direct_route"],
                ),
                raw={"source": "vehicle_usage_direct_route"},
            )
        local_decision = _admin_panel_clarification_decision(task.request, tool_definitions)
        if local_decision is not None:
            return OrchestratorDecisionResult(
                decision=local_decision,
                model_usage=ModelUsageRecord(
                    agent_id=manifest.id,
                    provider="",
                    model="",
                    status="skipped",
                    notes=["admin_panel_clarification_route"],
                ),
                raw={"source": "admin_panel_clarification_route"},
            )

        instructions = load_instructions(manifest)
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _decide_system_prompt(instructions)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent": {"id": manifest.id, "name": manifest.name},
                            "request": task.request,
                            "user": task.user.model_dump(),
                            "context": {k: v for k, v in task.context.items() if not k.startswith("_")},
                            "current_datetime": datetime.now(UTC).astimezone().isoformat(),
                            "dialog_history": dialog_history or [],
                            "retrieval_context": retrieval_context(retrieval_hits),
                            "tools": tool_definitions,
                            "tool_results": [compact_tool_result(r) for r in (tool_results or [])],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return OrchestratorDecisionResult(
            decision=_parse_decision(completion.json_content(), tool_definitions),
            model_usage=completion.model_usage,
            raw=completion.raw,
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: OrchestratorDecision,
        tool_results: list[ToolResult],
        **kwargs: Any,
    ) -> OrchestratorFinalResult:
        direct_result = _direct_specialist_answer(tool_results)
        if direct_result is not None:
            return direct_result

        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _compose_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "initial_decision_answer": decision.answer,
                            "tool_results": [compact_tool_result(r) for r in tool_results],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        raw = completion.json_content()
        return OrchestratorFinalResult(
            answer=str(raw.get("answer") or "").strip() or decision.answer or "",
            status=result_status(raw.get("status")),
            model_usage=completion.model_usage,
            raw=raw,
        )


def orchestrator_llm_failure_result(message: str) -> OrchestratorFinalResult:
    return OrchestratorFinalResult(
        answer=f"Не смог обработать запрос через Переговорщика: {message}",
        status="failed",
        model_usage=ModelUsageRecord(
            agent_id="internal_orchestrator",
            provider="",
            model="",
            status="error",
            notes=[message],
        ),
    )


def _admin_panel_clarification_decision(
    request: str,
    tool_definitions: list[dict[str, Any]],
) -> OrchestratorDecision | None:
    text = _strip_synthetic_prefix(request).casefold()
    if _looks_like_task_close_admin_panel_request(text) or _looks_like_vehicle_usage_admin_panel_request(text):
        return None
    if not _looks_like_ambiguous_admin_panel_request(text):
        return None
    return OrchestratorDecision(
        status="needs_clarification",
        answer="Уточните, какую панель показать: закрытие задач или отчет по машинам и людям.",
        tool_calls=[OrchestratorToolCall(name="none", args={}, summary="ambiguous admin panel clarification")],
        confidence=0.86,
    )


def _vehicle_usage_direct_decision(
    request: str,
    tool_definitions: list[dict[str, Any]],
) -> OrchestratorDecision | None:
    available_tools = {str(tool.get("name") or "") for tool in tool_definitions or []}
    if "call_specialist" not in available_tools:
        return None
    text = _strip_synthetic_prefix(request).casefold()
    if _explicit_agent_prefix(text) is not None:
        return None
    if not _looks_like_vehicle_usage_admin_panel_request(text):
        return None
    return OrchestratorDecision(
        status="completed",
        answer="Передаю запрос Логисту.",
        tool_calls=[
            OrchestratorToolCall(
                name="call_specialist",
                args={"specialist_id": "logistics", "request": request},
                summary="vehicle usage direct route",
            )
        ],
        confidence=0.92,
    )


def _explicit_agent_direct_decision(
    request: str,
    tool_definitions: list[dict[str, Any]],
) -> OrchestratorDecision | None:
    available_tools = {str(tool.get("name") or "") for tool in tool_definitions or []}
    if "call_specialist" not in available_tools:
        return None
    text = _strip_synthetic_prefix(request).casefold()
    target = _explicit_agent_prefix(text)
    if target is None:
        return None
    return OrchestratorDecision(
        status="completed",
        answer="Передаю запрос профильному специалисту.",
        tool_calls=[
            OrchestratorToolCall(
                name="call_specialist",
                args={"specialist_id": target, "request": request},
                summary=f"explicit agent prefix route to {target}",
            )
        ],
        confidence=0.95,
    )


def _explicit_agent_prefix(lowered: str) -> str | None:
    text = str(lowered or "").strip()
    normalized = re.sub(r"^[\s\ufeff]+", "", text)
    for prefix, specialist_id in (
        ("битрикс", "bitrix24"),
        ("bitrix", "bitrix24"),
        ("логист", "logistics"),
        ("диагност", "diagnost"),
        ("пто", "pto"),
    ):
        if re.match(rf"^{re.escape(prefix)}(?:\b|[\s:,.!?-])", normalized, flags=re.IGNORECASE):
            return specialist_id
    return None


def _strip_synthetic_prefix(request: str) -> str:
    text = str(request or "").removeprefix("\ufeff").strip()
    return re.sub(r"^\[[^\]]+\]\s*", "", text).strip()


def _looks_like_ambiguous_admin_panel_request(lowered: str) -> bool:
    return (
        ("админ" in lowered and "панел" in lowered)
        or ("спис" in lowered and ("оператор" in lowered or "пользовател" in lowered))
        or ("кто" in lowered and "оператор" in lowered)
    )


def _looks_like_task_close_admin_panel_request(lowered: str) -> bool:
    task_close_domain = (
        "контролируем" in lowered
        or ("автозакрыт" in lowered and "задач" in lowered)
        or ("контрол" in lowered and ("закрыт" in lowered or "задач" in lowered))
        or ("закрыт" in lowered and "задач" in lowered)
    )
    return task_close_domain and (
        "настрой" in lowered
        or "панел" in lowered
        or "спис" in lowered
        or "кто" in lowered
        or "оператор" in lowered
        or "пользовател" in lowered
        or "автозакрыт" in lowered
    )


def _looks_like_vehicle_usage_admin_panel_request(lowered: str) -> bool:
    vehicle_domain = (
        "машин" in lowered or "автомоб" in lowered or "люд" in lowered or "сотрудник" in lowered or "логист" in lowered
    )
    return vehicle_domain and ("отчет" in lowered or "оператор" in lowered or "пользовател" in lowered)


def _direct_specialist_answer(tool_results: list[ToolResult]) -> OrchestratorFinalResult | None:
    for result in reversed(tool_results):
        if result.tool != "call_specialist" or str(result.status) != "ok":
            continue
        data = result.data if isinstance(result.data, dict) else {}
        answer = str(data.get("answer") or "").strip()
        if not answer:
            continue
        status = result_status(data.get("status"))
        return OrchestratorFinalResult(
            answer=answer,
            status=status,
            model_usage=ModelUsageRecord(
                agent_id="internal_orchestrator",
                provider="local",
                model="specialist_answer_passthrough",
                status="skipped",
                notes=["single_specialist_answer_passthrough"],
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _decide_system_prompt(instructions: str = "") -> str:
    extra = f"\n\n{instructions}" if instructions else ""
    return (
        "Ты Переговорщик — старший AI-агент корпоративного AI-server. "
        "Ты посредник между людьми и специалистами-субагентами. "
        'Твоя задача: понять запрос, выбрать нужного специалиста (call_specialist) или ответить самому (tool_calls=[{"name":"none"}]). '
        "Маршрутизируй к специалисту только если запрос явно относится к его зоне ответственности (см. описания инструментов). "
        "Если запрос можно обработать самому (общий вопрос, пояснение, нет подходящего специалиста) — отвечай напрямую. "
        "Если запрос неоднозначен и неясно в какой системе его выполнять — уточни ТОЛЬКО выбор системы/специалиста "
        "(например: «В какой системе создать задачу?»). Параметры запроса (название, описание, сроки и т.п.) "
        "собирает специалист — не спрашивай их сам. "
        "Если предыдущие tool_results уже содержат нужные данные — не вызывай те же инструменты снова. "
        "dialog_history — только контекст. Write-операции и доменные задачи требуют вызова специалиста, "
        "даже если параметры видны из истории. "
        "Если context.pending_specialist установлен — это специалист, ожидающий ответа пользователя. "
        "Передай следующее сообщение пользователя этому специалисту через call_specialist без изменений, "
        "если пользователь явно не переключился на другую тему. "
        "Никогда не притворяйся специалистом и не выполняй их доменную работу сам. "
        "Если task.context содержит _source — задачу инициировал специалист. "
        "Читай context._intent и принимай решение какой инструмент вызвать: "
        "  _intent=deliver_to_dialog → call_specialist(bitrix24) для отправки в context.dialog_id; "
        "  _intent=escalate → call_specialist(bitrix24) для уведомления context.admin_user_ids. "
        "Верни только JSON-объект без markdown. Формат: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"предварительный ответ",'
        '"tool_calls":[{"name":"call_specialist|schedule_task|manage_suspended|none","args":{...},"summary":""}],'
        '"confidence":0.0}.'
        f"{extra}"
    )


def _compose_system_prompt() -> str:
    return (
        "Ты Переговорщик — составляешь финальный ответ пользователю на основе результатов специалистов. "
        "Если tool_results содержат ответы специалистов — объедини их в единый связный ответ. "
        "Если tool_results пусты или содержат только ошибки — используй initial_decision_answer. "
        "Никогда не выдумывай конкретные ID, ссылки, номера задач или результаты операций, "
        "которых нет в tool_results. "
        "Не дублируй информацию. Пиши от первого лица. Не раскрывай внутренние tool calls и имена специалистов. "
        'Верни только JSON-объект без markdown: {"answer":"...","status":"completed|needs_clarification|failed"}.'
    )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_decision(data: dict[str, Any], tool_definitions: list[dict[str, Any]]) -> OrchestratorDecision:
    known_tools = {td["name"] for td in tool_definitions} | {"none"}
    raw_calls = data.get("tool_calls")
    tool_calls: list[OrchestratorToolCall] = []
    if isinstance(raw_calls, list):
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name not in known_tools:
                continue
            tool_calls.append(
                OrchestratorToolCall(
                    name=name,
                    args=item.get("args") or {},
                    summary=str(item.get("summary") or ""),
                )
            )
    if not tool_calls:
        tool_calls = [OrchestratorToolCall(name="none")]
    return OrchestratorDecision(
        status=_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        tool_calls=tool_calls,
        confidence=confidence(data.get("confidence")),
    )


def _status(value: object) -> str:
    s = str(value or "completed").strip()
    return s if s in {"completed", "needs_clarification", "failed"} else "completed"
