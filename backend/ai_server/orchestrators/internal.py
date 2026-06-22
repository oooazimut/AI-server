from __future__ import annotations

import asyncio
import logging
from typing import Any

from ai_server.agents.ports import AgentDialogStorePort, SchedulerPort
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ToolResult, ToolStatus
from ai_server.orchestrators.orchestrator_llm import (
    OrchestratorDecision,
    OrchestratorDecisionResult,
    OrchestratorLLM,
    OrchestratorLLMService,
    apply_scheduled_tasks,
    orchestrator_llm_failure_result,
)
from ai_server.retrieval import HybridKnowledgeRetriever, RetrievalHit
from ai_server.specialists import Specialist, build_specialist_registry, manifest_by_id

logger = logging.getLogger(__name__)

_MAX_AGENT_STEPS = 4


class InternalOrchestrator:
    def __init__(
        self,
        manifests: list[AgentManifest],
        specialists: dict[str, Specialist] | None = None,
        *,
        orchestrator_llm: OrchestratorLLM | None = None,
        scheduler: SchedulerPort | None = None,
        store: AgentDialogStorePort | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
    ) -> None:
        self.manifests = manifests
        self.specialists = specialists or build_specialist_registry(manifests)
        self.orchestrator_llm = orchestrator_llm or OrchestratorLLMService()
        self.scheduler = scheduler
        self.store = store
        self.retriever = retriever
        self._manifest = manifest_by_id(manifests, "internal_orchestrator") or _dummy_manifest()

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": f"call_{m.id}",
                "description": m.handoff_description or f"Вызвать специалиста {m.name}",
                "parameters": {"request": {"type": "string", "description": "Запрос для специалиста"}},
            }
            for m in self.manifests
            if m.kind == "specialist" and m.id in self.specialists
        ]

    async def _execute_tool_call(
        self, tool_call: Any, task: AgentTask
    ) -> tuple[ToolResult, ActionRecord, list[ActionRecord]]:
        if tool_call.name.startswith("call_"):
            specialist_id = tool_call.name[len("call_") :]
            specialist = self.specialists.get(specialist_id)
            if specialist is None:
                result = ToolResult(
                    status=ToolStatus.ERROR,
                    tool=tool_call.name,
                    error=f"Специалист '{specialist_id}' не найден",
                )
                return (
                    result,
                    ActionRecord(
                        name="delegate_to_specialist",
                        status="error",
                        details={"specialist": specialist_id, "error": result.error},
                    ),
                    [],
                )
            sub_task = AgentTask(
                task_id=task.task_id,
                request=tool_call.args.get("request") or task.request,
                user=task.user,
                context=task.context,
            )
            try:
                sr = await specialist.handle(sub_task)
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                result = ToolResult(status=ToolStatus.ERROR, tool=tool_call.name, error=err)
                return (
                    result,
                    ActionRecord(
                        name="delegate_to_specialist",
                        status="error",
                        details={"specialist": specialist_id, "error": err},
                    ),
                    [],
                )
            result = ToolResult(
                status=ToolStatus.OK,
                tool=tool_call.name,
                data={"specialist": specialist_id, "answer": sr.answer, "status": sr.status},
            )
            return (
                result,
                ActionRecord(
                    name="delegate_to_specialist",
                    status="completed",
                    details={"specialist": specialist_id},
                ),
                list(sr.actions_requiring_approval),
            )
        result = ToolResult(
            status=ToolStatus.INVALID_TOOL_CALL,
            tool=tool_call.name,
            error=f"Неизвестный инструмент оркестратора: {tool_call.name}",
        )
        return result, ActionRecord(name=tool_call.name, status="error", details={"error": result.error}), []

    async def handle(self, task: AgentTask) -> AgentResult:
        if task.context.get("pending_action"):
            return await self._handle_pending_action(task)

        # Загрузить историю диалога
        dialog_key: str = task.context.get("dialog_key") or ""
        if self.store is not None and dialog_key:
            dialog_history: list[dict] = await self.store.load_turns(dialog_key, limit=20)
        else:
            dialog_history = list(task.context.get("dialog_history") or [])

        # RAG — поиск по базе знаний оркестратора
        retrieval_hits: list[RetrievalHit] = []
        if self.retriever is not None:
            retrieval_hits = self.retriever.search(self._manifest, task.request, limit=3)

        all_actions: list[ActionRecord] = []
        all_model_usage = []
        tool_results: list[ToolResult] = []
        specialist_ids: list[str] = []
        approval_actions: list[ActionRecord] = []
        decision: OrchestratorDecision | None = None
        decision_results: list[OrchestratorDecisionResult] = []

        for step in range(1, _MAX_AGENT_STEPS + 1):
            try:
                dr = await self.orchestrator_llm.decide(
                    manifest=self._manifest,
                    task=task,
                    dialog_history=dialog_history,
                    retrieval_hits=retrieval_hits,
                    tool_definitions=self.tool_definitions(),
                    tool_results=list(tool_results),
                )
            except Exception as exc:
                return AgentResult(
                    status="failed",
                    agent_id="internal_orchestrator",
                    answer=f"Не смог обработать запрос через LLM-оркестратор: {type(exc).__name__}: {exc}",
                    actions_taken=[
                        *all_actions,
                        ActionRecord(
                            name="orchestrator_llm_decision",
                            status="error",
                            details={"step": step, "error": f"{type(exc).__name__}: {exc}"},
                        ),
                    ],
                    model_usage=all_model_usage,
                    confidence=0.0,
                )

            decision_results.append(dr)
            decision = dr.decision
            all_model_usage.append(dr.model_usage)
            all_actions.append(
                ActionRecord(
                    name="orchestrator_llm_decision",
                    status=decision.status,
                    details={
                        "step": step,
                        "tool_calls": [{"name": tc.name, "summary": tc.summary} for tc in decision.tool_calls],
                        "confidence": decision.confidence,
                    },
                )
            )
            all_actions.extend(apply_scheduled_tasks(decision.scheduled_tasks, self.scheduler))

            executable = [tc for tc in decision.tool_calls if tc.name != "none"]
            if not executable:
                break

            # Параллельное выполнение инструментов (специалистов)
            raw = await asyncio.gather(
                *[self._execute_tool_call(tc, task) for tc in executable],
                return_exceptions=True,
            )
            for tc, item in zip(executable, raw, strict=False):
                if isinstance(item, Exception):
                    all_actions.append(
                        ActionRecord(
                            name="delegate_to_specialist",
                            status="error",
                            details={"error": f"{type(item).__name__}: {item}"},
                        )
                    )
                else:
                    tr, action, approvals = item
                    tool_results.append(tr)
                    all_actions.append(action)
                    approval_actions.extend(approvals)
                    if tr.status == ToolStatus.OK and tc.name.startswith("call_"):
                        specialist_ids.append(tc.name[len("call_") :])

        if decision is None:
            failure = orchestrator_llm_failure_result("пустой цикл решений оркестратора")
            return AgentResult(
                status="failed",
                agent_id="internal_orchestrator",
                answer=failure.answer,
                actions_taken=all_actions,
                model_usage=[failure.model_usage],
                confidence=0.0,
            )

        # Финальный ответ
        try:
            final = await self.orchestrator_llm.compose(
                manifest=self._manifest,
                task=task,
                decision=decision,
                tool_results=tool_results,
            )
        except Exception as exc:
            failure = orchestrator_llm_failure_result(f"{type(exc).__name__}: {exc}")
            return AgentResult(
                status="failed",
                agent_id="internal_orchestrator",
                answer=failure.answer,
                actions_taken=[
                    *all_actions,
                    ActionRecord(
                        name="orchestrator_llm_compose",
                        status="error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    ),
                ],
                model_usage=[*all_model_usage, failure.model_usage],
                confidence=0.0,
            )

        all_model_usage.append(final.model_usage)
        all_actions.append(
            ActionRecord(
                name="orchestrator_llm_compose",
                status=final.status,
                details={"specialists_used": specialist_ids},
            )
        )

        if self.store is not None and dialog_key and final.answer:
            await self.store.append_turn(dialog_key, task.request, final.answer)

        # Если любой специалист требует подтверждения — итоговый статус needs_human
        effective_status = "needs_human" if approval_actions else final.status

        return AgentResult(
            status=effective_status,
            agent_id="internal_orchestrator",
            answer=final.answer,
            actions_taken=all_actions,
            actions_requiring_approval=approval_actions,
            model_usage=all_model_usage,
            handoff_to=specialist_ids,
            confidence=decision.confidence,
        )

    async def _handle_pending_action(self, task: AgentTask) -> AgentResult:
        pending_data = task.context["pending_action"]
        default_specialist_id = next((m.id for m in self.manifests if m.kind == "specialist"), "")
        specialist_id = (
            pending_data.get("specialist_id") or default_specialist_id
            if isinstance(pending_data, dict)
            else default_specialist_id
        )
        specialist = self.specialists.get(specialist_id)
        if specialist is not None:
            sr = await specialist.handle(task)
            return AgentResult(
                status=sr.status,
                agent_id="internal_orchestrator",
                answer=sr.answer,
                artifacts=sr.artifacts,
                actions_taken=[
                    ActionRecord(
                        name="orchestrator_pending_route",
                        status="completed",
                        details={"handoff_to": specialist_id, "reason": "pending_action"},
                    ),
                    *sr.actions_taken,
                ],
                actions_requiring_approval=sr.actions_requiring_approval,
                model_usage=sr.model_usage,
                handoff_to=[specialist_id],
                confidence=sr.confidence,
                logs=sr.logs,
            )
        return AgentResult(
            status="failed",
            agent_id="internal_orchestrator",
            answer="Специалист для обработки ожидающего действия не найден.",
            actions_taken=[
                ActionRecord(
                    name="orchestrator_pending_route",
                    status="error",
                    details={"handoff_to": specialist_id, "reason": "specialist_not_found"},
                )
            ],
            confidence=0.0,
        )


def _dummy_manifest() -> AgentManifest:
    return AgentManifest(
        id="internal_orchestrator",
        name="Переговорщик",
        kind="orchestrator",
        description="Старший AI-агент. Посредник между людьми и специалистами.",
    )
