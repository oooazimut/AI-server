from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import ValidationError

from ai_server.agents.ports import AgentDialogStorePort, AgentQueuePort, ChannelPort, SchedulerPort
from ai_server.learning import LearningEventRecorder
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ScheduledTask, ToolResult, ToolStatus
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
from ai_server.technical_footer import TechnicalFooterService, append_footer

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
        channels: dict[str, ChannelPort] | None = None,
        footer_service: TechnicalFooterService | None = None,
        learning_recorder: LearningEventRecorder | None = None,
    ) -> None:
        self.manifests = manifests
        self.specialists = specialists or build_specialist_registry(manifests)
        self.orchestrator_llm = orchestrator_llm or OrchestratorLLMService()
        self.scheduler = scheduler
        self.store = store
        self.retriever = retriever
        self._channels: dict[str, ChannelPort] = channels or {}
        self._footer_svc = footer_service
        self._learning_recorder = learning_recorder
        self._manifest = manifest_by_id(manifests, "internal_orchestrator") or _dummy_manifest()

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        manifests: list[AgentManifest] | None = None,
        orchestrator_llm: OrchestratorLLM | None = None,
        orchestrator_store: AgentDialogStorePort | None = None,
        orchestrator_retriever: HybridKnowledgeRetriever | None = None,
        channels: dict[str, ChannelPort] | None = None,
        footer_service: TechnicalFooterService | None = None,
        learning_recorder: LearningEventRecorder | None = None,
        **specialist_deps: Any,
    ) -> InternalOrchestrator:
        _manifests = manifests or []
        if not specialist_deps.get("bitrix_bot"):
            specialist_deps["bitrix_bot"] = specialist_deps.get("bitrix_client")
        specialists = build_specialist_registry(
            _manifests,
            audience="employee",
            **{k: v for k, v in specialist_deps.items() if v is not None},
        )
        return cls(
            _manifests,
            specialists=specialists,
            orchestrator_llm=orchestrator_llm,
            scheduler=specialist_deps.get("scheduler"),
            store=orchestrator_store,
            retriever=orchestrator_retriever,
            channels=channels,
            footer_service=footer_service,
            learning_recorder=learning_recorder,
        )

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
            self._apply_agent_scheduled_tasks(sr.scheduled_tasks)
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
        t_start = time.monotonic()
        result = await self._handle_core(task)
        elapsed_ms = {"total_ms": round((time.monotonic() - t_start) * 1000, 1)}
        await self._send_to_channel(task, result)
        self._record_learning(task, result, elapsed_ms=elapsed_ms)
        return result

    async def _handle_core(self, task: AgentTask) -> AgentResult:
        dialog_key: str = task.context.get("dialog_key") or ""
        if self.store is not None and dialog_key:
            dialog_history: list[dict] = await self.store.load_turns(dialog_key, limit=20)
        else:
            dialog_history = list(task.context.get("dialog_history") or [])

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

    async def _send_to_channel(self, task: AgentTask, result: AgentResult) -> None:
        channel_id = task.context.get("channel_id", "")
        recipient_id = task.context.get("recipient_id", "")
        if not channel_id or not recipient_id:
            return
        channel = self._channels.get(channel_id)
        if channel is None:
            return
        footer = ""
        if self._footer_svc and result.answer:
            user_id_raw = task.user.id if task.user else None
            user_id = int(user_id_raw) if user_id_raw and str(user_id_raw).isdigit() else None
            try:
                footer = await self._footer_svc.build_for_agent_result(
                    result, user_id=user_id, channel=f"{channel_id}_chat"
                )
            except Exception:
                logger.exception("Footer build failed")
        body = append_footer(result.answer, footer) if result.answer else ""
        if body:
            try:
                await channel.send(recipient_id, body)
            except Exception:
                logger.exception("Channel send failed for channel=%s recipient=%s", channel_id, recipient_id)

    def _record_learning(
        self, task: AgentTask, result: AgentResult, *, elapsed_ms: dict[str, float] | None = None
    ) -> None:
        if self._learning_recorder is None:
            return
        try:
            self._learning_recorder.record_agent_result(
                task,
                result,
                metadata={"dialog_key": task.context.get("dialog_key", "")},
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            logger.exception("Learning recording failed")

    def _apply_agent_scheduled_tasks(self, tasks: list[ScheduledTask]) -> None:
        if not tasks or self.scheduler is None:
            return
        _orch = self
        for sched in tasks:
            if sched.cancel:
                self.scheduler.remove_job(sched.agent_id, sched.job_id)
            elif sched.task is not None:
                _task = sched.task

                async def _run(_t: AgentTask = _task, _o: InternalOrchestrator = _orch) -> None:
                    await _o.handle(_t)

                try:
                    self.scheduler.schedule_callback(sched.agent_id, sched.job_id, sched.trigger, _run)
                except Exception:
                    logger.exception("Failed to schedule task job_id=%s agent=%s", sched.job_id, sched.agent_id)

    async def run(self, queue: AgentQueuePort) -> None:
        """Queue consumer loop for the orchestrator.

        Handles two message types:
        - "task"   — new request from channel or scheduler → handle(task) → _send_to_channel()
        - "result" — proactive result from a specialist (e.g. morning_proposals) → _send_to_channel()

        Sprint 21: internal dispatch to specialists remains synchronous (direct calls in handle()).
        Sprint 22 will introduce async specialist dispatch via correlation_id.
        """
        _poll_interval = 0.1
        while True:
            message = await queue.claim_next("orchestrator")
            if message is None:
                await asyncio.sleep(_poll_interval)
                continue
            msg_id = str(message.get("id") or "")
            try:
                msg_type = str(message.get("type") or "")
                if msg_type in ("task", "bitrix_chat"):
                    try:
                        task = AgentTask.model_validate(message["payload"])
                    except (KeyError, ValidationError) as exc:
                        logger.warning("Orchestrator: invalid task message %s: %s", msg_id, exc)
                        await queue.nack(msg_id, error=f"invalid message: {exc}")
                        continue
                    await self.handle(task)
                elif msg_type == "result":
                    try:
                        result = AgentResult.model_validate(message["payload"])
                    except (KeyError, ValidationError) as exc:
                        logger.warning("Orchestrator: invalid result message %s: %s", msg_id, exc)
                        await queue.nack(msg_id, error=f"invalid message: {exc}")
                        continue
                    routing = message.get("routing") or {}
                    if routing.get("channel_id") and routing.get("recipient_id"):
                        stub_task = AgentTask(
                            task_id="",
                            request="",
                            context={
                                "channel_id": routing["channel_id"],
                                "recipient_id": routing["recipient_id"],
                                "dialog_key": routing.get("dialog_key") or "",
                            },
                        )
                        await self._send_to_channel(stub_task, result)
                await queue.ack(msg_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Orchestrator failed processing message %s", msg_id)
                await queue.nack(msg_id, error=f"{type(exc).__name__}: {exc}")


def _dummy_manifest() -> AgentManifest:
    return AgentManifest(
        id="internal_orchestrator",
        name="Переговорщик",
        kind="orchestrator",
        description="Старший AI-агент. Посредник между людьми и специалистами.",
    )
