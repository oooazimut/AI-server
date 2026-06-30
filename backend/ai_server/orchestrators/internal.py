from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import ValidationError

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.ports import AgentQueuePort, ChannelPort, OrchestratorStorePort, SchedulerPort
from ai_server.learning import LearningEventRecorder
from ai_server.models import (
    ActionRecord,
    AgentManifest,
    AgentResult,
    AgentTask,
    Artifact,
    ModelUsageRecord,
    ScheduledTask,
    ToolResult,
    ToolStatus,
)
from ai_server.orchestrators.orchestrator_llm import (
    OrchestratorFinalResult,
    OrchestratorLLM,
    OrchestratorLLMService,
    orchestrator_llm_failure_result,
)
from ai_server.orchestrators.tools import CallSpecialistTool, ManageSuspendedTool, ScheduleTaskTool
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.specialists import Specialist, build_specialist_registry
from ai_server.technical_footer import TechnicalFooterService, append_footer
from ai_server.tracing import TraceRecorder, parent_span_id_from_task

logger = logging.getLogger(__name__)


class InternalOrchestrator(BaseSpecialist):
    """Senior agent: routes requests to specialists and synthesises answers.

    Inherits the decide→execute→compose loop from BaseSpecialist.
    Overrides handle() to add post-compose side-effects (channel delivery, learning).
    Overrides run() to handle "result" messages from proactive specialists.
    """

    action_prefix = "orchestrator"
    max_steps = 4

    def __init__(
        self,
        manifest: AgentManifest | list[AgentManifest],
        *,
        manifests: list[AgentManifest] | None = None,
        specialists: dict[str, Specialist] | None = None,
        agent_tools: list | None = None,
        llm: OrchestratorLLM | None = None,
        orchestrator_llm: OrchestratorLLM | None = None,
        store: OrchestratorStorePort | None = None,
        scheduler: SchedulerPort | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        channels: dict[str, ChannelPort] | None = None,
        footer_service: TechnicalFooterService | None = None,
        learning_recorder: LearningEventRecorder | None = None,
        trace_recorder: TraceRecorder | None = None,
        feedback_loop: Any = None,
        diagnostic_specialists: dict[str, Specialist] | None = None,
        settings: Any = None,
    ) -> None:
        _manifests = manifests or []
        _manifest = manifest
        if isinstance(manifest, list):
            _manifests = manifest
            _manifest = _dummy_manifest()
        if orchestrator_llm is not None and llm is None:
            llm = orchestrator_llm
        if agent_tools is None and specialists is not None:
            agent_tools = [
                CallSpecialistTool(
                    specialists,
                    _manifests,
                    scheduler=scheduler,
                    store=store,
                )
            ]
        super().__init__(
            _manifest,
            agent_tools=agent_tools,
            llm=llm or OrchestratorLLMService(),
            store=store,
            scheduler=scheduler,
            retriever=retriever,
            trace_recorder=trace_recorder,
        )
        self._channels: dict[str, ChannelPort] = channels or {}
        self._footer_svc = footer_service
        self._learning_recorder = learning_recorder
        self._feedback_loop = feedback_loop
        self._diagnostic_specialists = diagnostic_specialists or (
            specialists if specialists and "diagnostic_agent" in specialists else {}
        )
        self._settings = settings

    # ------------------------------------------------------------------
    # BaseSpecialist hooks
    # ------------------------------------------------------------------

    def _logs(self) -> list[str]:
        return []

    def _llm_failure_result(self, message: str) -> OrchestratorFinalResult:
        return orchestrator_llm_failure_result(message)

    async def _load_extra_context(self, task: AgentTask) -> tuple[AgentTask, dict[str, Any]]:
        """Inject pending_specialist from KV so LLM sees it in task.context."""
        dialog_key = str(task.context.get("dialog_key") or "")
        if self.store is not None and dialog_key and hasattr(self.store, "get_kv"):
            try:
                pending = await self.store.get_kv(dialog_key, "pending_specialist")  # type: ignore[attr-defined]
                if pending:
                    task = task.model_copy(update={"context": {**task.context, "pending_specialist": pending}})
            except Exception:
                logger.exception("_load_extra_context: failed to load pending_specialist")
        return task, {}

    async def _execute_tool_call(
        self,
        tool_call: Any,
        task: AgentTask,
    ) -> tuple[ToolResult | None, ActionRecord | None, list[ActionRecord]]:
        if tool_call.name != "call_diagnostic_agent":
            return await super()._execute_tool_call(tool_call, task)
        diagnostic = self._diagnostic_specialists.get("diagnostic_agent")
        if diagnostic is None:
            result = ToolResult(
                status=ToolStatus.ERROR,
                tool=tool_call.name,
                error="diagnostic_agent_not_configured",
            )
            return (
                result,
                ActionRecord(name="delegate_to_specialist", status="error", details=result.model_dump()),
                [],
            )
        sub_task = task.model_copy(update={"request": str(tool_call.args.get("request") or task.request)})
        result = await diagnostic.handle(sub_task)
        return (
            ToolResult(
                status=ToolStatus.OK,
                tool=tool_call.name,
                data={
                    "specialist": "diagnostic_agent",
                    "answer": result.answer,
                    "status": result.status,
                    "artifacts": [artifact.model_dump() for artifact in result.artifacts],
                },
            ),
            ActionRecord(
                name="delegate_to_specialist",
                status="completed",
                details={"specialist": "diagnostic_agent"},
            ),
            list(result.actions_requiring_approval),
        )

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------

    async def handle(self, task: AgentTask) -> AgentResult:
        report_command = (
            task.context.get("error_report_request")
            if isinstance(task.context.get("error_report_request"), dict)
            else _error_report_command(task.request)
        )
        if report_command:
            result = await self._handle_error_report_command(task, report_command)
            await self._send_to_channel(task, result, learning_record=None)
            return result

        t_start = time.monotonic()
        dialog_key = task.context.get("dialog_key", "")
        logger.info("Orchestrator.handle: start task_id=%s dialog_key=%s", task.task_id, dialog_key)
        result = await super().handle(task)
        # Extract specialist IDs called during this turn for handoff_to
        specialist_ids = [
            a.details.get("data", {}).get("specialist")
            for a in result.actions_taken
            if a.name == "call_specialist" and a.status == "ok"
        ]
        specialist_ids = [s for s in specialist_ids if s]
        if specialist_ids:
            result = result.model_copy(update={"handoff_to": specialist_ids})
        elapsed_ms = {"total_ms": round((time.monotonic() - t_start) * 1000, 1)}
        logger.info(
            "Orchestrator.handle: done task_id=%s dialog_key=%s elapsed_ms=%.0f status=%s",
            task.task_id,
            dialog_key,
            elapsed_ms["total_ms"],
            result.status,
        )
        learning_record = self._record_learning(task, result, elapsed_ms=elapsed_ms)
        await self._send_to_channel(task, result, learning_record=learning_record)
        return result

    async def run(self, queue: AgentQueuePort) -> None:
        """Queue consumer loop.

        Handles two message types:
        - "task" / "bitrix_chat" — new request → handle()
        - "result" — proactive result from a specialist → _send_to_channel()
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
                _payload = message.get("payload") or {}
                _ctx = _payload.get("context") or {} if isinstance(_payload, dict) else {}
                _dlg = str(_ctx.get("dialog_key") or "") if isinstance(_ctx, dict) else ""
                logger.info("Orchestrator.run: claimed msg_id=%s type=%s dialog_key=%s", msg_id, msg_type, _dlg)
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

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        manifest: AgentManifest | None,
        *,
        manifests: list[AgentManifest] | None = None,
        orchestrator_llm: OrchestratorLLM | None = None,
        orchestrator_store: OrchestratorStorePort | None = None,
        orchestrator_retriever: HybridKnowledgeRetriever | None = None,
        channels: dict[str, ChannelPort] | None = None,
        footer_service: TechnicalFooterService | None = None,
        learning_recorder: LearningEventRecorder | None = None,
        trace_recorder: TraceRecorder | None = None,
        feedback_loop: Any = None,
        **specialist_deps: Any,
    ) -> InternalOrchestrator:
        _manifests = manifests or []
        if not specialist_deps.get("bitrix_bot"):
            specialist_deps["bitrix_bot"] = specialist_deps.get("bitrix_client")
        registry_deps = {
            **specialist_deps,
            "learning_recorder": learning_recorder,
            "trace_recorder": trace_recorder,
        }
        specialists = build_specialist_registry(
            _manifests,
            audience="employee",
            **{k: v for k, v in registry_deps.items() if v is not None},
        )
        diagnostic_specialists = build_specialist_registry(
            _manifests,
            audience="diagnostics",
            **{k: v for k, v in registry_deps.items() if v is not None},
        )
        _manifest = manifest or _dummy_manifest()

        call_tool = CallSpecialistTool(
            specialists,
            _manifests,
            scheduler=specialist_deps.get("scheduler"),
            store=orchestrator_store,
        )
        manage_tool = ManageSuspendedTool(store=orchestrator_store)
        schedule_tool = ScheduleTaskTool(scheduler=specialist_deps.get("scheduler"))

        orch = cls(
            _manifest,
            agent_tools=[call_tool, manage_tool, schedule_tool],
            llm=orchestrator_llm,
            store=orchestrator_store,
            scheduler=specialist_deps.get("scheduler"),
            retriever=orchestrator_retriever,
            channels=channels,
            footer_service=footer_service,
            learning_recorder=learning_recorder,
            trace_recorder=trace_recorder,
            feedback_loop=feedback_loop,
            diagnostic_specialists=diagnostic_specialists,
            settings=specialist_deps.get("settings"),
        )
        # Break circular dep: CallSpecialistTool needs orch to schedule specialist tasks
        call_tool.schedule_fn = orch._apply_scheduled_tasks_from_specialist
        return orch

    # ------------------------------------------------------------------
    # Specialist-initiated scheduling (called from CallSpecialistTool)
    # ------------------------------------------------------------------

    def _apply_scheduled_tasks_from_specialist(self, tasks: list[ScheduledTask]) -> None:
        if not tasks or self._scheduler is None:
            return
        _orch = self
        for sched in tasks:
            if sched.cancel:
                if hasattr(self._scheduler, "remove_job"):
                    self._scheduler.remove_job(sched.agent_id, sched.job_id)  # type: ignore[attr-defined]
            elif sched.task is not None:
                _task = sched.task

                async def _run(_t: AgentTask = _task, _o: InternalOrchestrator = _orch) -> None:
                    await _o.handle(_t)

                try:
                    if hasattr(self._scheduler, "schedule_callback"):
                        self._scheduler.schedule_callback(sched.agent_id, sched.job_id, sched.trigger, _run)  # type: ignore[attr-defined]
                except Exception:
                    logger.exception("Failed to schedule task job_id=%s agent=%s", sched.job_id, sched.agent_id)

    # ------------------------------------------------------------------
    # Post-handle side-effects (not part of agent loop)
    # ------------------------------------------------------------------

    async def _handle_error_report_command(self, task: AgentTask, report_request: dict[str, Any]) -> AgentResult:
        if not self._error_report_allowed(task):
            return AgentResult(
                status="needs_human",
                agent_id="internal_orchestrator",
                answer="Отчет Диагноста доступен только dev/admin пользователям.",
                actions_taken=[
                    ActionRecord(
                        name="diagnostic_error_report_access",
                        status="denied",
                        details={"source": task.source, "user_id": task.user.id if task.user else None},
                    )
                ],
                model_usage=[
                    ModelUsageRecord(
                        agent_id="internal_orchestrator",
                        provider="internal",
                        model="diagnostic_report_routing",
                    )
                ],
                confidence=1.0,
            )
        diagnostic = self._diagnostic_specialists.get("diagnostic_agent")
        if diagnostic is None:
            return AgentResult(
                status="failed",
                agent_id="internal_orchestrator",
                answer="Diagnostic Agent не настроен для внутреннего отчета.",
                actions_taken=[
                    ActionRecord(
                        name="orchestrator_diagnostic_report_command",
                        status="failed",
                        details={"reason": "diagnostic_agent_not_configured", **report_request},
                    )
                ],
                model_usage=[
                    ModelUsageRecord(
                        agent_id="internal_orchestrator",
                        provider="internal",
                        model="diagnostic_report_routing",
                        status="error",
                    )
                ],
                confidence=0.0,
            )
        diagnostic_task = task.model_copy(
            update={
                "context": {
                    **task.context,
                    "error_report_request": report_request,
                    "skip_feedback_prompt": True,
                }
            }
        )
        result = await diagnostic.handle(diagnostic_task)
        return AgentResult(
            status=result.status,
            agent_id="internal_orchestrator",
            answer=result.answer or "Diagnostic Agent не вернул отчет.",
            artifacts=[Artifact.model_validate(item.model_dump()) for item in result.artifacts],
            actions_taken=[
                ActionRecord(
                    name="orchestrator_diagnostic_report_command",
                    status="completed",
                    details={"tool_calls": [{"name": "call_diagnostic_agent"}], **report_request},
                ),
                *result.actions_taken,
            ],
            actions_requiring_approval=result.actions_requiring_approval,
            model_usage=[
                ModelUsageRecord(
                    agent_id="internal_orchestrator",
                    provider="internal",
                    model="diagnostic_report_routing",
                ),
                *result.model_usage,
            ],
            handoff_to=["diagnostic_agent"],
            confidence=result.confidence,
        )

    def _error_report_allowed(self, task: AgentTask) -> bool:
        if task.source != "bitrix24_chat":
            return True
        raw_user_id = task.user.id if task.user else None
        if not raw_user_id or not str(raw_user_id).isdigit() or self._settings is None:
            return False
        user_id = int(raw_user_id)
        allowed = set(getattr(self._settings, "resolved_diagnostic_report_admin_user_ids", []))
        allowed.update(getattr(self._settings, "resolved_supervisor_admin_user_ids", []))
        allowed.update(getattr(self._settings, "resolved_vehicle_usage_admin_notify_user_ids", []))
        manager_id = getattr(self._settings, "vehicle_usage_manager_user_id", None)
        if manager_id:
            allowed.add(int(manager_id))
        return user_id in allowed

    async def _send_to_channel(
        self,
        task: AgentTask,
        result: AgentResult,
        *,
        learning_record: dict[str, Any] | None = None,
    ) -> None:
        channel_id = task.context.get("channel_id", "")
        recipient_id = task.context.get("recipient_id", "")
        if not channel_id or not recipient_id:
            self._record_message_sent_trace(task, result, status="skipped", reason="no_channel")
            return
        channel = self._channels.get(channel_id)
        if channel is None:
            self._record_message_sent_trace(task, result, status="skipped", reason="unknown_channel")
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
        if (
            body
            and self._feedback_loop is not None
            and channel_id == "bitrix24"
            and not task.context.get("skip_feedback_prompt")
            and "diagnostic_agent" not in result.handoff_to
        ):
            body = self._feedback_loop.append_prompt(body)
            self._feedback_loop.remember_answer(task, result, learning_record)
        if body:
            try:
                await channel.send(recipient_id, body)
                self._record_message_sent_trace(task, result, status="sent", reason="")
            except Exception:
                logger.exception("Channel send failed for channel=%s recipient=%s", channel_id, recipient_id)
                self._record_message_sent_trace(task, result, status="error", reason="channel_send_failed")
        else:
            self._record_message_sent_trace(task, result, status="skipped", reason="empty_body")

    def _record_message_sent_trace(self, task: AgentTask, result: AgentResult, *, status: str, reason: str) -> None:
        if self._trace_recorder is None:
            return
        trace_id, span_id = self._trace_recorder.ensure_task_context(task)
        self._trace_recorder.record(
            event_name="message_sent_to_user",
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id_from_task(task),
            agent_id="internal_orchestrator",
            task_id=task.task_id,
            status=status,
            payload={
                "reason": reason,
                "answer_present": bool(result.answer),
                "channel_id": task.context.get("channel_id", ""),
            },
        )

    def _record_learning(
        self, task: AgentTask, result: AgentResult, *, elapsed_ms: dict[str, float] | None = None
    ) -> dict[str, Any] | None:
        if self._learning_recorder is None:
            return None
        try:
            return self._learning_recorder.record_agent_result(
                task,
                result,
                metadata={"dialog_key": task.context.get("dialog_key", "")},
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            logger.exception("Learning recording failed")
            return None

    # ------------------------------------------------------------------
    # Backward-compat: expose specialists dict for startup.py run() calls
    # ------------------------------------------------------------------

    @property
    def specialists(self) -> dict[str, Specialist]:
        call_tool = self._tool_registry.get("call_specialist")
        if isinstance(call_tool, CallSpecialistTool):
            return call_tool._specialists
        return {}


def _error_report_command(text: str) -> dict[str, Any] | None:
    normalized = str(text or "").strip().casefold().replace("ё", "е")
    if not normalized:
        return None
    report_markers = ("отчет", "отчёт", "report", "сводка", "покажи", "дай")
    diagnostic_markers = ("ошиб", "incident", "инцидент", "diagnostic", "диагност", "feedback")
    if not any(marker in normalized for marker in report_markers):
        return None
    if not any(marker in normalized for marker in diagnostic_markers):
        return None
    since_hours = 24
    if "недел" in normalized or "7д" in normalized or "7 д" in normalized:
        since_hours = 24 * 7
    elif "сегодня" in normalized:
        since_hours = 24
    elif "час" in normalized:
        since_hours = 1
    return {"since_hours": since_hours, "limit": 200, "max_groups": 5}


def _dummy_manifest() -> AgentManifest:
    return AgentManifest(
        id="internal_orchestrator",
        name="Переговорщик",
        kind="orchestrator",
        description="Старший AI-агент. Посредник между людьми и специалистами.",
    )
