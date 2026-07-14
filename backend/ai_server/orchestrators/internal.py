from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import ValidationError

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.ports import (
    AgentQueuePort,
    ChannelPort,
    OrchestratorStorePort,
    ResultPublisherPort,
    SchedulerPort,
)
from ai_server.models import AgentManifest, AgentResult, AgentTask, ScheduledTask
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
        manifest: AgentManifest,
        *,
        agent_tools: list | None = None,
        llm: OrchestratorLLM | None = None,
        store: OrchestratorStorePort | None = None,
        scheduler: SchedulerPort | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        channels: dict[str, ChannelPort] | None = None,
        footer_service: TechnicalFooterService | None = None,
        result_publisher: ResultPublisherPort | None = None,
        conversation_trace: Any = None,
    ) -> None:
        super().__init__(
            manifest,
            agent_tools=agent_tools,
            llm=llm or OrchestratorLLMService(),
            store=store,
            scheduler=scheduler,
            retriever=retriever,
        )
        self._channels: dict[str, ChannelPort] = channels or {}
        self._footer_svc = footer_service
        self._result_publisher = result_publisher
        self._conversation_trace = conversation_trace

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

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------

    async def handle(self, task: AgentTask) -> AgentResult:
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
        await self._send_to_channel(task, result)
        await self._publish_result(task, result)
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
        result_publisher: ResultPublisherPort | None = None,
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
            result_publisher=result_publisher,
            conversation_trace=specialist_deps.get("conversation_trace"),
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
                    removed = self._scheduler.remove_job(sched.agent_id, sched.job_id)  # type: ignore[attr-defined]
                    if not removed and hasattr(self._scheduler, "remove_jobs_by_prefix"):
                        self._scheduler.remove_jobs_by_prefix(sched.agent_id, sched.job_id)  # type: ignore[attr-defined]
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
                if self._conversation_trace is not None:
                    await self._conversation_trace.record_outbound(
                        task=task,
                        result=result,
                        recipient_id=recipient_id,
                        body=body,
                        status="error",
                        error="channel_send_failed",
                    )
            else:
                if self._conversation_trace is not None:
                    await self._conversation_trace.record_outbound(
                        task=task,
                        result=result,
                        recipient_id=recipient_id,
                        body=body,
                        status="sent",
                    )

    async def _publish_result(self, task: AgentTask, result: AgentResult) -> None:
        if self._result_publisher is None:
            return
        try:
            await self._result_publisher.publish(task, result)
        except Exception:
            logger.exception("Result publishing failed")

    # ------------------------------------------------------------------
    # Backward-compat: expose specialists dict for startup.py run() calls
    # ------------------------------------------------------------------

    @property
    def specialists(self) -> dict[str, Specialist]:
        call_tool = self._tool_registry.get("call_specialist")
        if isinstance(call_tool, CallSpecialistTool):
            return call_tool._specialists
        return {}


def _dummy_manifest() -> AgentManifest:
    return AgentManifest(
        id="internal_orchestrator",
        name="Переговорщик",
        kind="orchestrator",
        description="Старший AI-агент. Посредник между людьми и специалистами.",
    )
