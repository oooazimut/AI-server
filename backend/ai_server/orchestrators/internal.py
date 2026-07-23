from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime
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
from ai_server.integrations.redis.outbound_queue import outbound_delivery_key
from ai_server.models import AgentManifest, AgentResult, AgentTask, ScheduledTask
from ai_server.orchestrators.tools import CallSpecialistTool
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.specialists import Specialist
from ai_server.technical_footer import TechnicalFooterService, append_footer
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)

_DRAFT_CONFIRMATION_LINE = re.compile(
    r"\s*Для подтверждения отправьте фразу:\s*[«\"][^»\"]+[»\"]\.?",
    re.IGNORECASE,
)
_ACTIVE_DRAFT_MANAGEMENT_LINE = re.compile(
    r"\s*Для управления активным черновиком:\s*подтвердить или отменить\.?",
    re.IGNORECASE,
)
_NEXT_PAGE_HINT = re.compile(r"(?:можно\s+запросить|попросите\s+показать)\s+следующ", re.IGNORECASE)


def _append_conversation_reference(message: str, task: AgentTask) -> str:
    """Put the human-visible branch hint at the absolute end of every reply."""
    number = task.context.get("conversation_number")
    if number in (None, ""):
        return message
    visible = str(number)

    rendered, management_count = _ACTIVE_DRAFT_MANAGEMENT_LINE.subn("", message)
    rendered, confirmation_count = _DRAFT_CONFIRMATION_LINE.subn("", rendered)
    rendered = rendered.rstrip()
    if management_count:
        reference = (
            f"Диалог №{visible}. Для подтверждения: «{visible} подтвердить». "
            f"Для отмены: «{visible} отменить»"
        )
    elif confirmation_count:
        reference = f"Диалог №{visible}. Для подтверждения: «{visible} подтвердить»"
    elif _NEXT_PAGE_HINT.search(rendered):
        reference = f"Диалог №{visible}. Для продолжения: «{visible} следующая»"
    else:
        reference = f"Диалог №{visible}."
    return f"{rendered}\n\n{reference}" if rendered else reference


class OrchestratorTransportRuntime(BaseSpecialist):
    """Shared transport/lifecycle base for the sole plan-authoritative runtime."""

    action_prefix = "orchestrator"
    max_steps = 4

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        agent_tools: list | None = None,
        llm: Any | None = None,
        store: OrchestratorStorePort | None = None,
        scheduler: SchedulerPort | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        channels: dict[str, ChannelPort] | None = None,
        footer_service: TechnicalFooterService | None = None,
        result_publisher: ResultPublisherPort | None = None,
        conversation_trace: Any = None,
        dialog_guard: Any = None,
        outbound_queue: Any = None,
    ) -> None:
        super().__init__(
            manifest,
            agent_tools=agent_tools,
            llm=llm,
            store=store,
            scheduler=scheduler,
            retriever=retriever,
            conversation_trace=conversation_trace,
        )
        self._channels: dict[str, ChannelPort] = channels or {}
        self._footer_svc = footer_service
        self._result_publisher = result_publisher
        self._conversation_trace = conversation_trace
        self._dialog_guard = dialog_guard
        self._outbound_queue = outbound_queue

    # ------------------------------------------------------------------
    # BaseSpecialist hooks
    # ------------------------------------------------------------------

    def _logs(self) -> list[str]:
        return []

    def _llm_failure_result(self, message: str) -> Any:
        raise RuntimeError("PLAN_AUTHORITATIVE_HANDLER_REQUIRED")

    # ------------------------------------------------------------------
    # Lifecycle overrides
    # ------------------------------------------------------------------

    async def handle(self, task: AgentTask) -> AgentResult:
        raise RuntimeError("PLAN_AUTHORITATIVE_HANDLER_REQUIRED")

    async def run(
        self,
        queue: AgentQueuePort,
        *,
        worker_name: str = "",
        task_timeout_seconds: float | None = None,
    ) -> None:
        """Queue consumer loop.

        Handles two message types:
        - "task" / "bitrix_chat" — new request → handle()
        - "result" — proactive result from a specialist → _send_to_channel()
        """
        _poll_interval = 0.1
        while True:
            message, partition_key = await self._claim_queue_message(queue, "orchestrator")
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
                    await self._await_message_task(self.handle(task), timeout_seconds=task_timeout_seconds)
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
                            task_id=str(message.get("correlation_id") or msg_id),
                            request="",
                            context={
                                "channel_id": routing["channel_id"],
                                "recipient_id": routing["recipient_id"],
                                "dialog_key": routing.get("dialog_key") or "",
                            },
                        )
                        await self._await_message_task(
                            self._send_to_channel(stub_task, result),
                            timeout_seconds=task_timeout_seconds,
                        )
                await queue.ack(msg_id)
            except TimeoutError:
                logger.exception("Orchestrator worker %s timed out processing message %s", worker_name, msg_id)
                await queue.nack(msg_id, error=f"TimeoutError: task exceeded {task_timeout_seconds}s")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Orchestrator failed processing message %s", msg_id)
                await queue.nack(msg_id, error=f"{type(exc).__name__}: {exc}")
            finally:
                await self._release_queue_partition(partition_key)

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

                async def _run(_t: AgentTask = _task, _o: OrchestratorTransportRuntime = _orch) -> None:
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
            footer_started_at = _trace_now_iso()
            footer_t0 = time.monotonic()
            try:
                footer = await self._footer_svc.build_for_agent_result(
                    result, user_id=user_id, channel=f"{channel_id}_chat"
                )
            except Exception:
                logger.exception("Footer build failed")
                await self._record_timing(
                    task,
                    stage="footer_build",
                    started_at=footer_started_at,
                    elapsed_ms=(time.monotonic() - footer_t0) * 1000,
                    status="error",
                    details={"user_id": user_id},
                )
            else:
                await self._record_timing(
                    task,
                    stage="footer_build",
                    started_at=footer_started_at,
                    elapsed_ms=(time.monotonic() - footer_t0) * 1000,
                    status="completed",
                    details={"user_id": user_id, "footer_chars": len(footer)},
                )
        answer = result.answer or ""
        body = append_footer(answer, footer) if answer else ""
        if body:
            body = _append_conversation_reference(body, task)
        if body:
            if self._dialog_guard is not None and await self._dialog_guard.task_is_stale(task):
                logger.info(
                    "Suppressing stale channel send task_id=%s dialog_key=%s",
                    task.task_id,
                    task.context.get("dialog_key"),
                )
                if self._conversation_trace is not None:
                    await self._conversation_trace.record_outbound(
                        task=task,
                        result=result,
                        recipient_id=recipient_id,
                        body=body,
                        status="suppressed",
                        error="dialog_cancelled",
                    )
                return
            if self._outbound_queue is not None:
                send_started_at = _trace_now_iso()
                send_t0 = time.monotonic()
                delivery_key = outbound_delivery_key(
                    channel_id=str(channel_id),
                    recipient_id=str(recipient_id),
                    task_id=str(task.task_id),
                    body=body,
                )
                try:
                    delivery_id, created = await self._outbound_queue.enqueue(
                        delivery_key=delivery_key,
                        channel_id=str(channel_id),
                        recipient_id=str(recipient_id),
                        body=body,
                        task=task.model_dump(),
                        result=result.model_dump(),
                    )
                except Exception:
                    await self._record_timing(
                        task,
                        stage="channel_outbox_enqueue",
                        started_at=send_started_at,
                        elapsed_ms=(time.monotonic() - send_t0) * 1000,
                        status="error",
                        details={"channel_id": channel_id, "recipient_id": recipient_id},
                    )
                    raise
                await self._record_timing(
                    task,
                    stage="channel_outbox_enqueue",
                    started_at=send_started_at,
                    elapsed_ms=(time.monotonic() - send_t0) * 1000,
                    status="queued" if created else "deduplicated",
                    details={
                        "channel_id": channel_id,
                        "recipient_id": recipient_id,
                        "delivery_id": delivery_id,
                        "body_chars": len(body),
                    },
                )
                if self._conversation_trace is not None:
                    await self._conversation_trace.record_outbound(
                        task=task,
                        result=result,
                        recipient_id=str(recipient_id),
                        body=body,
                        status="queued" if created else "deduplicated",
                        delivery_id=delivery_id,
                    )
                return
            send_started_at = _trace_now_iso()
            send_t0 = time.monotonic()
            try:
                await channel.send(recipient_id, body)
            except Exception:
                logger.exception("Channel send failed for channel=%s recipient=%s", channel_id, recipient_id)
                await self._record_timing(
                    task,
                    stage="channel_send",
                    started_at=send_started_at,
                    elapsed_ms=(time.monotonic() - send_t0) * 1000,
                    status="error",
                    details={"channel_id": channel_id, "recipient_id": recipient_id, "body_chars": len(body)},
                )
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
                await self._record_timing(
                    task,
                    stage="channel_send",
                    started_at=send_started_at,
                    elapsed_ms=(time.monotonic() - send_t0) * 1000,
                    status="sent",
                    details={"channel_id": channel_id, "recipient_id": recipient_id, "body_chars": len(body)},
                )
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
        started_at = _trace_now_iso()
        t0 = time.monotonic()
        try:
            await self._result_publisher.publish(task, result)
        except Exception:
            logger.exception("Result publishing failed")
            await self._record_timing(
                task,
                stage="result_publish",
                started_at=started_at,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                status="error",
            )
        else:
            await self._record_timing(
                task,
                stage="result_publish",
                started_at=started_at,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                status="completed",
            )

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


def _trace_now_iso() -> str:
    return datetime.now(MOSCOW_TZ).isoformat()
