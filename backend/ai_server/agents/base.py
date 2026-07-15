from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from ai_server.agent_queue_utils import agent_queue_partition_key
from ai_server.agents.ports import AgentQueuePort, AgentStorePort, ResultPublisherPort, SchedulerPort
from ai_server.agents.tool import AgentTool
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import (
    ActionRecord,
    AgentManifest,
    AgentResult,
    AgentTask,
    ModelUsageRecord,
    ToolResult,
    ToolStatus,
)
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.utils import MOSCOW_TZ, optional_int, unique

logger = logging.getLogger(__name__)

_NOT_CONFIGURED_ANSWER = "Специалист временно недоступен: LLM не сконфигурирован."


class BaseSpecialist:
    """
    Shared control-flow template for specialist agents (decide-loop -> execute tools -> compose).

    Subclasses set ``max_steps``/``action_prefix`` and implement the hooks below; everything else
    (context assembly, the decide/execute loop, error handling, AgentResult assembly) lives here so
    a fix applied once (e.g. dialog_history forwarding) covers every specialist.
    """

    max_steps: int = 5
    action_prefix: str = ""
    _queue_poll_interval: float = 0.1

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        agent_tools: list[AgentTool] | None = None,
        llm: Any = None,
        scheduler: SchedulerPort | None = None,
        store: AgentStorePort | None = None,
        result_publisher: ResultPublisherPort | None = None,
        conversation_trace: Any = None,
    ) -> None:
        self.manifest = manifest
        self.knowledge_base = knowledge_base
        self.skill_store = skill_store
        self.retriever = retriever
        self._tool_registry: dict[str, AgentTool] = {t.name: t for t in (agent_tools or [])}
        self.llm = llm
        self._scheduler = scheduler
        self.store = store
        self._result_publisher = result_publisher
        self._conversation_trace = conversation_trace
        self._active_queue_partitions: set[str] = set()
        self._queue_partition_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Hooks subclasses implement/override
    # ------------------------------------------------------------------

    def tool_definitions(self) -> list[dict]:
        return [t.definition().model_dump() for t in self._tool_registry.values()]

    def _logs(self) -> list[str]:
        raise NotImplementedError

    def _llm_failure_result(self, message: str) -> Any:
        raise NotImplementedError

    async def _load_extra_context(self, task: AgentTask) -> tuple[AgentTask, dict[str, Any]]:
        """Override to merge specialist-specific context into the task before retrieval/decide.

        Returns the (possibly updated) task plus an extra-details dict merged into the
        ``load_{action_prefix}_specialist_context`` action record (e.g. bitrix24's permission_context).
        """
        return task, {}

    async def _execute_tool_call(
        self,
        tool_call: Any,
        task: AgentTask,
    ) -> tuple[ToolResult | None, ActionRecord | None, list[ActionRecord]]:
        if tool_call.name == "none":
            return None, None, []
        tool = self._tool_registry.get(tool_call.name)
        if tool is None:
            result = ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool=tool_call.name,
                error=f"unknown tool: {tool_call.name}",
            )
            return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump()), []
        user_id = optional_int(task.user.id) if task.user.id else None
        dialog_key = str(task.context.get("dialog_key") or "") or None
        dialog_id = str(task.context.get("dialog_id") or (task.user.raw or {}).get("dialog_id") or "") or None
        execute_with_task = getattr(tool, "execute_with_task", None)
        if execute_with_task is not None:
            result = await execute_with_task(tool_call.args, task=task)
        else:
            result = await tool.execute(tool_call.args, user_id=user_id, dialog_key=dialog_key, dialog_id=dialog_id)
        action = ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump())
        return result, action, []

    def _terminal_response_metadata(
        self,
        *,
        tool_call: Any,
        result: ToolResult | None,
        action: ActionRecord | None,
        approvals: list[ActionRecord],
        task: AgentTask,
    ) -> dict[str, Any] | None:
        """Return terminal-response metadata when the loop can skip the next decide step."""
        return None

    # ------------------------------------------------------------------
    # Lifecycle hooks (subclasses override as needed)
    # ------------------------------------------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def _record_timing(
        self,
        task: AgentTask,
        *,
        stage: str,
        started_at: str,
        elapsed_ms: float,
        status: str = "",
        step: int | None = None,
        tool: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        if self._conversation_trace is None:
            return
        try:
            await self._conversation_trace.record_timing(
                task=task,
                component=self.manifest.id,
                stage=stage,
                started_at=started_at,
                elapsed_ms=elapsed_ms,
                status=status,
                step=step,
                tool=tool,
                details=details,
            )
        except Exception:
            logger.debug("ConversationTrace timing failed", exc_info=True)

    # ------------------------------------------------------------------
    # Scheduler helpers (scheduler may be None — all methods are safe)
    # ------------------------------------------------------------------

    def schedule_job(self, job_id: str, func: Any, trigger: Any, **kwargs: Any) -> Any:
        if self._scheduler is None:
            return None
        return self._scheduler.add_job(self.manifest.id, job_id, func, trigger, **kwargs)

    def schedule_job_at(self, job_id: str, func: Any, run_date: datetime, **kwargs: Any) -> Any:
        if self._scheduler is None:
            return None
        return self._scheduler.add_job_at(self.manifest.id, job_id, func, run_date, **kwargs)

    def schedule_job_cron(self, job_id: str, func: Any, hour: int, minute: int, **kwargs: Any) -> Any:
        if self._scheduler is None:
            return None
        return self._scheduler.add_job_cron(self.manifest.id, job_id, func, hour, minute, **kwargs)

    def cancel_jobs_by_prefix(self, prefix: str) -> int:
        if self._scheduler is None:
            return 0
        return self._scheduler.remove_jobs_by_prefix(self.manifest.id, prefix)

    def list_jobs(self) -> list[dict[str, Any]]:
        if self._scheduler is None:
            return []
        return self._scheduler.list_jobs(self.manifest.id)

    # ------------------------------------------------------------------
    # Unified control flow
    # ------------------------------------------------------------------

    async def handle(self, task: AgentTask) -> AgentResult:
        if self.llm is None:
            failure = self._llm_failure_result(_NOT_CONFIGURED_ANSWER)
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer=failure.answer,
                actions_taken=[],
                actions_requiring_approval=[],
                model_usage=[failure.model_usage],
                confidence=0.0,
                logs=self._logs(),
            )

        context_started_at = _trace_now_iso()
        context_t0 = time.monotonic()
        # Load per-specialist dialog history from agent's own PG schema (if configured),
        # otherwise fall back to the shared history passed via task.context.
        dialog_key: str = task.context.get("dialog_key") or ""
        if self.store is not None and dialog_key:
            dialog_history: list[dict] = await self.store.load_turns(dialog_key, limit=20)
        else:
            dialog_history = list(task.context.get("dialog_history") or [])

        available_skills = (
            self.skill_store.list_skills_with_content(self.manifest) if self.skill_store is not None else []
        )
        task, extra_context_details = await self._load_extra_context(task)
        retrieval_hits = (
            self.retriever.search(self.manifest, task.request, limit=5) if self.retriever is not None else []
        )
        actions_taken = [
            ActionRecord(
                name=f"load_{self.action_prefix}_specialist_context",
                status="completed",
                details={
                    "available_skills": [
                        {"id": skill.id, "title": skill.title, "preview": skill.preview} for skill in available_skills
                    ],
                    "retrieval_topics": unique([hit.chunk.topic for hit in retrieval_hits]),
                    "retrieval_hits": [
                        {
                            "topic": hit.chunk.topic,
                            "section": hit.chunk.section,
                            "score": hit.score,
                            "keyword_score": hit.keyword_score,
                            "vector_score": hit.vector_score,
                            "embedding_provider": hit.embedding_provider,
                        }
                        for hit in retrieval_hits
                    ],
                    **extra_context_details,
                },
            )
        ]
        await self._record_timing(
            task,
            stage="context_load",
            started_at=context_started_at,
            elapsed_ms=(time.monotonic() - context_t0) * 1000,
            status="completed",
            details={
                "dialog_history_count": len(dialog_history),
                "available_skills_count": len(available_skills),
                "retrieval_hits_count": len(retrieval_hits),
            },
        )

        tool_results: list[ToolResult] = []
        approval_actions: list[ActionRecord] = []
        decision_results: list[Any] = []
        terminal_response_metadata: dict[str, Any] = {}
        decision = None

        for step in range(1, self.max_steps + 1):
            decision_started_at = _trace_now_iso()
            decision_t0 = time.monotonic()
            try:
                decision_result = await self.llm.decide(
                    manifest=self.manifest,
                    task=task,
                    retrieval_hits=retrieval_hits,
                    tool_definitions=self.tool_definitions(),
                    tool_results=list(tool_results),
                    dialog_history=dialog_history or None,
                    available_skills=available_skills,
                )
            except Exception as exc:
                await self._record_timing(
                    task,
                    stage="llm_decide",
                    started_at=decision_started_at,
                    elapsed_ms=(time.monotonic() - decision_t0) * 1000,
                    status="error",
                    step=step,
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
                failure = self._llm_failure_result(f"{type(exc).__name__}: {exc}")
                return AgentResult(
                    status="failed",
                    agent_id=self.manifest.id,
                    answer=failure.answer,
                    actions_taken=[
                        *actions_taken,
                        ActionRecord(
                            name=f"{self.action_prefix}_llm_decision",
                            status="error",
                            details={"step": step, "error": f"{type(exc).__name__}: {exc}"},
                        ),
                    ],
                    actions_requiring_approval=approval_actions,
                    model_usage=[*[item.model_usage for item in decision_results], failure.model_usage],
                    confidence=0.0,
                    logs=self._logs(),
                )

            decision_results.append(decision_result)
            decision = decision_result.decision
            await self._record_timing(
                task,
                stage="llm_decide",
                started_at=decision_started_at,
                elapsed_ms=(time.monotonic() - decision_t0) * 1000,
                status=decision.status,
                step=step,
                details={
                    "tool_calls": [{"name": call.name, "summary": call.summary} for call in decision.tool_calls],
                    "confidence": decision.confidence,
                },
            )
            actions_taken.append(
                ActionRecord(
                    name=f"{self.action_prefix}_llm_decision",
                    status=decision.status,
                    details={
                        "step": step,
                        "tool_calls": [
                            {"name": call.name, "args": call.args, "summary": call.summary}
                            for call in decision.tool_calls
                        ],
                        "confidence": decision.confidence,
                    },
                )
            )

            executable_calls = [call for call in decision.tool_calls if call.name != "none"]
            if not executable_calls:
                break
            stop_after_step = False
            for tool_call in executable_calls:
                tool_started_at = _trace_now_iso()
                tool_t0 = time.monotonic()
                try:
                    result, action, approvals = await self._execute_tool_call(tool_call, task)
                except Exception as exc:
                    await self._record_timing(
                        task,
                        stage="tool_execute",
                        started_at=tool_started_at,
                        elapsed_ms=(time.monotonic() - tool_t0) * 1000,
                        status="error",
                        step=step,
                        tool=tool_call.name,
                        details={"tool": tool_call.name, "error": f"{type(exc).__name__}: {exc}"},
                    )
                    raise
                await self._record_timing(
                    task,
                    stage="tool_execute",
                    started_at=tool_started_at,
                    elapsed_ms=(time.monotonic() - tool_t0) * 1000,
                    status=str(result.status) if result is not None else "completed",
                    step=step,
                    tool=tool_call.name,
                    details={
                        "tool": tool_call.name,
                        "tool_result_tool": result.tool if result is not None else "",
                        "tool_result_status": str(result.status) if result is not None else "",
                        "approvals_count": len(approvals),
                        "has_result": result is not None,
                    },
                )
                if result is not None:
                    tool_results.append(result)
                if action is not None:
                    actions_taken.append(action)
                approval_actions.extend(approvals)
                if len(executable_calls) == 1:
                    terminal_metadata = self._terminal_response_metadata(
                        tool_call=tool_call,
                        result=result,
                        action=action,
                        approvals=approvals,
                        task=task,
                    )
                    if terminal_metadata:
                        terminal_response_metadata = {
                            "terminal": True,
                            "answer_is_final": True,
                            "safe_to_send": True,
                            "fast_return": True,
                            **terminal_metadata,
                        }
                        actions_taken.append(
                            ActionRecord(
                                name=f"{self.action_prefix}_fast_return",
                                status="completed",
                                details=terminal_response_metadata,
                            )
                        )
                        await self._record_timing(
                            task,
                            stage="fast_return",
                            started_at=_trace_now_iso(),
                            elapsed_ms=0.0,
                            status="completed",
                            step=step,
                            tool=tool_call.name,
                            details=terminal_response_metadata,
                        )
                        stop_after_step = True
            if step == self.max_steps and self.max_steps > 1:
                actions_taken.append(
                    ActionRecord(
                        name=f"{self.action_prefix}_tool_loop_guardrail",
                        status="stopped",
                        details={"max_steps": self.max_steps},
                    )
                )
            if stop_after_step:
                break

        if decision is None:
            failure = self._llm_failure_result(f"empty {self.action_prefix} LLM decision loop")
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer=failure.answer,
                actions_taken=actions_taken,
                actions_requiring_approval=approval_actions,
                model_usage=[failure.model_usage],
                confidence=0.0,
                logs=self._logs(),
            )

        compose_started_at = _trace_now_iso()
        compose_t0 = time.monotonic()
        try:
            final_result = await self.llm.compose(
                manifest=self.manifest,
                task=task,
                decision=decision,
                tool_results=tool_results,
                approval_actions=[action.model_dump() for action in approval_actions],
            )
        except Exception as exc:
            await self._record_timing(
                task,
                stage="llm_compose",
                started_at=compose_started_at,
                elapsed_ms=(time.monotonic() - compose_t0) * 1000,
                status="error",
                details={"error": f"{type(exc).__name__}: {exc}"},
            )
            failure = self._llm_failure_result(f"{type(exc).__name__}: {exc}")
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer=failure.answer,
                actions_taken=[
                    *actions_taken,
                    ActionRecord(
                        name=f"{self.action_prefix}_llm_final_answer",
                        status="error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    ),
                ],
                actions_requiring_approval=approval_actions,
                model_usage=[*[item.model_usage for item in decision_results], failure.model_usage],
                confidence=0.0,
                logs=self._logs(),
            )

        await self._record_timing(
            task,
            stage="llm_compose",
            started_at=compose_started_at,
            elapsed_ms=(time.monotonic() - compose_t0) * 1000,
            status=final_result.status,
            details={
                "tool_results_count": len(tool_results),
                "approval_actions_count": len(approval_actions),
                "fast_return": bool(terminal_response_metadata.get("fast_return")),
                "fast_return_reason": str(terminal_response_metadata.get("fast_return_reason") or ""),
            },
        )
        actions_taken.append(
            ActionRecord(
                name=f"{self.action_prefix}_llm_final_answer",
                status=final_result.status,
                details={},
            )
        )
        terminal_answer_ready = bool(
            terminal_response_metadata.get("terminal")
            and terminal_response_metadata.get("answer_is_final")
            and terminal_response_metadata.get("safe_to_send")
        )
        # Decide status is authoritative for needs_clarification/needs_human:
        # compose only formats the answer text, not the conversational state.
        # A terminal fast-return is the exception: the tool already produced a final,
        # safe answer, so keep diagnostics aligned with what was actually sent.
        if terminal_answer_ready:
            effective_status = str(terminal_response_metadata.get("terminal_status") or "completed")
        elif (
            decision is not None
            and decision.status in ("needs_clarification", "needs_human")
            and final_result.status == "completed"
        ):
            effective_status = decision.status
        else:
            effective_status = final_result.status
        status = "needs_human" if approval_actions else effective_status

        if self.store is not None and dialog_key and final_result.answer:
            store_started_at = _trace_now_iso()
            store_t0 = time.monotonic()
            await self.store.append_turn(dialog_key, task.request, final_result.answer)
            await self._record_timing(
                task,
                stage="store_append_turn",
                started_at=store_started_at,
                elapsed_ms=(time.monotonic() - store_t0) * 1000,
                status="completed",
            )

        tool_model_usage = self._model_usage_from_tool_results(tool_results)
        return AgentResult(
            status=status,
            agent_id=self.manifest.id,
            answer=final_result.answer,
            actions_taken=actions_taken,
            actions_requiring_approval=approval_actions,
            model_usage=[*[item.model_usage for item in decision_results], *tool_model_usage, final_result.model_usage],
            confidence=decision.confidence,
            logs=self._logs(),
            metadata=terminal_response_metadata,
        )

    def _model_usage_from_tool_results(self, tool_results: list[ToolResult]) -> list[ModelUsageRecord]:
        usages: list[ModelUsageRecord] = []
        for result in tool_results:
            data = result.data if isinstance(result.data, dict) else {}
            raw_items = data.get("model_usage") or []
            if not isinstance(raw_items, list):
                continue
            for raw in raw_items:
                if isinstance(raw, ModelUsageRecord):
                    usages.append(raw)
                    continue
                if not isinstance(raw, dict):
                    continue
                try:
                    usages.append(ModelUsageRecord.model_validate(raw))
                except Exception:
                    continue
        return usages

    async def _claim_queue_message(
        self,
        queue: AgentQueuePort,
        agent_id: str,
    ) -> tuple[dict[str, Any] | None, str]:
        async with self._queue_partition_lock:
            message = await queue.claim_next(agent_id, blocked_partition_keys=self._active_queue_partitions)
            if message is None:
                return None, ""
            partition_key = str(message.get("_partition_key") or agent_queue_partition_key(message))
            if partition_key:
                self._active_queue_partitions.add(partition_key)
            return message, partition_key

    async def _release_queue_partition(self, partition_key: str) -> None:
        if not partition_key:
            return
        async with self._queue_partition_lock:
            self._active_queue_partitions.discard(partition_key)

    async def _await_message_task(self, coro: Any, *, timeout_seconds: float | None) -> Any:
        if timeout_seconds is None or timeout_seconds <= 0:
            return await coro
        return await asyncio.wait_for(coro, timeout=timeout_seconds)

    async def run(
        self,
        queue: AgentQueuePort,
        *,
        worker_name: str = "",
        task_timeout_seconds: float | None = None,
    ) -> None:
        """Queue consumer loop: claim → handle → publish result → ack/nack.

        Specialists never call queue.publish() directly inside handle().
        run() is the only place that touches the queue; handle() stays pure business logic.
        """
        agent_id = self.manifest.id
        while True:
            message, partition_key = await self._claim_queue_message(queue, agent_id)
            if message is None:
                await asyncio.sleep(self._queue_poll_interval)
                continue
            msg_id = str(message.get("id") or "")
            try:
                try:
                    task = AgentTask.model_validate(message["payload"])
                except (KeyError, ValidationError) as exc:
                    logger.warning("Agent %s: invalid message %s: %s", agent_id, msg_id, exc)
                    await queue.nack(msg_id, error=f"invalid message: {exc}")
                    continue
                result = await self._await_message_task(self.handle(task), timeout_seconds=task_timeout_seconds)
                if self._result_publisher is not None:
                    try:
                        await self._result_publisher.publish(task, result)
                    except Exception:
                        logger.exception("Agent %s: result_publisher failed", agent_id)
                reply_to = message.get("reply_to") or ""
                if reply_to:
                    await queue.publish(
                        {
                            "to": reply_to,
                            "from": agent_id,
                            "type": "result",
                            "correlation_id": message.get("correlation_id") or "",
                            "payload": result.model_dump(),
                            "routing": {
                                "channel_id": task.context.get("channel_id") or "",
                                "recipient_id": task.context.get("recipient_id") or "",
                                "dialog_key": task.context.get("dialog_key") or "",
                            },
                        }
                    )
                await queue.ack(msg_id)
            except TimeoutError:
                logger.exception("Agent %s worker %s timed out processing message %s", agent_id, worker_name, msg_id)
                await queue.nack(msg_id, error=f"TimeoutError: task exceeded {task_timeout_seconds}s")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Agent %s failed processing message %s", agent_id, msg_id)
                await queue.nack(msg_id, error=f"{type(exc).__name__}: {exc}")
            finally:
                await self._release_queue_partition(partition_key)


def _trace_now_iso() -> str:
    return datetime.now(MOSCOW_TZ).isoformat()
