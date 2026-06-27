from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from ai_server.agents.ports import AgentDialogStorePort, AgentQueuePort, SchedulerPort
from ai_server.agents.tool import AgentTool
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ToolResult, ToolStatus
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tracing import TraceRecorder, parent_span_id_from_task, span_id_from_task
from ai_server.utils import optional_int, unique

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
        store: AgentDialogStorePort | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self.manifest = manifest
        self.knowledge_base = knowledge_base
        self.skill_store = skill_store
        self.retriever = retriever
        self._tool_registry: dict[str, AgentTool] = {t.name: t for t in (agent_tools or [])}
        self.llm = llm
        self._scheduler = scheduler
        self.store = store
        self._trace_recorder = trace_recorder

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
        result = await tool.execute(tool_call.args, user_id=user_id, dialog_key=dialog_key, dialog_id=dialog_id)
        action = ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump())
        return result, action, []

    # ------------------------------------------------------------------
    # Lifecycle hooks (subclasses override as needed)
    # ------------------------------------------------------------------

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

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
        if self._trace_recorder is not None:
            trace_id, span_id = self._trace_recorder.ensure_task_context(task)
            self._trace_recorder.record(
                event_name="specialist_called",
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id_from_task(task),
                agent_id=self.manifest.id,
                task_id=task.task_id,
                status="received",
                payload={"request": task.request},
            )

        if self.llm is None:
            failure = self._llm_failure_result(_NOT_CONFIGURED_ANSWER)
            self._record_trace_event(
                task,
                event_name="specialist_final_answer",
                status="failed",
                payload={"reason": "llm_not_configured"},
            )
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

        # Load per-specialist dialog history from agent's own PG schema (if configured),
        # otherwise fall back to the shared history passed via task.context.
        dialog_key: str = task.context.get("dialog_key") or ""
        if self.store is not None and dialog_key:
            dialog_history: list[dict] = await self.store.load_turns(dialog_key, limit=20)
        else:
            dialog_history = list(task.context.get("dialog_history") or [])

        available_skills = self.skill_store.list_skills(self.manifest) if self.skill_store is not None else []
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
        self._record_trace_event(
            task,
            event_name="specialist_rules_retrieved",
            status="completed",
            payload={
                "available_skills": [
                    {"id": skill.id, "title": skill.title, "preview": skill.preview} for skill in available_skills
                ],
                "retrieval_topics": unique([hit.chunk.topic for hit in retrieval_hits]),
                "retrieval_hits": [
                    {"topic": hit.chunk.topic, "section": hit.chunk.section, "score": hit.score}
                    for hit in retrieval_hits
                ],
                **extra_context_details,
            },
        )

        tool_results: list[ToolResult] = []
        approval_actions: list[ActionRecord] = []
        decision_results: list[Any] = []
        decision = None

        for step in range(1, self.max_steps + 1):
            try:
                decision_result = await self.llm.decide(
                    manifest=self.manifest,
                    task=task,
                    retrieval_hits=retrieval_hits,
                    tool_definitions=self.tool_definitions(),
                    tool_results=list(tool_results),
                    dialog_history=dialog_history or None,
                )
            except Exception as exc:
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
                        "loaded_rules": decision_result.raw.get("loaded_rules", []),
                        "loaded_skills": decision_result.raw.get("loaded_skills", []),
                    },
                )
            )
            self._record_trace_event(
                task,
                event_name="specialist_llm_decision",
                status=decision.status,
                payload={
                    "step": step,
                    "tool_calls": [
                        {"name": call.name, "args": call.args, "summary": call.summary}
                        for call in decision.tool_calls
                    ],
                    "confidence": decision.confidence,
                    "loaded_rules": decision_result.raw.get("loaded_rules", []),
                    "loaded_skills": decision_result.raw.get("loaded_skills", []),
                },
            )

            executable_calls = [call for call in decision.tool_calls if call.name != "none"]
            if not executable_calls:
                break
            for tool_call in executable_calls:
                tool_span_id = ""
                if self._trace_recorder is not None:
                    trace_id, tool_span_id, _ = self._trace_recorder.child_context(task)
                    self._trace_recorder.record(
                        event_name="tool_called",
                        trace_id=trace_id,
                        span_id=tool_span_id,
                        parent_span_id=span_id_from_task(task),
                        agent_id=self.manifest.id,
                        task_id=task.task_id,
                        status="started",
                        payload={"name": tool_call.name, "args": tool_call.args, "summary": tool_call.summary},
                    )
                result, action, approvals = await self._execute_tool_call(tool_call, task)
                if self._trace_recorder is not None:
                    trace_id, _ = self._trace_recorder.ensure_task_context(task)
                    self._trace_recorder.record(
                        event_name="tool_result",
                        trace_id=trace_id,
                        span_id=tool_span_id,
                        parent_span_id=span_id_from_task(task),
                        agent_id=self.manifest.id,
                        task_id=task.task_id,
                        status=str(result.status) if result is not None else "skipped",
                        payload={
                            "name": tool_call.name,
                            "result": result.model_dump() if result is not None else {},
                            "approvals": [approval.model_dump() for approval in approvals],
                        },
                    )
                if result is not None:
                    tool_results.append(result)
                if action is not None:
                    actions_taken.append(action)
                approval_actions.extend(approvals)
            if step == self.max_steps and self.max_steps > 1:
                actions_taken.append(
                    ActionRecord(
                        name=f"{self.action_prefix}_tool_loop_guardrail",
                        status="stopped",
                        details={"max_steps": self.max_steps},
                    )
                )

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

        try:
            final_result = await self.llm.compose(
                manifest=self.manifest,
                task=task,
                decision=decision,
                tool_results=tool_results,
                approval_actions=[action.model_dump() for action in approval_actions],
            )
        except Exception as exc:
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

        actions_taken.append(
            ActionRecord(
                name=f"{self.action_prefix}_llm_final_answer",
                status=final_result.status,
                details={},
            )
        )
        self._record_trace_event(
            task,
            event_name="specialist_final_answer",
            status=final_result.status,
            payload={"answer_present": bool(final_result.answer), "approval_actions": len(approval_actions)},
        )
        status = "needs_human" if approval_actions else final_result.status

        if self.store is not None and dialog_key and final_result.answer:
            await self.store.append_turn(dialog_key, task.request, final_result.answer)

        return AgentResult(
            status=status,
            agent_id=self.manifest.id,
            answer=final_result.answer,
            actions_taken=actions_taken,
            actions_requiring_approval=approval_actions,
            model_usage=[*[item.model_usage for item in decision_results], final_result.model_usage],
            confidence=decision.confidence,
            logs=self._logs(),
        )

    def _record_trace_event(
        self,
        task: AgentTask,
        *,
        event_name: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._trace_recorder is None:
            return
        trace_id, span_id = self._trace_recorder.ensure_task_context(task)
        self._trace_recorder.record(
            event_name=event_name,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id_from_task(task),
            agent_id=self.manifest.id,
            task_id=task.task_id,
            status=status,
            payload=payload or {},
        )

    async def run(self, queue: AgentQueuePort) -> None:
        """Queue consumer loop: claim → handle → publish result → ack/nack.

        Specialists never call queue.publish() directly inside handle().
        run() is the only place that touches the queue; handle() stays pure business logic.
        """
        agent_id = self.manifest.id
        while True:
            message = await queue.claim_next(agent_id)
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
                result = await self.handle(task)
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
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Agent %s failed processing message %s", agent_id, msg_id)
                await queue.nack(msg_id, error=f"{type(exc).__name__}: {exc}")
