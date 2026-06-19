from __future__ import annotations

from typing import Any

from ai_server.agent_scheduler import SchedulerPort
from ai_server.agent_store import AgentStore
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ToolResult
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.utils import unique

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

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        tools: Any = None,
        llm: Any = None,
        scheduler: SchedulerPort | None = None,
        store: AgentStore | None = None,
    ) -> None:
        self.manifest = manifest
        self.knowledge_base = knowledge_base
        self.skill_store = skill_store
        self.retriever = retriever
        self.tools = tools
        self.llm = llm
        self._scheduler = scheduler
        self.store = store

    # ------------------------------------------------------------------
    # Hooks subclasses implement/override
    # ------------------------------------------------------------------

    def tool_definitions(self) -> list[dict]:
        raise NotImplementedError

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
        raise NotImplementedError

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
                    dialog_history=task.context.get("dialog_history"),
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
                    },
                )
            )

            executable_calls = [call for call in decision.tool_calls if call.name != "none"]
            if not executable_calls:
                break
            for tool_call in executable_calls:
                result, action, approvals = await self._execute_tool_call(tool_call, task)
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
        status = "needs_human" if approval_actions else final_result.status

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
