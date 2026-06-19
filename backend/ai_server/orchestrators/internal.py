from __future__ import annotations

import asyncio
import logging
import uuid

from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentResultStatus, AgentTask
from ai_server.orchestrators.internal_llm import InternalLLMRouter, InternalOrchestratorLLM, ScheduledTaskDecision
from ai_server.specialists import Specialist, build_specialist_registry

logger = logging.getLogger(__name__)


class InternalOrchestrator:
    def __init__(
        self,
        manifests: list[AgentManifest],
        specialists: dict[str, Specialist] | None = None,
        *,
        orchestrator_llm: InternalOrchestratorLLM | None = None,
        scheduler=None,
    ) -> None:
        self.manifests = manifests
        self.specialists = specialists or build_specialist_registry(manifests)
        self.orchestrator_llm = orchestrator_llm or InternalLLMRouter()
        self.scheduler = scheduler

    async def handle(self, task: AgentTask) -> AgentResult:
        if task.context.get("pending_action"):
            pending_data = task.context["pending_action"]
            default_specialist_id = next((m.id for m in self.manifests if m.kind == "specialist"), "")
            specialist_id = (
                pending_data.get("specialist_id") or default_specialist_id
                if isinstance(pending_data, dict)
                else default_specialist_id
            )
            specialist = self.specialists.get(specialist_id)
            if specialist is not None:
                specialist_result = await specialist.handle(task)
                return AgentResult(
                    status=specialist_result.status,
                    agent_id="internal_orchestrator",
                    answer=specialist_result.answer,
                    artifacts=specialist_result.artifacts,
                    actions_taken=[
                        ActionRecord(
                            name="orchestrator_pending_route",
                            status="completed",
                            details={"handoff_to": specialist_id, "reason": "pending_action"},
                        ),
                        *specialist_result.actions_taken,
                    ],
                    actions_requiring_approval=specialist_result.actions_requiring_approval,
                    model_usage=specialist_result.model_usage,
                    handoff_to=[specialist_id],
                    confidence=specialist_result.confidence,
                    logs=specialist_result.logs,
                )

        try:
            route_result = await self.orchestrator_llm.route(task=task, manifests=self.manifests)
        except Exception as exc:
            return AgentResult(
                status="failed",
                agent_id="internal_orchestrator",
                answer=f"Не смог обработать запрос через LLM-оркестратор: {type(exc).__name__}: {exc}",
                actions_taken=[
                    ActionRecord(
                        name="orchestrator_llm_route",
                        status="error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    )
                ],
                confidence=0.0,
            )

        decision = route_result.decision
        schedule_actions = _apply_scheduled_tasks(decision.scheduled_tasks, self.scheduler)
        route_action = ActionRecord(
            name="orchestrator_llm_route",
            status=decision.status,
            details={
                "handoff_to": decision.handoff_to,
                "scheduled_tasks": len(decision.scheduled_tasks),
                "confidence": decision.confidence,
            },
        )

        targets = [
            (agent_id, specialist)
            for agent_id in decision.handoff_to
            if (specialist := self.specialists.get(agent_id)) is not None
        ]

        if not targets:
            return AgentResult(
                status=decision.status,
                agent_id="internal_orchestrator",
                answer=decision.answer,
                actions_taken=[route_action, *schedule_actions],
                model_usage=[route_result.model_usage],
                confidence=decision.confidence,
            )

        raw_results = await asyncio.gather(
            *[specialist.handle(task) for _, specialist in targets],
            return_exceptions=True,
        )

        good: list[tuple[str, AgentResult]] = []
        delegate_actions: list[ActionRecord] = []
        all_model_usage = [route_result.model_usage]

        for (agent_id, _), result in zip(targets, raw_results, strict=False):
            if isinstance(result, Exception):
                delegate_actions.append(
                    ActionRecord(
                        name="delegate_to_specialist",
                        status="error",
                        details={"specialist": agent_id, "error": f"{type(result).__name__}: {result}"},
                    )
                )
            else:
                delegate_actions.append(
                    ActionRecord(
                        name="delegate_to_specialist",
                        status="completed",
                        details={"specialist": agent_id},
                    )
                )
                good.append((agent_id, result))
                all_model_usage.extend(result.model_usage)

        if not good:
            return AgentResult(
                status="failed",
                agent_id="internal_orchestrator",
                answer="Специалисты не смогли обработать запрос.",
                actions_taken=[route_action, *delegate_actions],
                model_usage=all_model_usage,
                confidence=0.0,
            )

        all_actions = [
            route_action,
            *schedule_actions,
            *delegate_actions,
            *[a for _, sr in good for a in sr.actions_taken],
        ]
        all_approvals = [a for _, sr in good for a in sr.actions_requiring_approval]
        all_artifacts = [art for _, sr in good for art in sr.artifacts]
        all_logs = [log for _, sr in good for log in sr.logs]
        agent_ids = [aid for aid, _ in good]

        if len(good) == 1:
            agent_id, sr = good[0]
            return AgentResult(
                status=sr.status,
                agent_id="internal_orchestrator",
                answer=sr.answer,
                artifacts=sr.artifacts,
                actions_taken=all_actions,
                actions_requiring_approval=all_approvals,
                model_usage=all_model_usage,
                handoff_to=[agent_id],
                confidence=sr.confidence,
                logs=sr.logs,
            )

        # Multiple specialists — synthesize answers into one response
        try:
            synthesis = await self.orchestrator_llm.synthesize(task=task, specialist_results=good)
            all_model_usage.append(synthesis.model_usage)
            all_actions.append(
                ActionRecord(
                    name="orchestrator_synthesize",
                    status="completed",
                    details={"specialists": agent_ids},
                )
            )
            synthesized_answer = synthesis.answer
        except Exception as exc:
            all_actions.append(
                ActionRecord(
                    name="orchestrator_synthesize",
                    status="error",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            # Fall back to first specialist's answer
            synthesized_answer = good[0][1].answer

        status = "needs_human" if all_approvals else _merge_status([sr.status for _, sr in good])
        return AgentResult(
            status=status,
            agent_id="internal_orchestrator",
            answer=synthesized_answer,
            artifacts=all_artifacts,
            actions_taken=all_actions,
            actions_requiring_approval=all_approvals,
            model_usage=all_model_usage,
            handoff_to=agent_ids,
            confidence=min(sr.confidence for _, sr in good),
            logs=all_logs,
        )


_STATUS_PRIORITY: dict[AgentResultStatus, int] = {
    "failed": 0,
    "needs_human": 1,
    "needs_clarification": 2,
    "completed": 3,
}


def _merge_status(statuses: list[AgentResultStatus]) -> AgentResultStatus:
    return min(statuses, key=lambda s: _STATUS_PRIORITY.get(s, 2))


def _apply_scheduled_tasks(
    tasks: list[ScheduledTaskDecision],
    scheduler,
) -> list[ActionRecord]:
    if not tasks or scheduler is None:
        return []
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger

    from ai_server.utils import MOSCOW_TZ

    actions: list[ActionRecord] = []
    for task in tasks:
        job_id = task.job_id or str(uuid.uuid4())[:8]
        try:
            trigger_data = task.trigger
            trigger_type = str(trigger_data.get("type") or "date").strip()
            if trigger_type == "date":
                trigger = DateTrigger(run_date=trigger_data.get("run_date"), timezone=MOSCOW_TZ)
            elif trigger_type == "cron":
                kwargs = {k: v for k, v in trigger_data.items() if k != "type"}
                trigger = CronTrigger(timezone=MOSCOW_TZ, **kwargs)
            else:
                raise ValueError(f"Unknown trigger type: {trigger_type!r}")

            job = scheduler.schedule_task(task.agent_id, job_id, trigger, task.task_description)
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            actions.append(
                ActionRecord(
                    name="orchestrator_schedule_task",
                    status="scheduled",
                    details={"agent_id": task.agent_id, "job_id": f"{task.agent_id}:{job_id}", "next_run": next_run},
                )
            )
            logger.info("Orchestrator scheduled task %s:%s next=%s", task.agent_id, job_id, next_run)
        except Exception as exc:
            logger.exception("Orchestrator failed to schedule task for %s", task.agent_id)
            actions.append(
                ActionRecord(
                    name="orchestrator_schedule_task",
                    status="error",
                    details={"agent_id": task.agent_id, "job_id": job_id, "error": f"{type(exc).__name__}: {exc}"},
                )
            )
    return actions
