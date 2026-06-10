from __future__ import annotations

import asyncio

from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask
from ai_server.orchestrators.internal_llm import InternalLLMRouter, InternalOrchestratorLLM
from ai_server.specialists import Specialist, build_specialist_registry


class InternalOrchestrator:
    def __init__(
        self,
        manifests: list[AgentManifest],
        specialists: dict[str, Specialist] | None = None,
        *,
        orchestrator_llm: InternalOrchestratorLLM | None = None,
    ) -> None:
        self.manifests = manifests
        self.specialists = specialists or build_specialist_registry(manifests)
        self.orchestrator_llm = orchestrator_llm or InternalLLMRouter()

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
        route_action = ActionRecord(
            name="orchestrator_llm_route",
            status=decision.status,
            details={"handoff_to": decision.handoff_to, "confidence": decision.confidence},
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
                actions_taken=[route_action],
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

        all_actions = [route_action, *delegate_actions, *[a for _, sr in good for a in sr.actions_taken]]
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


def _merge_status(statuses: list[str]) -> str:
    priority = {"failed": 0, "needs_human": 1, "needs_clarification": 2, "completed": 3}
    return min(statuses, key=lambda s: priority.get(s, 2))
