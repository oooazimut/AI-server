from __future__ import annotations

from typing import Any

from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.orchestrators.orchestrator_llm import (
    OrchestratorDecision,
    OrchestratorDecisionResult,
    OrchestratorFinalResult,
    OrchestratorToolCall,
)
from ai_server.specialists import build_specialist_registry
from ai_server.tracing import TraceRecorder


class DiagnosticOrchestratorLLM:
    """Deterministic orchestrator policy for the internal diagnostics contour."""

    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        dialog_history: list[dict[str, str]],
        retrieval_hits: list[Any],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
    ) -> OrchestratorDecisionResult:
        if tool_results:
            decision = OrchestratorDecision(status="completed", answer="", tool_calls=[], confidence=0.9)
        else:
            decision = OrchestratorDecision(
                status="completed",
                answer="",
                tool_calls=[
                    OrchestratorToolCall(
                        name="call_diagnostic_agent",
                        args={"request": task.request},
                        summary="Разобрать feedback, incident и trace.",
                    )
                ],
                confidence=0.9,
            )
        return OrchestratorDecisionResult(
            decision=decision,
            model_usage=ModelUsageRecord(agent_id="internal_orchestrator", provider="internal", model="diagnostics"),
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: OrchestratorDecision,
        tool_results: list[ToolResult],
    ) -> OrchestratorFinalResult:
        diagnostic = next((item for item in tool_results if item.tool == "call_diagnostic_agent"), None)
        answer = str((diagnostic.data or {}).get("answer") or "") if diagnostic and isinstance(diagnostic.data, dict) else ""
        status = str((diagnostic.data or {}).get("status") or diagnostic.status.value) if diagnostic else "failed"
        return OrchestratorFinalResult(
            answer=answer or "Diagnostic Agent не вернул отчет.",
            status=status,
            model_usage=ModelUsageRecord(agent_id="internal_orchestrator", provider="internal", model="diagnostics"),
        )


async def run_diagnostic_via_orchestrator(
    *,
    manifests: list[AgentManifest],
    task: AgentTask,
    diagnostic_llm: Any = None,
    trace_recorder: TraceRecorder | None = None,
) -> Any:
    specialists = build_specialist_registry(
        manifests,
        audience="diagnostics",
        diagnostic_llm=diagnostic_llm,
        trace_recorder=trace_recorder,
    )
    orchestrator = InternalOrchestrator(
        manifests,
        specialists=specialists,
        orchestrator_llm=DiagnosticOrchestratorLLM(),
        trace_recorder=trace_recorder,
    )
    return await orchestrator.handle(task)
