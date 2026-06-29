from __future__ import annotations

from typing import Any

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.diagnostic_agent.llm import (
    DiagnosticAgentLLM,
    DiagnosticLLMService,
    diagnostic_llm_failure_result,
)
from ai_server.agents.diagnostic_agent.error_report import ErrorReportService, format_error_report_markdown
from ai_server.agents.ports import SchedulerPort
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, Artifact, ModelUsageRecord
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tracing import TraceRecorder


class DiagnosticAgent(BaseSpecialist):
    max_steps = 1
    action_prefix = "diagnostic"

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        llm: DiagnosticAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        store: Any | None = None,
        trace_recorder: TraceRecorder | None = None,
        error_report_service: ErrorReportService | None = None,
    ) -> None:
        super().__init__(
            manifest,
            knowledge_base=knowledge_base,
            skill_store=skill_store,
            retriever=retriever,
            agent_tools=[],
            llm=llm,
            scheduler=scheduler,
            store=store,
            trace_recorder=trace_recorder,
        )
        self._error_report_service = error_report_service

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        diagnostic_retriever: HybridKnowledgeRetriever | None = None,
        diagnostic_llm: DiagnosticAgentLLM | None = None,
        diagnostic_store: Any | None = None,
        learning_recorder: Any | None = None,
        error_report_service: ErrorReportService | None = None,
        scheduler: SchedulerPort | None = None,
        trace_recorder: TraceRecorder | None = None,
        **_: Any,
    ) -> DiagnosticAgent:
        return cls(
            manifest,
            retriever=diagnostic_retriever,
            llm=diagnostic_llm or DiagnosticLLMService(),
            scheduler=scheduler,
            store=diagnostic_store,
            trace_recorder=trace_recorder,
            error_report_service=error_report_service
            or (ErrorReportService(learning_recorder) if learning_recorder is not None else None),
        )

    def _llm_failure_result(self, message: str):  # noqa: ANN201
        return diagnostic_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Diagnostic Agent analyzes feedback, learning events, trace and agent actions.",
            "Diagnostic Agent is not a business specialist and must not call business systems directly.",
            "Rule/code changes are recommendations only until a human approves them.",
        ]

    async def _early_result(self, task: Any, actions_taken: list[ActionRecord]) -> AgentResult | None:
        request = task.context.get("error_report_request") if isinstance(task.context, dict) else None
        if not isinstance(request, dict):
            return None
        if self._error_report_service is None:
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer="ErrorReportService не настроен для Diagnostic Agent.",
                actions_taken=[
                    *actions_taken,
                    ActionRecord(
                        name="diagnostic_error_report",
                        status="failed",
                        details={"reason": "error_report_service_not_configured"},
                    ),
                ],
                model_usage=[
                    ModelUsageRecord(
                        agent_id=self.manifest.id,
                        provider="internal",
                        model="error_report_service",
                        status="error",
                    )
                ],
                confidence=0.0,
                logs=self._logs(),
            )
        since_hours = int(request.get("since_hours") or 24)
        limit = int(request.get("limit") or 200)
        max_groups = int(request.get("max_groups") or 5)
        report = self._error_report_service.build(since_hours=since_hours, limit=limit)
        markdown = format_error_report_markdown(report, max_groups=max_groups)
        return AgentResult(
            status="completed",
            agent_id=self.manifest.id,
            answer=markdown,
            artifacts=[
                Artifact(
                    type="diagnostic_error_report",
                    title="Diagnostic Agent error report",
                    metadata={"report": report},
                )
            ],
            actions_taken=[
                *actions_taken,
                ActionRecord(
                    name="diagnostic_error_report",
                    status="completed",
                    details={
                        "since_hours": since_hours,
                        "limit": limit,
                        "groups": len(report.get("groups") or []),
                        "total_incidents": report.get("total_incidents"),
                    },
                ),
            ],
            model_usage=[
                ModelUsageRecord(
                    agent_id=self.manifest.id,
                    provider="internal",
                    model="error_report_service",
                )
            ],
            confidence=1.0,
            logs=self._logs(),
        )
