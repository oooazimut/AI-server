from __future__ import annotations

from typing import Any

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.diagnostic_agent.llm import (
    DiagnosticAgentLLM,
    DiagnosticLLMService,
    diagnostic_llm_failure_result,
)
from ai_server.agents.ports import SchedulerPort
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentManifest
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

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        diagnostic_retriever: HybridKnowledgeRetriever | None = None,
        diagnostic_llm: DiagnosticAgentLLM | None = None,
        diagnostic_store: Any | None = None,
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
        )

    def _llm_failure_result(self, message: str):  # noqa: ANN201
        return diagnostic_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Diagnostic Agent analyzes feedback, learning events, trace and agent actions.",
            "Diagnostic Agent is not a business specialist and must not call business systems directly.",
            "Rule/code changes are recommendations only until a human approves them.",
        ]
