from __future__ import annotations

from typing import Any

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.ports import SchedulerPort
from ai_server.agents.secure_org_data.llm import (
    SecureOrgDataLLM,
    SecureOrgDataLLMService,
    secure_org_data_llm_failure_result,
)
from ai_server.agents.secure_org_data.store import SecureOrgDataStore
from ai_server.agents.secure_org_data.tools import SecureOrgDataSearchTool
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentManifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import Settings, get_settings
from ai_server.skills import SkillStore
from ai_server.tracing import TraceRecorder


class SecureOrgDataAgent(BaseSpecialist):
    max_steps = 2
    action_prefix = "secure_org_data"

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        settings: Settings | None = None,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        llm: SecureOrgDataLLM | None = None,
        scheduler: SchedulerPort | None = None,
        store: SecureOrgDataStore | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.secure_store = store or SecureOrgDataStore(settings=self.settings)
        super().__init__(
            manifest,
            knowledge_base=knowledge_base,
            skill_store=skill_store,
            retriever=retriever,
            agent_tools=[SecureOrgDataSearchTool(self.secure_store)],
            llm=llm,
            scheduler=scheduler,
            store=None,
            trace_recorder=trace_recorder,
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        settings: Settings | None = None,
        secure_org_data_retriever: HybridKnowledgeRetriever | None = None,
        secure_org_data_llm: SecureOrgDataLLM | None = None,
        secure_org_data_store: SecureOrgDataStore | None = None,
        scheduler: SchedulerPort | None = None,
        trace_recorder: TraceRecorder | None = None,
        **_: Any,
    ) -> SecureOrgDataAgent:
        return cls(
            manifest,
            settings=settings,
            retriever=secure_org_data_retriever,
            llm=secure_org_data_llm or SecureOrgDataLLMService(),
            scheduler=scheduler,
            store=secure_org_data_store,
            trace_recorder=trace_recorder,
        )

    async def _load_extra_context(self, task):  # noqa: ANN001, ANN201
        return task, {"secure_org_data_status": self.secure_store.status(), "mode": "read_only"}

    def _llm_failure_result(self, message: str):  # noqa: ANN201
        return secure_org_data_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Secure Org Data Agent uses existing access metadata and content indexes.",
            "Secure Org Data Agent is read-only in the current stage.",
            "Access level is not inferred from words inside content; it comes from explicit metadata/index markers.",
        ]
