from __future__ import annotations

from typing import Any

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.ports import PortalSearchPort, SchedulerPort
from ai_server.agents.pto.llm import PtoAgentLLM, PtoLLMService, pto_llm_failure_result
from ai_server.agents.pto.tools import (
    DocumentDraftCreateTool,
    DocumentDraftListTool,
    DocumentReadTool,
    DocumentSearchTool,
    SpreadsheetCompareTool,
    SpreadsheetPreviewTool,
)
from ai_server.agents.tool import AgentTool
from ai_server.integrations.bitrix.ports import BitrixFileDownloadPort
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import AgentManifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import Settings
from ai_server.skills import SkillStore


class PtoSpecialist(BaseSpecialist):
    max_steps = 5
    action_prefix = "pto"

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        agent_tools: list[AgentTool] | None = None,
        llm: PtoAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        store: Any | None = None,
    ) -> None:
        super().__init__(
            manifest,
            knowledge_base=knowledge_base,
            skill_store=skill_store,
            retriever=retriever,
            agent_tools=agent_tools,
            llm=llm,
            scheduler=scheduler,
            store=store,
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        bitrix_client: BitrixFileDownloadPort | None = None,
        portal_search_index: PortalSearchPort | None = None,
        settings: Settings | None = None,
        pto_retriever: HybridKnowledgeRetriever | None = None,
        pto_llm: PtoAgentLLM | None = None,
        pto_store: Any | None = None,
        scheduler: SchedulerPort | None = None,
        **_: Any,
    ) -> PtoSpecialist:
        from ai_server.settings import get_settings

        _settings = settings or get_settings()
        tools: list[AgentTool] = [
            DocumentSearchTool(portal_search=portal_search_index, settings=_settings),
            DocumentReadTool(bitrix_client, portal_search=portal_search_index, settings=_settings),
            SpreadsheetPreviewTool(bitrix_client, portal_search=portal_search_index, settings=_settings),
            SpreadsheetCompareTool(bitrix_client, portal_search=portal_search_index, settings=_settings),
            DocumentDraftCreateTool(_settings),
            DocumentDraftListTool(_settings),
        ]
        return cls(
            manifest,
            retriever=pto_retriever,
            agent_tools=tools,
            llm=pto_llm or PtoLLMService(),
            scheduler=scheduler,
            store=pto_store,
        )

    def _llm_failure_result(self, message: str):
        return pto_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "PTO specialist is an LLM specialist; backend document tools only search/read/compare and apply access guardrails.",
            "Bitrix24 remains the transport/source layer for portal files; PTO owns document interpretation.",
        ]
