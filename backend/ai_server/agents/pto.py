from __future__ import annotations

from typing import Any

from ai_server.agent_scheduler import AgentScheduler
from ai_server.agent_store import AgentStore
from ai_server.agents.base import BaseSpecialist
from ai_server.agents.pto_llm import PtoAgentLLM, PtoLLMToolCall, pto_llm_failure_result
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentTask, ToolResult, ToolStatus
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.document_access import DocumentToolset


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
        tools: DocumentToolset | None = None,
        llm: PtoAgentLLM | None = None,
        scheduler: AgentScheduler | None = None,
        store: AgentStore | None = None,
    ) -> None:
        super().__init__(
            manifest,
            knowledge_base=knowledge_base,
            skill_store=skill_store,
            retriever=retriever,
            tools=tools,
            llm=llm,
            scheduler=scheduler,
            store=store,
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        document_tools: DocumentToolset | None = None,
        pto_retriever: HybridKnowledgeRetriever | None = None,
        pto_llm: PtoAgentLLM | None = None,
        scheduler: AgentScheduler | None = None,
        **_: Any,
    ) -> PtoSpecialist:
        return cls(manifest, retriever=pto_retriever, tools=document_tools, llm=pto_llm, scheduler=scheduler)

    def tool_definitions(self) -> list[dict]:
        if self.tools is None:
            return []
        return [definition.model_dump() for definition in self.tools.definitions()]

    async def _execute_tool_call(
        self,
        tool_call: PtoLLMToolCall,
        task: AgentTask,
    ) -> tuple[ToolResult | None, ActionRecord | None, list[ActionRecord]]:
        if tool_call.name == "none":
            return None, None, []
        if tool_call.name == "portal_document_search":
            result = self.tools.portal_document_search(tool_call.args)
            return (
                result,
                ActionRecord(name="pto_portal_document_search", status=result.status, details=result.model_dump()),
                [],
            )
        if tool_call.name == "document_read":
            result = await self.tools.document_read(tool_call.args)
            return (
                result,
                ActionRecord(name="pto_document_read", status=result.status, details=result.model_dump()),
                [],
            )
        if tool_call.name == "spreadsheet_preview":
            result = await self.tools.spreadsheet_preview(tool_call.args)
            return (
                result,
                ActionRecord(name="pto_spreadsheet_preview", status=result.status, details=result.model_dump()),
                [],
            )
        if tool_call.name == "spreadsheet_compare":
            result = await self.tools.spreadsheet_compare(tool_call.args)
            return (
                result,
                ActionRecord(name="pto_spreadsheet_compare", status=result.status, details=result.model_dump()),
                [],
            )
        if tool_call.name == "document_draft_create":
            result = self.tools.document_draft_create(tool_call.args)
            return (
                result,
                ActionRecord(name="pto_document_draft_create", status=result.status, details=result.model_dump()),
                [],
            )
        if tool_call.name == "document_draft_list":
            result = self.tools.document_draft_list(tool_call.args)
            return (
                result,
                ActionRecord(name="pto_document_draft_list", status=result.status, details=result.model_dump()),
                [],
            )

        result = ToolResult(
            status=ToolStatus.INVALID_TOOL_CALL,
            tool=tool_call.name,
            error=f"unknown PTO tool call: {tool_call.name}",
        )
        return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump()), []

    def _llm_failure_result(self, message: str):
        return pto_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "PTO specialist is an LLM specialist; backend document tools only search/read/compare and apply access guardrails.",
            "Bitrix24 remains the transport/source layer for portal files; PTO owns document interpretation.",
        ]
