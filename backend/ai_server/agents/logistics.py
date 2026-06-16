from __future__ import annotations

import logging
from typing import Any

from ai_server.agent_scheduler import AgentScheduler
from ai_server.agent_store import AgentStore
from ai_server.agents.base import BaseSpecialist
from ai_server.agents.logistics_llm import (
    LogisticsAgentLLM,
    LogisticsLLMService,
    LogisticsLLMToolCall,
    logistics_llm_failure_result,
)
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentTask, ToolResult
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.vehicle_usage import VehicleUsageToolset

logger = logging.getLogger(__name__)


class LogisticsSpecialist(BaseSpecialist):
    max_steps = 5
    action_prefix = "logistics"

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        tools: VehicleUsageToolset | None = None,
        llm: LogisticsAgentLLM | None = None,
        scheduler: AgentScheduler | None = None,
        store: AgentStore | None = None,
    ) -> None:
        super().__init__(
            manifest,
            knowledge_base=knowledge_base,
            skill_store=skill_store,
            retriever=retriever,
            tools=tools or VehicleUsageToolset(),
            llm=llm or LogisticsLLMService(),
            scheduler=scheduler,
            store=store,
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        vehicle_usage_tools: VehicleUsageToolset | None = None,
        logistics_retriever: HybridKnowledgeRetriever | None = None,
        logistics_llm: LogisticsAgentLLM | None = None,
        scheduler: AgentScheduler | None = None,
        **_: Any,
    ) -> LogisticsSpecialist:
        return cls(
            manifest,
            retriever=logistics_retriever,
            tools=vehicle_usage_tools,
            llm=logistics_llm,
            scheduler=scheduler,
        )

    def tool_definitions(self) -> list[dict]:
        return [definition.model_dump() for definition in self.tools.definitions()]

    async def _execute_tool_call(
        self,
        tool_call: LogisticsLLMToolCall,
        task: AgentTask,
    ) -> tuple[ToolResult | None, ActionRecord | None, list[ActionRecord]]:
        if tool_call.name == "none":
            return None, None, []
        if tool_call.name == "vehicle_usage_context":
            result = self.tools.vehicle_usage_context(tool_call.args)
            return (
                result,
                ActionRecord(name="logistics_vehicle_usage_context", status=result.status, details=result.model_dump()),
                [],
            )
        if tool_call.name == "vehicle_usage_save_draft":
            result = self.tools.vehicle_usage_save_draft(tool_call.args)
            return (
                result,
                ActionRecord(
                    name="logistics_vehicle_usage_save_draft", status=result.status, details=result.model_dump()
                ),
                [],
            )
        if tool_call.name == "vehicle_usage_save_report":
            result = self.tools.vehicle_usage_save_report(tool_call.args)
            if result.status == "ok":
                date_str = str((tool_call.args or {}).get("request_date") or "")
                cancelled = self.cancel_self_jobs_by_prefix("morning_")
                cancelled += self.cancel_self_jobs_by_prefix("escalation_")
                if cancelled:
                    logger.info(
                        "LogisticsSpecialist: cancelled %d scheduled jobs after report saved %s", cancelled, date_str
                    )
            return (
                result,
                ActionRecord(
                    name="logistics_vehicle_usage_save_report", status=result.status, details=result.model_dump()
                ),
                [],
            )

        result = ToolResult(
            status="invalid_tool_call",
            tool=tool_call.name,
            error=f"unknown Logistics tool call: {tool_call.name}",
        )
        return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump()), []

    def _llm_failure_result(self, message: str):
        return logistics_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Logistics specialist is an LLM specialist; vehicle tools only read/write structured state.",
            "User-facing delivery belongs to the Negotiator/channel runtime.",
            "Bitrix remains the channel/source layer; Logistics owns vehicle usage interpretation.",
        ]
