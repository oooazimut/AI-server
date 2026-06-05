from __future__ import annotations

from ai_server.agents.logistics_llm import (
    LogisticsAgentLLM,
    LogisticsLLMService,
    LogisticsLLMToolCall,
    logistics_llm_failure_result,
)
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ToolResult
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.vehicle_usage import VehicleUsageToolset


class LogisticsSpecialist:
    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        tools: VehicleUsageToolset | None = None,
        llm: LogisticsAgentLLM | None = None,
    ) -> None:
        self.manifest = manifest
        self.knowledge_base = knowledge_base or MarkdownKnowledgeBase()
        self.skill_store = skill_store or SkillStore()
        self.retriever = retriever or HybridKnowledgeRetriever(knowledge_base=self.knowledge_base)
        self.tools = tools or VehicleUsageToolset()
        self.llm = llm or LogisticsLLMService()

    async def handle(self, task: AgentTask) -> AgentResult:
        available_skills = self.skill_store.list_skills(self.manifest)
        retrieval_hits = self.retriever.search(self.manifest, task.request, limit=5)
        actions_taken = [
            ActionRecord(
                name="load_logistics_specialist_context",
                status="completed",
                details={
                    "available_skills": [
                        {"id": skill.id, "title": skill.title, "preview": skill.preview}
                        for skill in available_skills
                    ],
                    "retrieval_topics": _unique([hit.chunk.topic for hit in retrieval_hits]),
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
                },
            )
        ]

        tool_results: list[ToolResult] = []
        decision_results = []
        decision = None
        max_steps = 5
        for step in range(1, max_steps + 1):
            try:
                decision_result = await self.llm.decide(
                    manifest=self.manifest,
                    task=task,
                    retrieval_hits=retrieval_hits,
                    tool_definitions=self.tool_definitions(),
                    tool_results=list(tool_results),
                )
            except Exception as exc:
                failure = logistics_llm_failure_result(f"{type(exc).__name__}: {exc}")
                return AgentResult(
                    status="failed",
                    agent_id=self.manifest.id,
                    answer=failure.answer,
                    actions_taken=[
                        *actions_taken,
                        ActionRecord(
                            name="logistics_llm_decision",
                            status="error",
                            details={"step": step, "error": f"{type(exc).__name__}: {exc}"},
                        ),
                    ],
                    model_usage=[*[item.model_usage for item in decision_results], failure.model_usage],
                    confidence=0.0,
                    logs=_logs(),
                )

            decision_results.append(decision_result)
            decision = decision_result.decision
            actions_taken.append(
                ActionRecord(
                    name="logistics_llm_decision",
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
                result, action = await self._execute_tool_call(tool_call)
                if result is not None:
                    tool_results.append(result)
                if action is not None:
                    actions_taken.append(action)
            if step == max_steps:
                actions_taken.append(
                    ActionRecord(
                        name="logistics_tool_loop_guardrail",
                        status="stopped",
                        details={"max_steps": max_steps},
                    )
                )

        if decision is None:
            failure = logistics_llm_failure_result("empty Logistics LLM decision loop")
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer=failure.answer,
                actions_taken=actions_taken,
                model_usage=[failure.model_usage],
                confidence=0.0,
                logs=_logs(),
            )

        try:
            final_result = await self.llm.compose(task=task, decision=decision, tool_results=tool_results)
        except Exception as exc:
            failure = logistics_llm_failure_result(f"{type(exc).__name__}: {exc}")
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer=failure.answer,
                actions_taken=[
                    *actions_taken,
                    ActionRecord(
                        name="logistics_llm_final_answer",
                        status="error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    ),
                ],
                model_usage=[*[item.model_usage for item in decision_results], failure.model_usage],
                confidence=0.0,
                logs=_logs(),
            )

        actions_taken.append(ActionRecord(name="logistics_llm_final_answer", status=final_result.status))
        return AgentResult(
            status=final_result.status,
            agent_id=self.manifest.id,
            answer=final_result.answer,
            actions_taken=actions_taken,
            model_usage=[*[item.model_usage for item in decision_results], final_result.model_usage],
            confidence=decision.confidence,
            logs=_logs(),
        )

    def tool_definitions(self) -> list[dict]:
        return [definition.model_dump() for definition in self.tools.definitions()]

    async def _execute_tool_call(
        self,
        tool_call: LogisticsLLMToolCall,
    ) -> tuple[ToolResult | None, ActionRecord | None]:
        if tool_call.name == "none":
            return None, None
        if tool_call.name == "vehicle_usage_context":
            result = self.tools.vehicle_usage_context(tool_call.args)
            return result, ActionRecord(name="logistics_vehicle_usage_context", status=result.status, details=result.model_dump())
        if tool_call.name == "vehicle_usage_save_draft":
            result = self.tools.vehicle_usage_save_draft(tool_call.args)
            return result, ActionRecord(name="logistics_vehicle_usage_save_draft", status=result.status, details=result.model_dump())
        if tool_call.name == "vehicle_usage_save_report":
            result = self.tools.vehicle_usage_save_report(tool_call.args)
            return result, ActionRecord(name="logistics_vehicle_usage_save_report", status=result.status, details=result.model_dump())
        if tool_call.name == "vehicle_usage_send_message":
            result = await self.tools.vehicle_usage_send_message(tool_call.args)
            return result, ActionRecord(name="logistics_vehicle_usage_send_message", status=result.status, details=result.model_dump())

        result = ToolResult(
            status="invalid_tool_call",
            tool=tool_call.name,
            error=f"unknown Logistics tool call: {tool_call.name}",
        )
        return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump())


def _logs() -> list[str]:
    return [
        "Logistics specialist is an LLM specialist; vehicle tools only read/write structured state and send messages.",
        "Bitrix remains the channel/source layer; Logistics owns vehicle usage interpretation.",
    ]


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
