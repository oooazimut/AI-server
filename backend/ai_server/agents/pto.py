from __future__ import annotations

from ai_server.agents.pto_llm import PtoAgentLLM, PtoLLMService, PtoLLMToolCall, pto_llm_failure_result
from ai_server.knowledge import MarkdownKnowledgeBase
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ToolResult
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.skills import SkillStore
from ai_server.tools.document_access import DocumentToolset


class PtoSpecialist:
    def __init__(
        self,
        manifest: AgentManifest,
        *,
        knowledge_base: MarkdownKnowledgeBase | None = None,
        skill_store: SkillStore | None = None,
        retriever: HybridKnowledgeRetriever | None = None,
        tools: DocumentToolset | None = None,
        llm: PtoAgentLLM | None = None,
    ) -> None:
        self.manifest = manifest
        self.knowledge_base = knowledge_base or MarkdownKnowledgeBase()
        self.skill_store = skill_store or SkillStore()
        self.retriever = retriever or HybridKnowledgeRetriever(knowledge_base=self.knowledge_base)
        self.tools = tools or DocumentToolset()
        self.llm = llm or PtoLLMService()

    async def handle(self, task: AgentTask) -> AgentResult:
        available_skills = self.skill_store.list_skills(self.manifest)
        retrieval_hits = self.retriever.search(self.manifest, task.request, limit=5)
        actions_taken = [
            ActionRecord(
                name="load_pto_specialist_context",
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

        try:
            decision_result = await self.llm.decide(
                manifest=self.manifest,
                task=task,
                retrieval_hits=retrieval_hits,
                tool_definitions=self.tool_definitions(),
            )
        except Exception as exc:
            failure = pto_llm_failure_result(f"{type(exc).__name__}: {exc}")
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer=failure.answer,
                actions_taken=[
                    *actions_taken,
                    ActionRecord(
                        name="pto_llm_decision",
                        status="error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    ),
                ],
                model_usage=[failure.model_usage],
                confidence=0.0,
                logs=_logs(),
            )

        decision = decision_result.decision
        actions_taken.append(
            ActionRecord(
                name="pto_llm_decision",
                status=decision.status,
                details={
                    "tool_calls": [
                        {"name": call.name, "args": call.args, "summary": call.summary}
                        for call in decision.tool_calls
                    ],
                    "confidence": decision.confidence,
                },
            )
        )

        tool_results: list[ToolResult] = []
        for tool_call in decision.tool_calls:
            result, action = await self._execute_tool_call(tool_call)
            if result is not None:
                tool_results.append(result)
            if action is not None:
                actions_taken.append(action)

        try:
            final_result = await self.llm.compose(
                task=task,
                decision=decision,
                tool_results=tool_results,
            )
        except Exception as exc:
            failure = pto_llm_failure_result(f"{type(exc).__name__}: {exc}")
            return AgentResult(
                status="failed",
                agent_id=self.manifest.id,
                answer=failure.answer,
                actions_taken=[
                    *actions_taken,
                    ActionRecord(
                        name="pto_llm_final_answer",
                        status="error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    ),
                ],
                model_usage=[decision_result.model_usage, failure.model_usage],
                confidence=0.0,
                logs=_logs(),
            )

        actions_taken.append(ActionRecord(name="pto_llm_final_answer", status=final_result.status))
        return AgentResult(
            status=final_result.status,
            agent_id=self.manifest.id,
            answer=final_result.answer,
            actions_taken=actions_taken,
            model_usage=[decision_result.model_usage, final_result.model_usage],
            confidence=decision.confidence,
            logs=_logs(),
        )

    def tool_definitions(self) -> list[dict]:
        return [definition.model_dump() for definition in self.tools.definitions()]

    async def _execute_tool_call(
        self,
        tool_call: PtoLLMToolCall,
    ) -> tuple[ToolResult | None, ActionRecord | None]:
        if tool_call.name == "none":
            return None, None
        if tool_call.name == "portal_document_search":
            result = self.tools.portal_document_search(tool_call.args)
            return result, ActionRecord(name="pto_portal_document_search", status=result.status, details=result.model_dump())
        if tool_call.name == "document_read":
            result = await self.tools.document_read(tool_call.args)
            return result, ActionRecord(name="pto_document_read", status=result.status, details=result.model_dump())
        if tool_call.name == "spreadsheet_compare":
            result = await self.tools.spreadsheet_compare(tool_call.args)
            return result, ActionRecord(name="pto_spreadsheet_compare", status=result.status, details=result.model_dump())

        result = ToolResult(
            status="invalid_tool_call",
            tool=tool_call.name,
            error=f"unknown PTO tool call: {tool_call.name}",
        )
        return result, ActionRecord(name=tool_call.name, status=result.status, details=result.model_dump())


def _logs() -> list[str]:
    return [
        "PTO specialist is an LLM specialist; backend document tools only search/read/compare and apply access guardrails.",
        "Bitrix24 remains the transport/source layer for portal files; PTO owns document interpretation.",
    ]


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
