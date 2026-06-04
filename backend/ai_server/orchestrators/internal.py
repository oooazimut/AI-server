from __future__ import annotations

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask
from ai_server.orchestrator import suggest_agents
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.bitrix import BitrixToolset


class InternalOrchestrator:
    def __init__(
        self,
        manifests: list[AgentManifest],
        *,
        bitrix_retriever: HybridKnowledgeRetriever | None = None,
        bitrix_tools: BitrixToolset | None = None,
    ) -> None:
        self.manifests = manifests
        self.bitrix_retriever = bitrix_retriever
        self.bitrix_tools = bitrix_tools

    async def handle(self, task: AgentTask) -> AgentResult:
        matches = [agent for agent in suggest_agents(task.request, self.manifests) if agent.kind == "specialist"]
        bitrix = next((agent for agent in matches if agent.id == "bitrix24"), None)
        if bitrix is not None:
            specialist_result = await Bitrix24Specialist(
                bitrix,
                retriever=self.bitrix_retriever,
                tools=self.bitrix_tools,
            ).handle(task)
            return AgentResult(
                status=specialist_result.status,
                agent_id="internal_orchestrator",
                answer=specialist_result.answer,
                artifacts=specialist_result.artifacts,
                actions_taken=[
                    ActionRecord(
                        name="delegate_to_specialist",
                        status="completed",
                        details={"specialist": "bitrix24"},
                    ),
                    *specialist_result.actions_taken,
                ],
                actions_requiring_approval=specialist_result.actions_requiring_approval,
                handoff_to=["bitrix24"],
                confidence=specialist_result.confidence,
                logs=specialist_result.logs,
            )

        return AgentResult(
            status="needs_clarification",
            agent_id="internal_orchestrator",
            answer=(
                "Пока не вижу подключенного специалиста под этот запрос. "
                "В MVP уже выделен Битрикс24-специалист; остальные специалисты будут добавляться модулями."
            ),
            actions_taken=[
                ActionRecord(
                    name="route_request",
                    status="no_specialist",
                    details={"available_specialists": [agent.id for agent in self.manifests if agent.kind == "specialist"]},
                )
            ],
            confidence=0.35,
        )
