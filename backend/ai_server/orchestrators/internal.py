from __future__ import annotations

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.bitrix_llm import BitrixAgentLLM
from ai_server.agents.pto import PtoSpecialist
from ai_server.agents.pto_llm import PtoAgentLLM
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask
from ai_server.orchestrators.internal_llm import InternalLLMRouter, InternalOrchestratorLLM
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.tools.bitrix import BitrixToolset
from ai_server.tools.document_access import DocumentToolset


class InternalOrchestrator:
    def __init__(
        self,
        manifests: list[AgentManifest],
        *,
        bitrix_retriever: HybridKnowledgeRetriever | None = None,
        bitrix_tools: BitrixToolset | None = None,
        bitrix_llm: BitrixAgentLLM | None = None,
        pto_retriever: HybridKnowledgeRetriever | None = None,
        document_tools: DocumentToolset | None = None,
        pto_llm: PtoAgentLLM | None = None,
        orchestrator_llm: InternalOrchestratorLLM | None = None,
    ) -> None:
        self.manifests = manifests
        self.bitrix_retriever = bitrix_retriever
        self.bitrix_tools = bitrix_tools
        self.bitrix_llm = bitrix_llm
        self.pto_retriever = pto_retriever
        self.document_tools = document_tools
        self.pto_llm = pto_llm
        self.orchestrator_llm = orchestrator_llm or InternalLLMRouter()

    async def handle(self, task: AgentTask) -> AgentResult:
        try:
            route_result = await self.orchestrator_llm.route(task=task, manifests=self.manifests)
        except Exception as exc:
            return AgentResult(
                status="failed",
                agent_id="internal_orchestrator",
                answer=f"Не смог обработать запрос через LLM-оркестратор: {type(exc).__name__}: {exc}",
                actions_taken=[
                    ActionRecord(
                        name="orchestrator_llm_route",
                        status="error",
                        details={"error": f"{type(exc).__name__}: {exc}"},
                    )
                ],
                confidence=0.0,
            )

        decision = route_result.decision
        route_action = ActionRecord(
            name="orchestrator_llm_route",
            status=decision.status,
            details={"handoff_to": decision.handoff_to, "confidence": decision.confidence},
        )
        bitrix = _manifest_by_id(self.manifests, "bitrix24") if "bitrix24" in decision.handoff_to else None
        if bitrix is not None:
            specialist_result = await Bitrix24Specialist(
                bitrix,
                retriever=self.bitrix_retriever,
                tools=self.bitrix_tools,
                llm=self.bitrix_llm,
            ).handle(task)
            return AgentResult(
                status=specialist_result.status,
                agent_id="internal_orchestrator",
                answer=specialist_result.answer,
                artifacts=specialist_result.artifacts,
                actions_taken=[
                    route_action,
                    ActionRecord(
                        name="delegate_to_specialist",
                        status="completed",
                        details={"specialist": "bitrix24"},
                    ),
                    *specialist_result.actions_taken,
                ],
                actions_requiring_approval=specialist_result.actions_requiring_approval,
                model_usage=[route_result.model_usage, *specialist_result.model_usage],
                handoff_to=["bitrix24"],
                confidence=specialist_result.confidence,
                logs=specialist_result.logs,
            )

        pto = _manifest_by_id(self.manifests, "pto") if "pto" in decision.handoff_to else None
        if pto is not None:
            specialist_result = await PtoSpecialist(
                pto,
                retriever=self.pto_retriever,
                tools=self.document_tools,
                llm=self.pto_llm,
            ).handle(task)
            return AgentResult(
                status=specialist_result.status,
                agent_id="internal_orchestrator",
                answer=specialist_result.answer,
                artifacts=specialist_result.artifacts,
                actions_taken=[
                    route_action,
                    ActionRecord(
                        name="delegate_to_specialist",
                        status="completed",
                        details={"specialist": "pto"},
                    ),
                    *specialist_result.actions_taken,
                ],
                actions_requiring_approval=specialist_result.actions_requiring_approval,
                model_usage=[route_result.model_usage, *specialist_result.model_usage],
                handoff_to=["pto"],
                confidence=specialist_result.confidence,
                logs=specialist_result.logs,
            )

        return AgentResult(
            status=decision.status,
            agent_id="internal_orchestrator",
            answer=decision.answer,
            actions_taken=[route_action],
            model_usage=[route_result.model_usage],
            confidence=decision.confidence,
        )


def _manifest_by_id(manifests: list[AgentManifest], agent_id: str) -> AgentManifest | None:
    return next((manifest for manifest in manifests if manifest.id == agent_id), None)
