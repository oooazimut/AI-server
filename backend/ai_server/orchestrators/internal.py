from __future__ import annotations

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.models import ActionRecord, AgentManifest, AgentResult, AgentTask, ModelUsageRecord
from ai_server.orchestrator import suggest_agents
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
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
        if _looks_like_model_question(task.request):
            settings = get_settings()
            configured = "подключён" if settings.llm_configured else "пока не подключён ключом"
            return AgentResult(
                status="completed",
                agent_id="internal_orchestrator",
                answer=(
                    "Я новый внутренний оркестратор AI-server. "
                    f"LLM-контур в конфиге: provider `{settings.llm_provider}`, "
                    f"model `{settings.llm_model}`; gateway {configured}. "
                    "Bitrix24-сценарии MVP сейчас частично выполняются детерминированными skills."
                ),
                actions_taken=[
                    ActionRecord(
                        name="report_runtime_model",
                        status="completed",
                        details={
                            "llm_provider": settings.llm_provider,
                            "llm_model": settings.llm_model,
                            "llm_configured": settings.llm_configured,
                        },
                    )
                ],
                model_usage=[
                    ModelUsageRecord(
                        agent_id="internal_orchestrator",
                        provider=settings.llm_provider,
                        model=settings.llm_model,
                        status="configured",
                    )
                ],
                confidence=0.95,
            )

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
                model_usage=specialist_result.model_usage,
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


def _looks_like_model_question(text: str) -> bool:
    normalized = text.casefold()
    model_markers = (
        "какая модель",
        "какая ты модель",
        "что за модель",
        "на какой модели",
        "какой llm",
        "какая llm",
        "модель под капотом",
        "deepseek",
        "дипсик",
    )
    return any(marker in normalized for marker in model_markers)
