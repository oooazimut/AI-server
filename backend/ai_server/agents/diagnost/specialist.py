from __future__ import annotations

from typing import Any

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.diagnost.llm import DiagnostAgentLLM, DiagnostLLMService, diagnost_llm_failure_result
from ai_server.agents.diagnost.tools import (
    CreateIncidentTool,
    ErrorReportTool,
    GetIncidentTool,
    ListIncidentsTool,
    SearchEventsTool,
)
from ai_server.agents.ports import SchedulerPort
from ai_server.agents.tool import AgentTool
from ai_server.models import AgentManifest


class DiagnostSpecialist(BaseSpecialist):
    max_steps = 5
    action_prefix = "diagnost"

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        diagnost_store: Any | None = None,
        diagnost_llm: DiagnostAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        **_: Any,
    ) -> DiagnostSpecialist:
        tools: list[AgentTool] = [
            SearchEventsTool(diagnost_store),
            GetIncidentTool(diagnost_store),
            ListIncidentsTool(diagnost_store),
            CreateIncidentTool(diagnost_store),
            ErrorReportTool(diagnost_store),
        ]
        return cls(
            manifest,
            agent_tools=tools,
            llm=diagnost_llm or DiagnostLLMService(),
            scheduler=scheduler,
            store=diagnost_store,
        )

    def _llm_failure_result(self, message: str):  # noqa: ANN201
        return diagnost_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "ИИ-Диагност анализирует качество работы системы: инциденты, ошибки, паттерны.",
            "Не работает с Bitrix или бизнес-данными — только с диагностическим хранилищем.",
        ]
