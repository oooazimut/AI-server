from __future__ import annotations

from typing import Any

from ai_server.agents.base import BaseSpecialist
from ai_server.agents.kartoteka.llm import KartotekaAgentLLM, KartotekaLLMService, kartoteka_llm_failure_result
from ai_server.agents.kartoteka.tools import (
    FileAddTool,
    FileDeleteTool,
    FileMoveTool,
    KartotekaContextTool,
    KartotekaSearchTool,
)
from ai_server.agents.ports import SchedulerPort
from ai_server.agents.tool import AgentTool
from ai_server.models import AgentManifest


class KartotekaSpecialist(BaseSpecialist):
    max_steps = 5
    action_prefix = "kartoteka"

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        kartoteka_store: Any | None = None,
        kartoteka_llm: KartotekaAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        **_: Any,
    ) -> KartotekaSpecialist:
        tools: list[AgentTool] = [
            KartotekaSearchTool(kartoteka_store),
            KartotekaContextTool(kartoteka_store),
            FileAddTool(kartoteka_store),
            FileDeleteTool(kartoteka_store),
            FileMoveTool(kartoteka_store),
        ]
        return cls(
            manifest,
            agent_tools=tools,
            llm=kartoteka_llm or KartotekaLLMService(),
            scheduler=scheduler,
            store=kartoteka_store,
        )

    def _llm_failure_result(self, message: str):  # noqa: ANN201
        return kartoteka_llm_failure_result(message, agent_id=self.manifest.id)

    def _logs(self) -> list[str]:
        return [
            "Картотека — read-only специалист по локальному файловому индексу.",
            "Операции записи (добавление, удаление, перемещение) отключены до подключения файлового сервера.",
            "Работа с Bitrix-диском — зона ИИ-Битрикс, не Картотека.",
        ]
