from __future__ import annotations

import logging
from typing import Any

from ai_server.agents.bitrix24 import BitrixAgentLLM
from ai_server.agents.ports import SchedulerPort
from ai_server.integrations.bitrix.bitrix_store import BitrixAgentStore
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.integrations.bitrix.ports import BitrixAgentStorePort
from ai_server.models import AgentManifest
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import Settings
from ai_server.workers.bitrix.quality_control import handle_quality_control_webhook_event

logger = logging.getLogger(__name__)


class QualityControlHandlerAdapter:
    """Adapter: implements QualityControlHandlerPort.

    Encapsulates OAuth-based actor-client resolution (was _quality_control_bitrix_client)
    and Bitrix24Specialist construction (was _build_quality_control_specialist), eliminating
    Feature Envy from BitrixWebhookProcessor.
    """

    def __init__(
        self,
        *,
        bitrix: BitrixClient,
        bitrix_oauth: BitrixOAuthService | None,
        manifests: list[AgentManifest],
        bitrix_retriever: HybridKnowledgeRetriever | None = None,
        bitrix_llm: BitrixAgentLLM | None = None,
        scheduler: SchedulerPort | None = None,
        bitrix_store: BitrixAgentStorePort | None = None,
        settings: Settings,
    ) -> None:
        self._bitrix = bitrix
        self._bitrix_oauth = bitrix_oauth
        self._manifests = manifests
        self._bitrix_retriever = bitrix_retriever
        self._bitrix_llm = bitrix_llm
        self._scheduler = scheduler
        self._bitrix_store: BitrixAgentStorePort = bitrix_store or BitrixAgentStore()
        self._settings = settings

    async def handle(self, payload: dict[str, Any], *, status: dict[str, Any]) -> dict[str, Any]:
        actor_bitrix, actor_error = await self._resolve_actor_client(status)
        if actor_error:
            return actor_error
        specialist = self._build_specialist(actor_bitrix)
        return await handle_quality_control_webhook_event(
            actor_bitrix,
            payload=payload,
            status=status,
            specialist=specialist,
            settings=self._settings,
        )

    async def _resolve_actor_client(self, status: dict[str, Any]) -> tuple[BitrixClient, dict[str, Any] | None]:
        settings = self._settings
        if not settings.quality_control_webhook_enabled:
            return self._bitrix, None
        actor_user_id = settings.quality_control_actor_user_id
        if not actor_user_id:
            if settings.quality_control_dry_run:
                return self._bitrix, None
            error: dict[str, Any] = {
                "handled": False,
                "reason": "quality_actor_not_configured",
                "message": (
                    "Для боевого фонового контроля качества нужно задать "
                    "QUALITY_CONTROL_ACTOR_USER_ID и авторизовать этого пользователя через Bitrix OAuth."
                ),
            }
            _record_error(status, error)
            return self._bitrix, error
        if settings.quality_control_dry_run:
            return self._bitrix, None
        if not settings.bitrix_oauth_enabled or self._bitrix_oauth is None:
            error = {
                "handled": False,
                "reason": "quality_actor_oauth_disabled",
                "actor_user_id": actor_user_id,
            }
            _record_error(status, error)
            return self._bitrix, error
        try:
            return await self._bitrix_oauth.client_for_user(actor_user_id), None
        except BitrixOAuthTokenMissing:
            error = {
                "handled": False,
                "reason": "quality_actor_oauth_missing",
                "actor_user_id": actor_user_id,
                "message": (
                    "Для фонового контроля качества нужно один раз авторизовать локальное "
                    "приложение под служебным пользователем AI-помощника."
                ),
            }
            _record_error(status, error)
            return self._bitrix, error

    def _build_specialist(self, actor_bitrix: BitrixClient) -> Any:
        from ai_server.agents.bitrix24 import Bitrix24Specialist

        bitrix24_manifest = next((m for m in self._manifests if m.id == "bitrix24"), None)
        if bitrix24_manifest is None:
            return None

        settings = self._settings
        return Bitrix24Specialist.build(
            bitrix24_manifest,
            bitrix_client=self._bitrix,
            actor_client=actor_bitrix,
            auto_execute=not settings.quality_control_dry_run,
            bitrix_retriever=self._bitrix_retriever,
            bitrix_llm=self._bitrix_llm,
            scheduler=self._scheduler,
            bitrix_store=self._bitrix_store,
            bitrix_bot=self._bitrix,
            settings=settings,
        )


def _record_error(status: dict[str, Any], error: dict[str, Any]) -> None:
    status["last_error"] = error.get("message") or error.get("reason")
    status["last_reason"] = error.get("reason")
    status["errors"] = int(status.get("errors") or 0) + 1
