from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

from ai_server.models import AgentManifest, AgentResult, AgentTask


class Specialist(Protocol):
    async def handle(self, task: AgentTask) -> AgentResult: ...


@dataclass
class SpecialistDeps:
    """Collected dependencies for all agent construction.

    To add specialist N: add its deps as a field here and populate in startup.py.
    build_specialist_registry and BitrixWebhookProcessor need no changes.
    """

    settings: Any  # Settings — typed as Any to avoid a circular import at module level
    # channel infrastructure — passed through to Bitrix24Specialist and PlanAuthoritativeOrchestrator.build()
    manifests: Any = None  # list[AgentManifest] — needed by PlanAuthoritativeOrchestrator.build()
    bitrix_client: Any = None  # BitrixClient (HTTP REST)
    portal_search_index: Any = None  # PortalSearchIndex
    bitrix_oauth: Any = None  # BitrixOAuthService | None — for OAuth-based Bitrix writes
    bitrix_bot: Any = None  # BitrixBotPort; defaults to bitrix_client in PlanAuthoritativeOrchestrator.build()
    # orchestrator
    scheduler: Any = None  # SchedulerPort | None
    orchestrator_llm: Any = None
    orchestrator_store: Any = None  # AgentDialogStorePort | None
    orchestrator_retriever: Any = None  # HybridKnowledgeRetriever | None
    orchestrator_entity_catalog: Any = None  # OrchestratorEntityCatalog | None
    task_close_report_renderer: Any = None  # Orchestrator-owned four-block report renderer
    task_close_result_text_renderer: Any = None  # Orchestrator-owned stored-result renderer
    draft_confirmation_phrase_renderer: Any = None  # Orchestrator-owned confirmation text
    draft_confirmation_matcher: Any = None  # Orchestrator-owned confirmation recognition
    # bitrix24 specialist
    bitrix_store: Any = None
    # deterministic logistics automation
    vehicle_usage_store: Any = None  # VehicleUsageStorePort | None
    logistics_vu_settings: Any = None  # VehicleUsageSettings | None
    # channel delivery (captured by PlanAuthoritativeOrchestrator.build, not passed to specialists)
    channels: Any = None  # dict[str, ChannelPort]
    footer_service: Any = None  # TechnicalFooterService | None
    result_publisher: Any = None  # ResultPublisherPort | None (orchestrator)
    specialist_result_publisher: Any = None  # ResultPublisherPort | None (specialists, not diagnost)
    conversation_trace: Any = None  # RedisConversationTrace | None
    dialog_guard: Any = None  # RedisDialogGuard | None
    outbound_queue: Any = None  # RedisOutboundQueue | None

    def as_build_kwargs(self) -> dict[str, Any]:
        """All non-None fields — pass to any agent build() method."""
        return {k: v for k, v in vars(self).items() if v is not None}


def build_specialist_registry(
    manifests: list[AgentManifest],
    *,
    audience: str | None = None,
    **deps: Any,
) -> dict[str, Specialist]:
    registry: dict[str, Specialist] = {}
    for manifest in manifests:
        if manifest.kind != "specialist" or not manifest.entrypoint:
            continue
        if audience is not None and manifest.audience != audience:
            continue
        cls = _load_entrypoint(manifest.entrypoint)
        if manifest.reasoning_mode == "executor" and not callable(
            getattr(cls, "execute_structured_command", None)
        ):
            # An executor without a structured entrypoint is not exposed to the
            # planner.  Falling back to handle(free_text) would silently restore
            # a second semantic authority inside the specialist.
            continue
        registry[manifest.id] = cls.build(manifest, **deps)
    return registry


def manifest_by_id(manifests: list[AgentManifest], agent_id: str) -> AgentManifest | None:
    return next((m for m in manifests if m.id == agent_id), None)


def _load_entrypoint(entrypoint: str) -> Any:
    module_path, _, class_name = entrypoint.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
