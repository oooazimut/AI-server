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
    # channel infrastructure — passed through to Bitrix24Specialist and InternalOrchestrator.build()
    manifests: Any = None  # list[AgentManifest] — needed by InternalOrchestrator.build()
    bitrix_client: Any = None  # BitrixClient (HTTP REST)
    portal_search_index: Any = None  # PortalSearchIndex
    bitrix_oauth: Any = None  # BitrixOAuthService | None — for OAuth-based Bitrix writes
    bitrix_bot: Any = None  # BitrixBotPort; defaults to bitrix_client in InternalOrchestrator.build()
    # orchestrator
    scheduler: Any = None  # SchedulerPort | None
    orchestrator_llm: Any = None
    orchestrator_store: Any = None  # AgentDialogStorePort | None
    orchestrator_retriever: Any = None  # HybridKnowledgeRetriever | None
    # bitrix24 specialist
    bitrix_llm: Any = None
    bitrix_retriever: Any = None
    bitrix_store: Any = None
    # pto specialist
    pto_llm: Any = None
    pto_store: Any = None  # AgentDialogStorePort | None
    # logistics specialist
    vehicle_usage_store: Any = None  # VehicleUsageStorePort | None
    logistics_llm: Any = None
    logistics_vu_settings: Any = None  # VehicleUsageSettings | None
    # channel delivery + telemetry (captured by InternalOrchestrator.build, not passed to specialists)
    channels: Any = None  # dict[str, ChannelPort]
    footer_service: Any = None  # TechnicalFooterService | None
    learning_recorder: Any = None  # LearningEventRecorder | None
    trace_recorder: Any = None  # TraceRecorder | None

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
        registry[manifest.id] = cls.build(manifest, **deps)
    return registry


def manifest_by_id(manifests: list[AgentManifest], agent_id: str) -> AgentManifest | None:
    return next((m for m in manifests if m.id == agent_id), None)


def _load_entrypoint(entrypoint: str) -> Any:
    module_path, _, class_name = entrypoint.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
