from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Protocol

from ai_server.models import AgentManifest, AgentResult, AgentTask


class Specialist(Protocol):
    async def handle(self, task: AgentTask) -> AgentResult: ...


@dataclass
class SpecialistDeps:
    """Collected dependencies for specialist construction.

    To add specialist N: add its deps as a field here and populate in startup.py.
    build_specialist_registry, _build_orchestrator, and BitrixWebhookProcessor need no changes.
    """

    settings: Any  # Settings — typed as Any to avoid a circular import at module level
    scheduler: Any = None  # SchedulerPort | None
    orchestrator_llm: Any = None
    # bitrix24
    bitrix_llm: Any = None
    bitrix_retriever: Any = None
    bitrix_store: Any = None
    # pto
    pto_llm: Any = None
    pto_store: Any = None  # AgentDialogStorePort | None
    # logistics
    vehicle_usage_store: Any = None  # VehicleUsageStorePort | None
    logistics_llm: Any = None
    logistics_vu_settings: Any = None  # VehicleUsageSettings | None

    orchestrator_store: Any = None  # AgentDialogStorePort | None
    orchestrator_retriever: Any = None  # HybridKnowledgeRetriever | None

    def as_registry_kwargs(self) -> dict[str, Any]:
        """Returns kwargs for build_specialist_registry.

        Excludes orchestrator_llm, orchestrator_store, orchestrator_retriever
        (passed directly to InternalOrchestrator) and None values.
        """
        excluded = {"orchestrator_llm", "orchestrator_store", "orchestrator_retriever"}
        return {k: v for k, v in vars(self).items() if k not in excluded and v is not None}


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
