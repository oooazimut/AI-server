from __future__ import annotations

import importlib
from typing import Any, Protocol

from ai_server.models import AgentManifest, AgentResult, AgentTask


class Specialist(Protocol):
    async def handle(self, task: AgentTask) -> AgentResult: ...


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
