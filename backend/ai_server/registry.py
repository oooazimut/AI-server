from pathlib import Path

import yaml

from .models import AgentManifest, AgentSummary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_PACKAGE_DIR = PROJECT_ROOT / "agents"
LEGACY_AGENT_CONFIG_DIR = PROJECT_ROOT / "config" / "agents"


def load_agent_manifests() -> list[AgentManifest]:
    manifests: list[AgentManifest] = []
    seen: set[str] = set()

    for path in _manifest_paths():
        payload = _read_yaml(path)
        manifest = AgentManifest.model_validate(payload)
        if manifest.id in seen:
            continue
        manifests.append(manifest)
        seen.add(manifest.id)

    return manifests


def get_agent_manifest(agent_id: str) -> AgentManifest | None:
    for manifest in load_agent_manifests():
        if manifest.id == agent_id:
            return manifest
    return None


def summarize_agents(manifests: list[AgentManifest]) -> list[AgentSummary]:
    return [
        AgentSummary(
            id=agent.id,
            name=agent.name,
            kind=agent.kind,
            capabilities=agent.capabilities,
            tools=agent.tools,
            automations=[automation.id for automation in agent.automations],
            handoff_description=agent.handoff_description,
        )
        for agent in manifests
    ]


def resolve_project_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def agent_package_path(agent_id: str) -> Path:
    return AGENT_PACKAGE_DIR / agent_id


def _manifest_paths() -> list[Path]:
    paths: list[Path] = []
    if AGENT_PACKAGE_DIR.exists():
        paths.extend(sorted(AGENT_PACKAGE_DIR.glob("*/manifest.yaml")))
    if LEGACY_AGENT_CONFIG_DIR.exists():
        paths.extend(sorted(LEGACY_AGENT_CONFIG_DIR.glob("*.yaml")))
    return paths


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Agent manifest must be a mapping: {path}")
    return payload
