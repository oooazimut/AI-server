from pathlib import Path

import yaml

from .models import AgentAutomationManifest, AgentManifest, AgentSummary, AutomationSummary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_PACKAGE_DIR = PROJECT_ROOT / "agents"


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
            reasoning_mode=agent.reasoning_mode,
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
    return paths


def load_automation_manifests(agent_id: str | None = None) -> list[AgentAutomationManifest]:
    automations: list[AgentAutomationManifest] = []
    for agent in load_agent_manifests():
        if agent_id is not None and agent.id != agent_id:
            continue
        for automation in agent.automations:
            automations.append(_with_owner(automation, agent.id))
    return automations


def get_automation_manifest(automation_id: str) -> AgentAutomationManifest | None:
    for automation in load_automation_manifests():
        if automation.id == automation_id:
            return automation
    return None


def summarize_automations(automations: list[AgentAutomationManifest]) -> list[AutomationSummary]:
    return [
        AutomationSummary(
            id=automation.id,
            name=automation.name,
            kind=automation.kind,
            trigger=automation.trigger,
            owner_agent_id=automation.owner_agent_id,
            enabled_by_default=automation.enabled_by_default,
            uses_llm=automation.uses_llm,
            requires_oauth_actor=automation.requires_oauth_actor,
            human_approval_required=automation.human_approval_required,
        )
        for automation in automations
    ]


def _with_owner(automation: AgentAutomationManifest, agent_id: str) -> AgentAutomationManifest:
    if automation.owner_agent_id == agent_id:
        return automation
    return automation.model_copy(update={"owner_agent_id": agent_id})


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Agent manifest must be a mapping: {path}")
    if payload.get("kind") in {"orchestrator", "specialist"} and "reasoning_mode" not in payload:
        raise ValueError(f"Agent manifest must declare reasoning_mode explicitly: {path}")
    return payload
