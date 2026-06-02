from pathlib import Path

import yaml

from .models import AgentManifest, AgentSummary


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGENT_CONFIG_DIR = PROJECT_ROOT / "config" / "agents"


def load_agent_manifests(config_dir: Path | None = None) -> list[AgentManifest]:
    directory = config_dir or DEFAULT_AGENT_CONFIG_DIR
    manifests: list[AgentManifest] = []

    for path in sorted(directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
        manifests.append(AgentManifest.model_validate(payload))

    return manifests


def summarize_agents(manifests: list[AgentManifest]) -> list[AgentSummary]:
    return [
        AgentSummary(
            id=agent.id,
            name=agent.name,
            kind=agent.kind,
            capabilities=agent.capabilities,
            tools=agent.tools,
        )
        for agent in manifests
    ]
