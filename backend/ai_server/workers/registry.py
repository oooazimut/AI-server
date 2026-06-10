from ai_server.models import AgentAutomationManifest, AutomationSummary
from ai_server.registry import load_agent_manifests


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
