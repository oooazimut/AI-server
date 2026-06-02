from .models import AgentManifest


KEYWORD_CAPABILITY_HINTS = {
    "битрикс": "bitrix24",
    "задач": "bitrix24",
    "камера": "ip_camera_diagnostics",
    "регистратор": "recorder_diagnostics",
    "роутер": "network_equipment",
    "1с": "accounting",
    "excel": "excel_analysis",
    "пто": "technical_documentation",
    "чертеж": "cad_drafting",
    "код": "software_development",
}


def suggest_agents(request: str, manifests: list[AgentManifest]) -> list[AgentManifest]:
    """Small deterministic router for the prototype.

    Production routing will use policies, user rights, model routing and agent self-checks.
    """
    text = request.lower()
    wanted_capabilities = {
        capability
        for keyword, capability in KEYWORD_CAPABILITY_HINTS.items()
        if keyword in text
    }

    if not wanted_capabilities:
        return [agent for agent in manifests if agent.kind in {"orchestrator", "operator"}]

    return [
        agent
        for agent in manifests
        if wanted_capabilities.intersection(agent.capabilities)
    ]
