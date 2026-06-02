from .models import AgentManifest


KEYWORD_CAPABILITY_HINTS = {
    "битрикс": "bitrix24",
    "bitrix": "bitrix24",
    "задач": "crm_tasks",
    "заявк": "ticket_creation",
    "документ": "document_search",
    "файл": "document_search",
    "диск": "document_search",
    "смет": "document_search",
    "проект": "projects_crm",
    "групп": "projects_crm",
    "crm": "projects_crm",
    "сделк": "projects_crm",
    "лид": "projects_crm",
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
    text = request.casefold()
    wanted_capabilities = {
        capability
        for keyword, capability in KEYWORD_CAPABILITY_HINTS.items()
        if keyword in text
    }

    if not wanted_capabilities:
        return [agent for agent in manifests if agent.kind in {"orchestrator", "operator"}]

    return [agent for agent in manifests if wanted_capabilities.intersection(agent.capabilities)]
