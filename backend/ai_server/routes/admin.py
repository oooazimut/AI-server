from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from ..knowledge import MarkdownKnowledgeBase
from ..models import AgentManifest
from ..registry import get_agent_manifest, summarize_agents
from ..retrieval import HybridKnowledgeRetriever
from ..settings import get_settings
from ..skills import SkillStore
from ..specialists import manifest_by_id
from ..workers.registry import get_automation_manifest, load_automation_manifests, summarize_automations

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    manifests: list[AgentManifest] = request.app.state.manifests
    settings = get_settings()
    return {
        "status": "ok",
        "architecture": "orchestrator_plus_modular_specialists",
        "agent_count": len(manifests),
        "agents": [agent.id for agent in manifests],
        "bitrix_configured": settings.bitrix_configured,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "llm_configured": settings.llm_configured,
        "tech_footer_enabled": settings.tech_footer_enabled,
        "tech_footer_allowed_user_ids": settings.resolved_tech_footer_allowed_user_ids,
        "deepseek_balance_configured": bool(settings.deepseek_api_key),
        "learning_events_enabled": settings.learning_events_enabled,
        "learning_events_path": str(settings.learning_events_path),
        "bitrix_webhook_queue_enabled": settings.webhook_event_queue_enabled,
        "bitrix_webhook_worker_enabled": settings.webhook_event_worker_enabled,
        "bitrix_search_indexer_enabled": settings.search_background_indexer_enabled,
        "bitrix_quality_control_enabled": settings.quality_control_webhook_enabled,
        "bitrix_quality_control_dry_run": settings.quality_control_dry_run,
        "bitrix_task_supervisor_enabled": settings.supervisor_enabled,
        "bitrix_reconciler_enabled": settings.reconcile_enabled,
        "logistics_vehicle_usage_enabled": settings.vehicle_usage_enabled,
    }


@router.get("/agents")
def agents(request: Request) -> Any:
    return summarize_agents(request.app.state.manifests)


@router.get("/agents/{agent_id}")
def agent_detail(agent_id: str, request: Request) -> Any:
    manifest = manifest_by_id(request.app.state.manifests, agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return manifest


@router.get("/agents/{agent_id}/skills")
def agent_skills(agent_id: str, request: Request) -> Any:
    manifest = manifest_by_id(request.app.state.manifests, agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return SkillStore().list_skills(manifest)


@router.get("/agents/{agent_id}/knowledge/topics")
def agent_knowledge_topics(agent_id: str, request: Request) -> Any:
    manifest = manifest_by_id(request.app.state.manifests, agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return MarkdownKnowledgeBase().list_topics(manifest)


@router.get("/agents/{agent_id}/knowledge/search")
def agent_knowledge_search(
    agent_id: str,
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=5, ge=1, le=20),
    topic: str | None = None,
) -> Any:
    manifest = manifest_by_id(request.app.state.manifests, agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return HybridKnowledgeRetriever().search(manifest, q, limit=limit, topic=topic)


@router.get("/agents/{agent_id}/automations")
def agent_automations(agent_id: str, request: Request) -> Any:
    manifest = manifest_by_id(request.app.state.manifests, agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return summarize_automations(load_automation_manifests(agent_id=agent_id))


@router.get("/automations")
def automations(agent_id: str | None = None) -> Any:
    if agent_id is not None and get_agent_manifest(agent_id) is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return summarize_automations(load_automation_manifests(agent_id=agent_id))


@router.get("/automations/{automation_id}")
def automation_detail(automation_id: str) -> Any:
    automation = get_automation_manifest(automation_id)
    if automation is None:
        raise HTTPException(status_code=404, detail="automation not found")
    return automation


@router.get("/agent/tools")
def legacy_agent_tools(request: Request) -> dict[str, Any]:
    manifests: list[AgentManifest] = request.app.state.manifests
    return {
        "tools": [
            {"agent_id": agent.id, "tools": agent.tools, "capabilities": agent.capabilities} for agent in manifests
        ]
    }


@router.post("/agent/documents/compare")
def legacy_documents_compare() -> dict[str, Any]:
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Document comparison now belongs to the PTO LLM specialist; use the orchestrator/chat flow.",
    )
