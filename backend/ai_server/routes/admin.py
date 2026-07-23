from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from ..knowledge import MarkdownKnowledgeBase
from ..models import AgentManifest
from ..registry import (
    get_agent_manifest,
    get_automation_manifest,
    load_automation_manifests,
    summarize_agents,
    summarize_automations,
)
from ..retrieval import HybridKnowledgeRetriever
from ..settings import get_settings
from ..skills import SkillStore
from ..specialists import manifest_by_id

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    manifests = _active_agent_manifests(request.app.state.manifests)
    settings = get_settings()
    entity_catalog = getattr(request.app.state, "orchestrator_entity_catalog", None)
    entity_snapshot = entity_catalog.snapshot() if entity_catalog is not None else {}
    return {
        "status": "ok",
        "architecture": "pro_orchestrator_with_structured_executors",
        "agent_count": len(manifests),
        "agents": [agent.id for agent in manifests],
        "bitrix_configured": settings.bitrix_configured,
        "llm_provider": settings.llm_provider,
        "orchestrator_model": settings.orchestrator_llm_model,
        "orchestrator_model_policy": "pro_only_fail_closed",
        "orchestrator_runtime_owner": "ai-server-worker",
        "llm_configured": settings.llm_configured,
        "tech_footer_enabled": settings.tech_footer_enabled,
        "tech_footer_allowed_user_ids": settings.resolved_tech_footer_allowed_user_ids,
        "deepseek_balance_configured": bool(settings.deepseek_api_key),
        "diagnost_enabled": settings.diagnost_enabled,
        "diagnost_trace_snapshot_enabled": settings.diagnost_trace_snapshot_enabled,
        "conversation_trace_enabled": settings.conversation_trace_enabled,
        "learning_events_enabled": settings.learning_events_enabled,
        "learning_events_path": str(settings.learning_events_path),
        "bitrix_webhook_queue_enabled": settings.webhook_event_queue_enabled,
        "bitrix_webhook_worker_enabled": settings.webhook_event_worker_enabled,
        "agent_orchestrator_worker_count": settings.agent_orchestrator_worker_count,
        "agent_bitrix_worker_count": settings.agent_bitrix_worker_count,
        "agent_task_timeout_seconds": settings.agent_task_timeout_seconds,
        "orchestrator_entity_catalog_status": entity_snapshot.get("status", "worker_owned"),
        "orchestrator_entity_catalog_version": entity_snapshot.get("version"),
        "orchestrator_entity_catalog_counts": {
            key: len(entity_snapshot.get(key) or []) for key in ("users", "projects", "warehouses")
        },
        "bitrix_dialog_guard_enabled": settings.bitrix_dialog_guard_enabled,
        "bitrix_dialog_stuck_seconds": settings.bitrix_dialog_stuck_seconds,
        "bitrix_dialog_pending_ttl_seconds": settings.bitrix_dialog_pending_ttl_seconds,
        "bitrix_search_indexer_enabled": settings.search_background_periodic_enabled,
        "bitrix_search_metadata_enabled": settings.search_background_periodic_metadata_enabled,
        "bitrix_search_content_enabled": (
            settings.search_content_enabled and settings.search_background_periodic_content_enabled
        ),
        "bitrix_search_delta_enabled": (
            settings.search_delta_indexer_enabled and settings.search_background_periodic_delta_enabled
        ),
        "bitrix_task_close_control_worker_enabled": settings.bitrix_task_close_control_worker_enabled,
        "bitrix_reconciler_enabled": settings.reconcile_enabled,
        "logistics_vehicle_usage_enabled": settings.vehicle_usage_enabled,
    }


@router.get("/agents")
def agents(request: Request) -> Any:
    return summarize_agents(_active_agent_manifests(request.app.state.manifests))


@router.get("/agents/{agent_id}")
def agent_detail(agent_id: str, request: Request) -> Any:
    manifest = manifest_by_id(_active_agent_manifests(request.app.state.manifests), agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return manifest


@router.get("/agents/{agent_id}/skills")
def agent_skills(agent_id: str, request: Request) -> Any:
    manifest = manifest_by_id(_active_agent_manifests(request.app.state.manifests), agent_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail="agent not found")
    return SkillStore().list_skills(manifest)


@router.get("/agents/{agent_id}/knowledge/topics")
def agent_knowledge_topics(agent_id: str, request: Request) -> Any:
    manifest = manifest_by_id(_active_agent_manifests(request.app.state.manifests), agent_id)
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
    manifest = manifest_by_id(_active_agent_manifests(request.app.state.manifests), agent_id)
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


def _active_agent_manifests(manifests: list[AgentManifest]) -> list[AgentManifest]:
    return [manifest for manifest in manifests if manifest.kind in {"orchestrator", "specialist"}]
