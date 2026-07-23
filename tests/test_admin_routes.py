"""Tests for admin API routes (/health, /agents, /automations)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ai_server.main import app


def test_health_ok():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "agents" in data
    assert isinstance(data["agents"], list)


def test_health_includes_config_flags():
    with TestClient(app) as client:
        response = client.get("/health")
    data = response.json()
    assert "bitrix_configured" in data
    assert "llm_configured" in data
    assert "logistics_vehicle_usage_enabled" in data
    assert data["orchestrator_entity_catalog_status"] in {"missing", "ready", "stale", "error"}
    assert data["orchestrator_runtime_owner"] == "ai-server-worker"
    assert set(data["orchestrator_entity_catalog_counts"]) == {"users", "projects", "warehouses"}


def test_health_reads_worker_catalog_snapshot():
    class _CatalogHealth:
        async def read(self):
            return {
                "status": "ready",
                "version": "worker-catalog-v1",
                "published_at": "2026-07-23T12:00:00+03:00",
                "counts": {"users": 12, "projects": 4, "warehouses": 7},
            }

    with TestClient(app) as client:
        app.state.orchestrator_catalog_health = _CatalogHealth()
        response = client.get("/health")

    data = response.json()
    assert data["orchestrator_entity_catalog_status"] == "ready"
    assert data["orchestrator_entity_catalog_version"] == "worker-catalog-v1"
    assert data["orchestrator_entity_catalog_published_at"] == "2026-07-23T12:00:00+03:00"
    assert data["orchestrator_entity_catalog_counts"] == {
        "users": 12,
        "projects": 4,
        "warehouses": 7,
    }


def test_health_reports_split_search_indexer_flags(monkeypatch):
    monkeypatch.setenv("SEARCH_BACKGROUND_INDEXER_ENABLED", "false")
    monkeypatch.setenv("SEARCH_BACKGROUND_METADATA_ENABLED", "true")
    monkeypatch.setenv("SEARCH_BACKGROUND_CONTENT_ENABLED", "true")
    monkeypatch.setenv("SEARCH_BACKGROUND_DELTA_ENABLED", "true")
    monkeypatch.setenv("SEARCH_CONTENT_ENABLED", "true")
    monkeypatch.setenv("SEARCH_DELTA_INDEXER_ENABLED", "true")

    with TestClient(app) as client:
        response = client.get("/health")

    data = response.json()
    assert data["bitrix_search_indexer_enabled"] is True
    assert data["bitrix_search_metadata_enabled"] is True
    assert data["bitrix_search_content_enabled"] is True
    assert data["bitrix_search_delta_enabled"] is True


def test_agents_list():
    with TestClient(app) as client:
        response = client.get("/agents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    ids = [a["id"] for a in data]
    assert "bitrix24" in ids
    assert "internal_orchestrator" in ids
    assert "pto" not in ids
    assert "logistics" in ids


def test_agent_detail_found():
    with TestClient(app) as client:
        response = client.get("/agents/bitrix24")
    assert response.status_code == 200
    assert response.json()["id"] == "bitrix24"


def test_agent_detail_not_found():
    with TestClient(app) as client:
        response = client.get("/agents/no_such_agent")
    assert response.status_code == 404


def test_agent_skills_returns_list():
    with TestClient(app) as client:
        response = client.get("/agents/bitrix24/skills")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_agent_knowledge_topics():
    with TestClient(app) as client:
        response = client.get("/agents/bitrix24/knowledge/topics")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_agent_knowledge_search():
    pytest.importorskip("fastembed")
    with TestClient(app) as client:
        response = client.get("/agents/bitrix24/knowledge/search", params={"q": "задача"})
    assert response.status_code == 200


def test_agent_automations():
    with TestClient(app) as client:
        response = client.get("/agents/logistics/automations")
    assert response.status_code == 200


def test_automations_list():
    with TestClient(app) as client:
        response = client.get("/automations")
    assert response.status_code == 200


def test_automations_filtered_by_agent():
    with TestClient(app) as client:
        response = client.get("/automations", params={"agent_id": "logistics"})
    assert response.status_code == 200


def test_automations_unknown_agent_404():
    with TestClient(app) as client:
        response = client.get("/automations", params={"agent_id": "no_such_agent"})
    assert response.status_code == 404


def test_removed_legacy_documents_compare_returns_404():
    with TestClient(app) as client:
        response = client.post("/agent/documents/compare")
    assert response.status_code == 404


def test_removed_legacy_agent_tools_returns_404():
    with TestClient(app) as client:
        response = client.get("/agent/tools")
    assert response.status_code == 404
