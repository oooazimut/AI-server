from __future__ import annotations

import asyncio

from ai_server.integrations.redis.orchestrator_catalog_health import RedisOrchestratorCatalogHealth


def test_catalog_health_round_trip_omits_catalog_contents():
    async def _run():
        health = RedisOrchestratorCatalogHealth("redis://localhost/15", ttl_seconds=300)
        try:
            published = await health.publish(
                {
                    "status": "ready",
                    "version": "catalog-v1",
                    "updated_at": "2026-07-23T12:00:00+03:00",
                    "users": [{"id": 1, "name": "Private user data"}],
                    "projects": [{"id": 2, "name": "Private project data"}],
                    "warehouses": [{"id": 3, "name": "Private warehouse data"}],
                }
            )
            restored = await health.read()
        finally:
            await health.close()
        return published, restored

    published, restored = asyncio.run(_run())
    assert restored["status"] == "ready"
    assert restored["version"] == "catalog-v1"
    assert restored["counts"] == {"users": 1, "projects": 1, "warehouses": 1}
    assert published == restored
    assert "users" not in restored
    assert "projects" not in restored
    assert "warehouses" not in restored


def test_catalog_health_reports_missing_without_worker_publication():
    async def _run():
        health = RedisOrchestratorCatalogHealth("redis://localhost/15", ttl_seconds=300)
        try:
            return await health.read()
        finally:
            await health.close()

    snapshot = asyncio.run(_run())
    assert snapshot["status"] == "missing"
    assert snapshot["counts"] == {"users": 0, "projects": 0, "warehouses": 0}
