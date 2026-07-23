from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import redis.asyncio as aioredis

from ai_server.utils import MOSCOW_TZ

_KEY = "ai_server:orchestrator:entity_catalog:health:v1"


class RedisOrchestratorCatalogHealth:
    """Cross-process health view for the worker-owned entity catalog.

    Only readiness metadata is shared. The catalog contents and all semantic
    resolution remain private to the orchestrator worker.
    """

    def __init__(self, redis_url: str, *, ttl_seconds: int) -> None:
        self._ttl_seconds = max(120, int(ttl_seconds))
        self._client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)

    async def publish(self, catalog_snapshot: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": "orchestrator.entity_catalog.health.v1",
            "status": str(catalog_snapshot.get("status") or "error"),
            "version": catalog_snapshot.get("version"),
            "updated_at": catalog_snapshot.get("updated_at"),
            "published_at": datetime.now(MOSCOW_TZ).isoformat(),
            "counts": {
                key: len(catalog_snapshot.get(key) or [])
                for key in ("users", "projects", "warehouses")
            },
        }
        await self._client.set(
            _KEY,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ex=self._ttl_seconds,
        )
        return payload

    async def read(self) -> dict[str, Any]:
        try:
            raw = await self._client.get(_KEY)
        except Exception:
            return {
                **self.missing_snapshot(),
                "status": "error",
            }
        if not raw:
            return self.missing_snapshot()
        try:
            payload = json.loads(raw)
            counts = payload.get("counts")
            if not isinstance(payload, dict) or not isinstance(counts, dict):
                raise ValueError("invalid payload")
            return {
                "schema_version": "orchestrator.entity_catalog.health.v1",
                "status": str(payload.get("status") or "error"),
                "version": payload.get("version"),
                "updated_at": payload.get("updated_at"),
                "published_at": payload.get("published_at"),
                "counts": {
                    key: max(0, int(counts.get(key) or 0))
                    for key in ("users", "projects", "warehouses")
                },
            }
        except (TypeError, ValueError, json.JSONDecodeError):
            return {
                **self.missing_snapshot(),
                "status": "error",
            }

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def missing_snapshot() -> dict[str, Any]:
        return {
            "schema_version": "orchestrator.entity_catalog.health.v1",
            "status": "missing",
            "version": None,
            "updated_at": None,
            "published_at": None,
            "counts": {key: 0 for key in ("users", "projects", "warehouses")},
        }


__all__ = ["RedisOrchestratorCatalogHealth"]
