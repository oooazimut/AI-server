"""Global test fixtures — make TestClient(app) work without real Postgres or Redis."""

from __future__ import annotations

from unittest.mock import patch

import fakeredis
import fakeredis.aioredis
import pytest


@pytest.fixture(autouse=True)
def fake_infra(monkeypatch):
    """Patch startup infrastructure so tests run without real Postgres or Redis.

    - Sets DATABASE_URL / REDIS_URL so lifespan() validation passes.
    - Replaces redis.asyncio.from_url with in-memory fakeredis (per-test isolated).
    - Patches ensure_schema on all Postgres stores to async no-ops.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://fake/fake")
    monkeypatch.setenv("REDIS_URL", "redis://localhost/15")

    server = fakeredis.FakeServer()

    def _fake_from_url(url=None, *args, **kwargs):
        decode = kwargs.get("decode_responses", False) or kwargs.get("encoding") == "utf-8"
        return fakeredis.aioredis.FakeRedis(server=server, decode_responses=bool(decode))

    async def _noop(self):
        pass

    with (
        patch("redis.asyncio.from_url", _fake_from_url),
        patch(
            "ai_server.integrations.postgres.bitrix_agent.PostgresBitrixAgentStore.ensure_schema",
            _noop,
        ),
        patch(
            "ai_server.integrations.postgres.pto_agent.PostgresPtoAgentStore.ensure_schema",
            _noop,
        ),
        patch(
            "ai_server.integrations.postgres.orchestrator_agent.PostgresOrchestratorStore.ensure_schema",
            _noop,
        ),
        patch(
            "ai_server.integrations.postgres.vehicle_usage.PostgresVehicleUsageStore.ensure_schema",
            _noop,
        ),
    ):
        yield
