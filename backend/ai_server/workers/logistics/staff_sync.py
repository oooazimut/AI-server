from __future__ import annotations

import asyncio
import json
import logging

import redis.asyncio as aioredis

from ai_server.tools.vehicle_usage import StaffMember, VehicleUsageStorePort
from ai_server.utils import optional_int

logger = logging.getLogger(__name__)

_ROSTER_KEY = "ai_server:staff_roster:pending"


async def run_staff_sync(store: VehicleUsageStorePort, redis_url: str) -> None:
    while True:
        try:
            await _apply_pending_roster(store, redis_url)
        except Exception:
            logger.exception("staff_sync: error applying pending roster")
        await asyncio.sleep(3600)


async def _apply_pending_roster(store: VehicleUsageStorePort, redis_url: str) -> None:
    r = aioredis.from_url(redis_url)
    async with r:
        raw = await r.get(_ROSTER_KEY)
        if not raw:
            return
        members_data: list[dict] = json.loads(raw)
        members = [
            StaffMember(
                order=int(d["display_order"]),
                name=str(d["full_name"]),
                user_id=optional_int(d.get("user_id")),
            )
            for d in members_data
        ]
        store.upsert_employees(members)
        await r.delete(_ROSTER_KEY)
    logger.info("staff_sync: applied %d employees from Redis roster", len(members))
