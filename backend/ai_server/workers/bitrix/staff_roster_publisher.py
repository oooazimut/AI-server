"""Fetches the employee roster from Bitrix and publishes it to Redis.

Run as a weekly cron via AgentScheduler. The logistics staff_sync worker
reads the published data and updates the logistics store, keeping the two
domains decoupled: Bitrix fetching happens here, store writing happens there.
"""

from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis

from ai_server.settings import Settings
from ai_server.tools.bitrix_ports import BitrixUserPort
from ai_server.tools.vehicle_usage import StaffMember
from ai_server.utils import optional_int

logger = logging.getLogger(__name__)

_ROSTER_KEY = "ai_server:staff_roster:pending"
_ROSTER_TTL = 7 * 24 * 3600  # 7 days


async def publish_staff_roster(bitrix: BitrixUserPort, redis_url: str, *, settings: Settings) -> None:
    """Fetch active employees from Bitrix and write their roster to Redis."""
    try:
        members = await fetch_staff_roster(
            bitrix,
            exclude_user_ids=settings.resolved_vehicle_usage_excluded_user_ids,
        )
        payload = json.dumps([m.as_dict() for m in members])
        r = aioredis.from_url(redis_url)
        async with r:
            await r.set(_ROSTER_KEY, payload, ex=_ROSTER_TTL)
        logger.info("staff_roster_publisher: published %d employees", len(members))
    except Exception:
        logger.exception("staff_roster_publisher: failed to publish roster")


async def fetch_staff_roster(
    client: BitrixUserPort,
    *,
    exclude_user_ids: set[int] | None = None,
) -> list[StaffMember]:
    users = await client.list_all_users(
        filter_={"ACTIVE": True, "USER_TYPE": "employee"},
        select=["ID", "NAME", "LAST_NAME"],
    )
    excluded = exclude_user_ids or set()
    candidates: list[tuple[str, int, str]] = []
    for user in users:
        user_id = optional_int(user.get("ID") or user.get("id"))
        if user_id is None or user_id in excluded:
            continue
        first = str(user.get("NAME") or "").strip()
        last = str(user.get("LAST_NAME") or "").strip()
        name = f"{last} {first}".strip() if last else first
        if not name:
            continue
        candidates.append((last.casefold() or first.casefold(), user_id, name))
    candidates.sort(key=lambda x: x[0])
    return [StaffMember(order=i + 1, user_id=uid, name=name) for i, (_, uid, name) in enumerate(candidates)]
