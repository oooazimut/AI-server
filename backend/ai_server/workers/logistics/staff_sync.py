from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.ports import VehicleUsageStorePort
from ai_server.settings import Settings
from ai_server.tools.vehicle_usage import fetch_staff_roster
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)


async def run_staff_sync(bitrix: BitrixClient, store: VehicleUsageStorePort, *, settings: Settings) -> None:
    while True:
        try:
            roster = await fetch_staff_roster(
                bitrix, exclude_user_ids=settings.resolved_vehicle_usage_excluded_user_ids
            )
            store.upsert_employees(roster)
            logger.info("staff_sync: upserted %d employees", len(roster))
        except Exception:
            logger.exception("staff_sync: failed to sync employees from Bitrix")
        await _sleep_until_midnight_moscow()


async def _sleep_until_midnight_moscow() -> None:
    now = datetime.now(MOSCOW_TZ)
    tomorrow_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    sleep_seconds = (tomorrow_midnight - now).total_seconds()
    await asyncio.sleep(max(sleep_seconds, 1))
