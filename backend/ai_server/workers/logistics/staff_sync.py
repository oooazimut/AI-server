from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.tools.vehicle_usage import VehicleUsageStore, fetch_staff_roster
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)


async def run_staff_sync(bitrix: BitrixClient, store: VehicleUsageStore) -> None:
    while True:
        await _sleep_until_midnight_moscow()
        try:
            roster = await fetch_staff_roster(bitrix)
            filtered = [m for m in roster if "AI" not in m.name.upper()]
            store.upsert_employees(filtered)
            logger.info(
                "staff_sync: upserted %d employees (filtered %d AI accounts)",
                len(filtered),
                len(roster) - len(filtered),
            )
        except Exception:
            logger.exception("staff_sync: failed to sync employees from Bitrix")


async def _sleep_until_midnight_moscow() -> None:
    now = datetime.now(MOSCOW_TZ)
    tomorrow_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    sleep_seconds = (tomorrow_midnight - now).total_seconds()
    await asyncio.sleep(max(sleep_seconds, 1))
