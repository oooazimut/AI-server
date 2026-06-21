"""Standalone scheduler process.

Run as a separate systemd unit so APScheduler cron jobs fire exactly once
even when uvicorn runs with --workers N.

Usage:
    python -m ai_server.scheduler_worker
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ai_server.settings import get_settings
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)


async def _enqueue_vehicle_usage_trigger(redis_url: str, request_time: str, reminder_count: int = 0) -> None:
    from datetime import datetime

    from ai_server.integrations.redis.event_queue import RedisEventQueue
    from ai_server.utils import MOSCOW_TZ as tz

    queue = RedisEventQueue(redis_url)
    today = datetime.now(tz).date().isoformat()
    event_data: dict = {
        "event": "vehicle_usage_morning_trigger",
        "data": {"reminder_count": reminder_count, "scheduled_date": today},
    }
    dedupe_key = f"vehicle_usage_morning:{today}:reminder_{reminder_count}"
    event_id, inserted = await queue.enqueue(
        event_data,
        event_type="vehicle_usage_morning_trigger",
        dedupe_key=dedupe_key,
    )
    await queue.close()
    if inserted:
        logger.info("Enqueued vehicle_usage_morning_trigger id=%s date=%s reminder=%s", event_id, today, reminder_count)
    else:
        logger.debug("vehicle_usage_morning_trigger already queued (id=%s, dedupe)", event_id)


def _parse_hhmm(value: str) -> tuple[int, int]:
    parts = value.split(":")
    return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    settings = get_settings()

    if not settings.redis_url:
        logger.error("REDIS_URL is not set — scheduler_worker requires Redis. Exiting.")
        sys.exit(1)

    if not settings.vehicle_usage_enabled:
        logger.info("VEHICLE_USAGE_ENABLED=false — nothing to schedule. Sleeping forever.")
        await asyncio.sleep(float("inf"))
        return

    redis_url = settings.redis_url
    request_time = settings.vehicle_usage_request_time
    hour, minute = _parse_hhmm(request_time)

    scheduler = AsyncIOScheduler(timezone=MOSCOW_TZ)

    async def _trigger() -> None:
        await _enqueue_vehicle_usage_trigger(redis_url, request_time, reminder_count=0)

    scheduler.add_job(
        _trigger,
        CronTrigger(hour=hour, minute=minute, timezone=MOSCOW_TZ),
        id="vehicle_usage_morning",
        replace_existing=True,
        misfire_grace_time=300,
    )
    scheduler.start()
    logger.info("Scheduler started — vehicle_usage_morning cron at %s МСК", request_time)

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _on_signal(*_: object) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    await stop.wait()
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped")


if __name__ == "__main__":
    asyncio.run(main())
