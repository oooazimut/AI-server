"""Standalone scheduler process.

Legacy no-op systemd process.

Agent-owned scheduled jobs run inside ``ai_server.agent_worker``:
morning vehicle usage requests, reminders, unknown-data finalization,
and day-off auto close. Keeping this process idle avoids a second
vehicle-usage morning trigger source while preserving the existing
systemd unit shape.

Usage:
    python -m ai_server.scheduler_worker
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    logger.info("Standalone scheduler_worker disabled; agent_worker owns scheduled agent jobs")

    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _on_signal(*_: object) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    logger.info("Scheduler worker idle; waiting for stop signal")
    await stop.wait()
    logger.info("Scheduler stopped")


if __name__ == "__main__":
    asyncio.run(main())
