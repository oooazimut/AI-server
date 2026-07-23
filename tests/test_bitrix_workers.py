from __future__ import annotations

from datetime import datetime

import fakeredis.aioredis

from ai_server.integrations.redis.event_queue import RedisEventQueue
from ai_server.settings import get_settings
from ai_server.utils import MOSCOW_TZ
from ai_server.workers.bitrix.reconciler import reconcile_once
from ai_server.workers.bitrix.search_indexer import _next_scheduled_run_at


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)


def test_reconciler_enqueues_task_updates_with_dedupe(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("RECONCILE_TASKS_ENABLED", "true")
    monkeypatch.setenv("RECONCILE_DISK_DELTA_ENABLED", "false")
    settings = get_settings()
    # redis.asyncio.from_url is patched by conftest to return fakeredis;
    # override _client directly to share the same in-memory instance for this test
    queue = RedisEventQueue("redis://localhost/15")
    queue._client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    bitrix = FakeReconcileBitrix()

    first = anyio_run(reconcile_once(bitrix, queue, FakeSearchIndexer(), settings=settings))
    second = anyio_run(reconcile_once(bitrix, queue, FakeSearchIndexer(), settings=settings))

    assert first.tasks["seen"] == 1
    assert first.tasks["enqueued"] == 1
    assert second.tasks["duplicates"] == 1
    stats = anyio_run(queue.stats())
    assert stats["pending"] == 1


def test_search_indexer_weekly_metadata_schedule_uses_next_matching_slot():
    saturday = datetime(2026, 7, 11, 12, 0, tzinfo=MOSCOW_TZ)
    after_sunday_slot = datetime(2026, 7, 12, 1, 0, tzinfo=MOSCOW_TZ)

    assert _next_scheduled_run_at(saturday, time_spec="00:30", weekday_spec="sun") == datetime(
        2026,
        7,
        12,
        0,
        30,
        tzinfo=MOSCOW_TZ,
    )
    assert _next_scheduled_run_at(after_sunday_slot, time_spec="00:30", weekday_spec="sun") == datetime(
        2026,
        7,
        19,
        0,
        30,
        tzinfo=MOSCOW_TZ,
    )


class FakeReconcileBitrix:
    async def list_all_tasks(self, *, filter_=None, select=None, order=None, limit=None):
        return [
            {
                "ID": "8413",
                "TITLE": "Измененная задача",
                "STATUS": "3",
                "RESPONSIBLE_ID": "15",
                "GROUP_ID": "7",
                "CHANGED_DATE": "2026-06-05T11:30:00+03:00",
            }
        ]


class FakeSearchIndexer:
    async def run_delta_once(self):
        raise AssertionError("disk delta should be disabled in this test")
