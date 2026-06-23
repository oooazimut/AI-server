"""Tests for RedisEventQueue (mocked redis client)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from ai_server.integrations.redis.event_queue import (
    _MAX_ATTEMPTS,
    RedisEventQueue,
    _decode_json,
    _iso_from_ms,
    _iso_now,
)


def anyio_run(coro):
    import anyio

    async def _runner():
        return await coro

    return anyio.run(_runner)


def _make_queue() -> tuple[RedisEventQueue, MagicMock]:
    queue = RedisEventQueue("redis://localhost/0")
    mock_client = MagicMock()
    queue._client = mock_client
    return queue, mock_client


def _make_pipe(mock_client: MagicMock) -> AsyncMock:
    pipe = AsyncMock()
    pipe.hset = AsyncMock()
    pipe.zadd = AsyncMock()
    pipe.lpush = AsyncMock()
    pipe.ltrim = AsyncMock()
    pipe.zrem = AsyncMock()
    pipe.sadd = AsyncMock()
    pipe.execute = AsyncMock(return_value=[1, 1, 1, 1])
    mock_client.pipeline = MagicMock(return_value=pipe)
    return pipe


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_decode_json_valid():
    result = _decode_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_decode_json_invalid():
    assert _decode_json("not json") == {}


def test_decode_json_non_dict():
    assert _decode_json("[1, 2, 3]") == {}


def test_iso_now_has_timezone():
    ts = _iso_now()
    assert "+" in ts or "Z" in ts or ts.endswith("+00:00")


def test_iso_from_ms_parses():
    import time

    ms = time.time() * 1000
    ts = _iso_from_ms(ms)
    assert "T" in ts


# ---------------------------------------------------------------------------
# enqueue
# ---------------------------------------------------------------------------


def test_enqueue_new_event():
    queue, client = _make_queue()
    client.get = AsyncMock(return_value=None)
    client.incr = AsyncMock(return_value=1)
    client.set = AsyncMock(return_value=True)
    pipe = _make_pipe(client)

    event_id, is_new = anyio_run(
        queue.enqueue(
            {"event": "ONIMBOTMESSAGEADD", "data": {}},
            event_type="ONIMBOTMESSAGEADD",
            dedupe_key="unique-key-1",
        )
    )

    assert event_id == 1
    assert is_new is True
    pipe.hset.assert_called_once()
    pipe.zadd.assert_called_once()


def test_enqueue_duplicate_returns_existing():
    queue, client = _make_queue()
    client.get = AsyncMock(return_value="5")

    event_id, is_new = anyio_run(
        queue.enqueue(
            {"event": "ONIMBOTMESSAGEADD"},
            event_type="ONIMBOTMESSAGEADD",
            dedupe_key="duplicate-key",
        )
    )

    assert event_id == 5
    assert is_new is False


def test_enqueue_race_condition_nx_fails():
    """NX set fails (race) — reads back existing id."""
    queue, client = _make_queue()
    client.get = AsyncMock(side_effect=[None, "3"])  # first get: empty; second get: existing
    client.incr = AsyncMock(return_value=2)
    client.set = AsyncMock(return_value=False)  # NX failed

    event_id, is_new = anyio_run(
        queue.enqueue(
            {"event": "test"},
            event_type="test",
            dedupe_key="race-key",
        )
    )

    assert event_id == 3
    assert is_new is False


# ---------------------------------------------------------------------------
# mark_done
# ---------------------------------------------------------------------------


def test_mark_done_removes_from_processing():
    queue, client = _make_queue()
    pipe = _make_pipe(client)

    anyio_run(queue.mark_done(42, {"result": "ok"}))

    pipe.zrem.assert_called_once()
    pipe.hset.assert_called_once()
    pipe.execute.assert_called_once()


def test_mark_done_sets_status_done():
    queue, client = _make_queue()
    pipe = _make_pipe(client)

    anyio_run(queue.mark_done(99, {"detail": "processed"}))

    hset_kwargs = pipe.hset.call_args
    mapping = hset_kwargs[1].get("mapping") or hset_kwargs[0][1]
    assert mapping["status"] == "done"


# ---------------------------------------------------------------------------
# mark_failed
# ---------------------------------------------------------------------------


def test_mark_failed_reschedules_retry():
    queue, client = _make_queue()
    client.hgetall = AsyncMock(return_value={"attempts": "1"})
    pipe = _make_pipe(client)

    anyio_run(queue.mark_failed(42, "connection reset"))

    pipe.zadd.assert_called_once()  # rescheduled
    called_keys = [str(c) for c in pipe.mock_calls]
    assert not any("sadd" in c for c in called_keys)


def test_mark_failed_permanently_after_max_attempts():
    queue, client = _make_queue()
    client.hgetall = AsyncMock(return_value={"attempts": str(_MAX_ATTEMPTS)})
    pipe = _make_pipe(client)

    anyio_run(queue.mark_failed(42, "permanent error"))

    pipe.sadd.assert_called_once()  # moved to failed set


def test_mark_failed_sets_last_error():
    queue, client = _make_queue()
    client.hgetall = AsyncMock(return_value={"attempts": "2"})
    pipe = _make_pipe(client)

    anyio_run(queue.mark_failed(42, "timeout"))

    hset_kwargs = pipe.hset.call_args
    mapping = hset_kwargs[1].get("mapping") or hset_kwargs[0][1]
    assert mapping["last_error"] == "timeout"


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_returns_counts():
    queue, client = _make_queue()
    client.zcard = AsyncMock(side_effect=[3, 1])
    client.scard = AsyncMock(return_value=2)
    client.lrange = AsyncMock(return_value=[])

    result = anyio_run(queue.stats())

    assert result["backend"] == "redis"
    assert result["pending"] == 3
    assert result["processing"] == 1
    assert result["failed"] == 2
    assert result["latest"] is None


def test_stats_with_latest_event():
    queue, client = _make_queue()
    client.zcard = AsyncMock(side_effect=[0, 0])
    client.scard = AsyncMock(return_value=0)
    client.lrange = AsyncMock(return_value=["7"])
    client.hgetall = AsyncMock(
        return_value={
            "id": "7",
            "event_type": "ONIMBOTMESSAGEADD",
            "status": "done",
            "attempts": "1",
            "received_at": "2026-06-23T10:00:00",
            "processed_at": "2026-06-23T10:00:01",
            "last_error": "",
        }
    )

    result = anyio_run(queue.stats())
    assert result["latest"]["id"] == "7"
    assert result["latest"]["event_type"] == "ONIMBOTMESSAGEADD"


# ---------------------------------------------------------------------------
# claim_next
# ---------------------------------------------------------------------------


def test_claim_next_returns_none_when_empty():
    queue, client = _make_queue()
    client.zrangebyscore = AsyncMock(side_effect=[[], []])
    _make_pipe(client)

    result = anyio_run(queue.claim_next())
    assert result is None


def test_claim_next_skips_exhausted_events():
    """Events that exceed MAX_ATTEMPTS are moved to failed."""
    queue, client = _make_queue()
    client.zrangebyscore = AsyncMock(side_effect=[[], ["1"]])
    client.hgetall = AsyncMock(
        return_value={
            "attempts": str(_MAX_ATTEMPTS),
            "event_type": "test",
            "payload_json": "{}",
        }
    )
    pipe = _make_pipe(client)

    result = anyio_run(queue.claim_next())

    assert result is None
    pipe.sadd.assert_called_once()  # moved to failed
