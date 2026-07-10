"""Tests for FeedbackReceiverAdapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import anyio

from ai_server.models import AgentTask, UserContext
from ai_server.workers.diagnost.feedback_receiver import FeedbackReceiverAdapter, _detect_rating


def _run(coro):
    async def _r():
        return await coro

    return anyio.run(_r)


def _task(text: str = "👍", user_id: str = "42") -> AgentTask:
    return AgentTask(
        task_id="t-1",
        request=text,
        user=UserContext(id=user_id, display_name="Тест"),
    )


def _store(pending: dict | None = None) -> AsyncMock:
    store = AsyncMock()
    store.get_pending_feedback_for_user = AsyncMock(return_value=pending)
    store.save_feedback = AsyncMock()
    store.mark_pending_received = AsyncMock()
    return store


# ── _detect_rating ─────────────────────────────────────────────────────────────


def test_detect_rating_thumbs_up():
    assert _detect_rating("👍")[0] == 10


def test_detect_rating_thumbs_down():
    assert _detect_rating("👎")[0] == 1


def test_detect_rating_digit_10():
    assert _detect_rating("10")[0] == 10


def test_detect_rating_digit_6():
    assert _detect_rating("6")[0] == 6


def test_detect_rating_fraction_10():
    assert _detect_rating("10/10")[0] == 10


def test_detect_rating_digit_1():
    assert _detect_rating("1")[0] == 1


def test_detect_rating_with_explanation():
    rating, raw_text = _detect_rating("7 — ответ неполный")
    assert rating == 7
    assert raw_text == "7 — ответ неполный"


def test_detect_rating_fraction_with_explanation():
    assert _detect_rating("8/10, помогло")[0] == 8


def test_detect_rating_unknown_text():
    assert _detect_rating("привет")[0] is None


def test_detect_rating_long_sentence():
    assert _detect_rating("расскажи про отчёт за неделю")[0] is None


# ── FeedbackReceiverAdapter.handle ────────────────────────────────────────────


def test_handle_returns_false_when_no_user_id():
    store = _store()
    adapter = FeedbackReceiverAdapter(store)
    task = AgentTask(task_id="t-1", request="👍")  # no user
    result = _run(adapter.handle(task))
    assert result is False
    store.get_pending_feedback_for_user.assert_not_called()


def test_handle_returns_false_when_not_a_rating():
    store = _store(pending={"id": 1, "event_id": "ev-1", "dialog_key": "dk"})
    adapter = FeedbackReceiverAdapter(store)
    task = _task(text="что делать с этой ошибкой?")
    result = _run(adapter.handle(task))
    assert result is False
    store.save_feedback.assert_not_called()


def test_handle_returns_false_when_no_pending_for_user():
    store = _store(pending=None)
    adapter = FeedbackReceiverAdapter(store)
    task = _task(text="👍")
    result = _run(adapter.handle(task))
    assert result is False
    store.save_feedback.assert_not_called()


def test_handle_returns_true_and_saves_feedback():
    pending = {"id": 7, "event_id": "ev-42", "dialog_key": "dk-1"}
    store = _store(pending=pending)
    adapter = FeedbackReceiverAdapter(store)
    task = _task(text="👍", user_id="99")
    result = _run(adapter.handle(task))
    assert result is True
    store.save_feedback.assert_awaited_once_with("ev-42", "99", rating=10, raw_text="👍", dialog_key="dk-1")
    store.mark_pending_received.assert_awaited_once_with(7)


def test_handle_returns_true_for_numeric_rating():
    pending = {"id": 3, "event_id": "ev-1", "dialog_key": "dk"}
    store = _store(pending=pending)
    adapter = FeedbackReceiverAdapter(store)
    task = _task(text="3", user_id="55")
    result = _run(adapter.handle(task))
    assert result is True
    call_kwargs = store.save_feedback.call_args.kwargs
    assert call_kwargs["rating"] == 3
