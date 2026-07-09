from __future__ import annotations

from scripts.bitrix_staging_e2e_runner import (
    TestCase as RunnerTestCase,
)
from scripts.bitrix_staging_e2e_runner import (
    event_processed,
    matching_response_messages,
    queue_is_idle,
)


def test_matching_response_messages_skips_unmatched_delayed_messages() -> None:
    messages = [
        {"id": 1, "text": "Старый задержанный ответ по другому тесту"},
        {"id": 2, "text": "Черновик задачи: подготовить тестовый отчет MARKER"},
    ]
    test = RunnerTestCase(test_id="draft", text="", expect_any=("MARKER",))

    selected = matching_response_messages(messages, test)

    assert [item["id"] for item in selected] == [2]


def test_matching_response_messages_returns_all_when_no_expectations() -> None:
    messages = [
        {"id": 1, "text": "Первый ответ"},
        {"id": 2, "text": "Второй ответ"},
    ]
    test = RunnerTestCase(test_id="smoke", text="")

    assert matching_response_messages(messages, test) == messages


def test_queue_is_idle_requires_no_pending_or_processing_events() -> None:
    assert queue_is_idle({"queue": {"pending": 0, "processing": 0, "failed": 3}})
    assert not queue_is_idle({"queue": {"pending": 1, "processing": 0}})
    assert not queue_is_idle({"queue": {"pending": 0, "processing": 1}})


def test_event_processed_uses_current_event_status() -> None:
    status = {"latest_events": [{"id": 41, "status": "processing"}, {"id": 42, "status": "done"}]}

    assert event_processed(status, 42)
    assert not event_processed(status, 41)


def test_event_processed_falls_back_to_worker_last_event_when_queue_idle() -> None:
    status = {"queue": {"pending": 0, "processing": 0}, "worker": {"last_event_id": 77}}

    assert event_processed(status, 77)
    assert not event_processed(status, 78)
