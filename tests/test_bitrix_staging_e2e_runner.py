from __future__ import annotations

from pathlib import Path

from scripts.bitrix_staging_e2e_runner import (
    TestCase as RunnerTestCase,
)
from scripts.bitrix_staging_e2e_runner import (
    acquire_dialog_lock,
    cleanup_tests_after_failure,
    default_lock_path,
    evaluate_response_text,
    event_processed,
    make_run_id,
    matching_response_messages,
    queue_is_idle,
    release_dialog_lock,
)
from scripts.bitrix_staging_e2e_runner import (
    tests_for_suite as runner_tests_for_suite,
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


def test_matching_response_messages_uses_expect_all_fragments() -> None:
    messages = [
        {"id": 1, "text": "Черновик задачи: подготовить тестовый отчет"},
        {"id": 2, "text": "Черновик задачи: подготовить тестовый отчет. Срок: 13.07.2026. Подтвердите создание."},
    ]
    test = RunnerTestCase(test_id="draft", text="", expect_all=("черновик", "срок", "подтверд"))

    selected = matching_response_messages(messages, test)

    assert [item["id"] for item in selected] == [1, 2]


def test_evaluate_response_text_requires_all_expected_fragments() -> None:
    test = RunnerTestCase(
        test_id="draft",
        text="",
        expect_all=("черновик", "срок", "подтверд"),
        reject_any=("Срок: Без срока",),
    )

    missing = evaluate_response_text("Черновик задачи. Подтвердите создание.", test)
    rejected = evaluate_response_text("Черновик задачи. Срок: Без срока. Подтвердите создание.", test)
    ok = evaluate_response_text("Черновик задачи. Срок: 13.07.2026. Подтвердите создание.", test)

    assert not missing["matched"]
    assert rejected["matched"]
    assert rejected["rejected"]
    assert ok["matched"]
    assert not ok["rejected"]


def test_queue_is_idle_requires_no_pending_or_processing_events() -> None:
    assert queue_is_idle({"queue": {"pending": 0, "processing": 0, "failed": 3}})
    assert not queue_is_idle({"queue": {"pending": 1, "processing": 0}})
    assert not queue_is_idle({"queue": {"pending": 0, "processing": 1}})


def test_make_run_id_is_unique_and_keeps_readable_prefix() -> None:
    first = make_run_id()
    second = make_run_id()

    assert first.startswith("AI-TEST-")
    assert second.startswith("AI-TEST-")
    assert first != second


def test_default_lock_path_is_scoped_to_dialog() -> None:
    assert default_lock_path("chat4321").endswith("ai-server-bitrix-e2e-chat4321.lock")


def test_acquire_dialog_lock_blocks_parallel_runner() -> None:
    lock_dir = Path(__file__).resolve().parents[1] / ".tmp_e2e_lock_tests"
    lock_dir.mkdir(exist_ok=True)
    lock_path = str(lock_dir / "dialog.lock")
    first = acquire_dialog_lock(lock_path, timeout_seconds=0.1, stale_seconds=60, poll_interval=0.01)
    second = acquire_dialog_lock(lock_path, timeout_seconds=0.1, stale_seconds=60, poll_interval=0.01)

    try:
        assert first["ok"]
        assert not second["ok"]
        assert second["error"] == "dialog_lock_timeout"
    finally:
        release_dialog_lock(first)

    third = acquire_dialog_lock(lock_path, timeout_seconds=0.1, stale_seconds=60, poll_interval=0.01)
    try:
        assert third["ok"]
    finally:
        release_dialog_lock(third)
        lock_dir.rmdir()


def test_event_processed_uses_current_event_status() -> None:
    status = {"latest_events": [{"id": 41, "status": "processing"}, {"id": 42, "status": "done"}]}

    assert event_processed(status, 42)
    assert not event_processed(status, 41)


def test_event_processed_falls_back_to_worker_last_event_when_queue_idle() -> None:
    status = {"queue": {"pending": 0, "processing": 0}, "worker": {"last_event_id": 77}}

    assert event_processed(status, 77)
    assert not event_processed(status, 78)


def test_tests_for_suite_all_stays_read_only_by_default() -> None:
    tests = runner_tests_for_suite("all", include_draft=False)

    assert tests
    assert all(test.kind in {"read", "smoke"} for test in tests)


def test_tests_for_suite_quick_uses_small_read_only_subset() -> None:
    tests = runner_tests_for_suite("quick", include_draft=False)

    assert [test.test_id for test in tests] == ["BITRIX-SMOKE-01", "BITRIX-PROJECT-HYPHEN-01"]
    assert all(test.kind in {"read", "smoke"} for test in tests)


def test_tests_for_suite_tasks_advanced_is_optional_read_only() -> None:
    tests = runner_tests_for_suite("tasks_advanced", include_draft=False)

    assert [test.test_id for test in tests] == ["BITRIX-TASK-ADVANCED-COMMENT-01"]
    assert tests[0].kind == "read"
    assert "задач" in tests[0].expect_all
    assert "понаблюдать" in tests[0].expect_any


def test_tests_for_suite_drafts_adds_cleanup_steps() -> None:
    tests = runner_tests_for_suite("drafts", include_draft=False)

    assert [test.test_id for test in tests] == [
        "BITRIX-TASK-DRAFT-01",
        "BITRIX-TASK-DRAFT-DISCARD-01",
        "BITRIX-CALENDAR-DRAFT-01",
        "BITRIX-CALENDAR-DRAFT-DISCARD-01",
    ]
    assert tests[0].reject_any
    assert tests[1].kind == "draft_cleanup"


def test_tests_for_suite_project_draft_requires_project_in_preview_and_cleanup() -> None:
    tests = runner_tests_for_suite("drafts_project", include_draft=False)

    assert [test.test_id for test in tests] == [
        "BITRIX-TASK-PROJECT-DRAFT-01",
        "BITRIX-TASK-PROJECT-DRAFT-DISCARD-01",
    ]
    assert "проект" in tests[0].expect_all
    assert "Ларгус" in tests[0].expect_all
    assert tests[1].kind == "draft_cleanup"


def test_cleanup_tests_after_failure_returns_next_cleanup_block_only() -> None:
    tests = [
        RunnerTestCase(test_id="task-draft", text="", kind="draft"),
        RunnerTestCase(test_id="task-cleanup", text="", kind="draft_cleanup"),
        RunnerTestCase(test_id="calendar-draft", text="", kind="draft"),
        RunnerTestCase(test_id="calendar-cleanup", text="", kind="draft_cleanup"),
    ]

    cleanup = cleanup_tests_after_failure(tests, 0)

    assert [test.test_id for test in cleanup] == ["task-cleanup"]


def test_cleanup_tests_after_failure_ignores_read_failures() -> None:
    tests = [
        RunnerTestCase(test_id="read", text="", kind="read"),
        RunnerTestCase(test_id="cleanup", text="", kind="draft_cleanup"),
    ]

    assert cleanup_tests_after_failure(tests, 0) == []
