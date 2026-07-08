from __future__ import annotations

from scripts.bitrix_staging_e2e_runner import TestCase as RunnerTestCase
from scripts.bitrix_staging_e2e_runner import matching_response_messages


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
