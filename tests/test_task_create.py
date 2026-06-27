"""Tests for BitrixTaskCreateDraft builder (build_task_create_draft_from_args)."""

from __future__ import annotations

from ai_server.agents.bitrix24.tools.task_create import (
    BitrixTaskCreateDraft,
    build_task_create_draft_from_args,
)

# ---------------------------------------------------------------------------
# BitrixTaskCreateDraft.is_ready
# ---------------------------------------------------------------------------


def test_draft_is_ready_with_fields():
    draft = BitrixTaskCreateDraft(params={"fields": {"TITLE": "Тест"}})
    assert draft.is_ready


def test_draft_not_ready_with_contract_errors():
    draft = BitrixTaskCreateDraft(params={"fields": {"TITLE": "Тест"}}, contract_errors=["missing responsible"])
    assert not draft.is_ready


def test_draft_not_ready_without_fields():
    draft = BitrixTaskCreateDraft(params={})
    assert not draft.is_ready


# ---------------------------------------------------------------------------
# build_task_create_draft_from_args — happy paths
# ---------------------------------------------------------------------------


def test_draft_complete_with_no_deadline():
    draft = build_task_create_draft_from_args(
        {"title": "Проверить камеру", "responsible_id": 9, "no_deadline": True},
        user_id=9,
    )
    assert draft.is_ready
    assert draft.params["fields"]["TITLE"] == "Проверить камеру"
    assert draft.params["fields"]["RESPONSIBLE_ID"] == 9
    assert draft.params["fields"]["NO_DEADLINE"] is True


def test_draft_strips_title_punctuation():
    draft = build_task_create_draft_from_args(
        {"title": "  Задача.  ", "responsible_id": 9, "no_deadline": True},
        user_id=9,
    )
    assert draft.params["fields"]["TITLE"] == "Задача"


def test_draft_sets_created_by_from_user():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "responsible_id": 9, "no_deadline": True},
        user_id=15,
    )
    assert draft.params["fields"]["CREATED_BY"] == 15


def test_draft_deadline_iso():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "responsible_id": 9, "deadline_iso": "2026-07-01T09:00:00+03:00"},
        user_id=9,
    )
    assert draft.is_ready
    assert draft.params["fields"]["DEADLINE"] == "2026-07-01T09:00:00+03:00"


def test_draft_deadline_z_suffix():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "responsible_id": 9, "deadline_iso": "2026-07-01T09:00:00Z"},
        user_id=9,
    )
    assert draft.is_ready


def test_draft_responsible_self():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "responsible_self": True, "no_deadline": True},
        user_id=15,
    )
    assert draft.is_ready
    assert draft.params["fields"]["RESPONSIBLE_ID"] == 15


def test_draft_description_included():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "description": "Подробное описание", "responsible_id": 9, "no_deadline": True},
        user_id=9,
    )
    assert draft.params["fields"]["DESCRIPTION"] == "Подробное описание"


# ---------------------------------------------------------------------------
# build_task_create_draft_from_args — contract errors
# ---------------------------------------------------------------------------


def test_draft_missing_title():
    draft = build_task_create_draft_from_args(
        {"title": "", "responsible_id": 9, "no_deadline": True},
        user_id=9,
    )
    assert not draft.is_ready
    assert any("title" in e for e in draft.contract_errors)


def test_draft_missing_responsible():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "no_deadline": True},
        user_id=9,
    )
    assert not draft.is_ready
    assert any("responsible" in e for e in draft.contract_errors)


def test_draft_responsible_self_without_user_id():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "responsible_self": True, "no_deadline": True},
        user_id=None,
    )
    assert not draft.is_ready


def test_draft_missing_deadline():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "responsible_id": 9},
        user_id=9,
    )
    assert not draft.is_ready
    assert any("deadline" in e for e in draft.contract_errors)


def test_draft_invalid_deadline_format():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "responsible_id": 9, "deadline_iso": "01.07.2026"},
        user_id=9,
    )
    assert not draft.is_ready
    assert any("ISO 8601" in e for e in draft.contract_errors)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_draft_summary_contains_title():
    draft = build_task_create_draft_from_args(
        {"title": "Сводка теста", "responsible_id": 9, "no_deadline": True},
        user_id=9,
    )
    assert "Сводка теста" in draft.summary


def test_draft_summary_contains_responsible_id():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "responsible_id": 9, "no_deadline": True},
        user_id=9,
    )
    assert "#9" in draft.summary


def test_draft_as_action_details_keys():
    draft = build_task_create_draft_from_args(
        {"title": "Задача", "responsible_id": 9, "no_deadline": True},
        user_id=9,
    )
    details = draft.as_action_details()
    assert "method" in details
    assert "params" in details
    assert "contract_errors" in details
    assert details["method"] == "tasks.task.add"
