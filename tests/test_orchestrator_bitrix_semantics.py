from datetime import datetime

import pytest

from ai_server.models import AgentTask
from ai_server.orchestrators.bitrix_semantics import SemanticPolicyViolation, normalize_plan
from ai_server.orchestrators.plan_authoritative import Plan, StructuredCommand, Subtask
from ai_server.utils import MOSCOW_TZ


def _plan(request: str, tool: str, args: dict) -> tuple[Plan, AgentTask, dict]:
    command = StructuredCommand("CURRENT", tool, args)
    plan = Plan("p1", "EXECUTE", None, [Subtask("s1", None, "bitrix24", tool, request, command)])
    task = AgentTask(task_id="t1", request=request)
    catalog = {"bitrix24": {"tools": [{"id": tool, "parameters": {"type": "object"}, "structured_command": True}]}}
    return plan, task, {"capability_catalog": catalog}


def _args(request: str, tool: str, args: dict, *, now: datetime | None = None) -> dict:
    plan, task, constraints = _plan(request, tool, args)
    normalized = normalize_plan(plan, task=task, constraints=constraints, now=now)
    return normalized.subtasks[0].structured_command.arguments


def test_warehouse_search_verbs_have_the_same_contents_semantics():
    shown = _args("Покажи склад Борисова", "bitrix_warehouse_search", {"query": "Борисова"})
    found = _args("Найди склад Борисова", "bitrix_warehouse_search", {"query": "Борисова"})
    listed = _args("Выведи склад Борисова", "bitrix_warehouse_search", {"query": "Борисова"})

    for result in (shown, found, listed):
        assert result["include_products"] is True
        assert result["product_limit"] == 50


def test_product_on_warehouse_is_canonical_and_not_left_to_specialist_reasoning():
    result = _args(
        "Найди амортизатор на складе Борисова",
        "bitrix_warehouse_search",
        {"query": "амортизатор"},
    )

    assert result["query"] == "борисова"
    assert result["product_query"] == "амортизатор"
    assert result["include_products"] is True
    assert result["product_limit"] == 50


def test_default_task_template_is_current_user_and_three_working_days_at_1900():
    now = datetime(2026, 7, 24, 9, 0, tzinfo=MOSCOW_TZ)  # Friday
    result = _args("Создай задачу проверить договор", "task_create_draft", {"title": "Проверить договор"}, now=now)

    assert result["responsible_self"] is True
    assert result["deadline_iso"] == "2026-07-29T19:00:00+03:00"


def test_default_calendar_template_is_three_calendar_days_at_noon():
    now = datetime(2026, 7, 24, 9, 0, tzinfo=MOSCOW_TZ)
    result = _args("Напомни позвонить Борисову", "calendar_event_draft", {"title": "Позвонить Борисову"}, now=now)

    assert result["start_iso"] == "2026-07-27T12:00:00+03:00"
    assert result["end_iso"] == "2026-07-27T12:30:00+03:00"


def test_explicit_calendar_values_are_never_overwritten_by_defaults():
    result = _args(
        "Напомни завтра в 16:00 позвонить Борисову",
        "calendar_event_draft",
        {"title": "Позвонить Борисову", "start_iso": "2026-07-23T16:00:00+03:00"},
    )

    assert result["start_iso"] == "2026-07-23T16:00:00+03:00"
    assert "end_iso" not in result


def test_semantically_wrong_tool_is_rejected_before_dispatch():
    plan, task, constraints = _plan("Покажи склад Борисова", "portal_search", {"query": "Борисова"})

    with pytest.raises(SemanticPolicyViolation, match="SEMANTIC_TOOL_MISMATCH"):
        normalize_plan(plan, task=task, constraints=constraints)


def test_generic_tasks_are_current_users_open_tasks():
    result = _args("Покажи задачи", "bitrix_my_tasks", {})

    assert result == {"status": "open", "limit": 10}


def test_unjustified_all_tasks_scope_fails_safe_to_current_user():
    result = _args("Какие задачи сейчас активны", "bitrix_task_search", {"scope": "all"})

    assert result["scope"] == "my"
    assert result["status"] == "active"


def test_document_lookup_has_focused_scope():
    result = _args("Найди договор на диске", "portal_search", {"query": "договор"})

    assert result["scope"] == "documents"
    assert result["limit"] == 10
