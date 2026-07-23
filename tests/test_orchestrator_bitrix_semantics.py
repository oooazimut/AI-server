from datetime import datetime

import pytest

from ai_server.models import AgentTask, UserContext
from ai_server.orchestrators.bitrix_semantics import (
    SemanticPolicyViolation,
    _expected_tool,
    canonicalize_plan,
    normalize_plan,
)
from ai_server.orchestrators.plan_authoritative import Plan, StructuredCommand, Subtask
from ai_server.utils import MOSCOW_TZ


def _plan(request: str, tool: str, args: dict) -> tuple[Plan, AgentTask, dict]:
    command = StructuredCommand("CURRENT", tool, args)
    plan = Plan("p1", "EXECUTE", None, [Subtask("s1", None, "bitrix24", tool, request, command)])
    task = AgentTask(task_id="t1", request=request)
    catalog = {"bitrix24": {"tools": [{"id": tool, "parameters": {"type": "object"}, "structured_command": True}]}}
    return plan, task, {"capability_catalog": catalog}


_ENTITY_CATALOG = {
    "status": "ready",
    "users": [{"id": 22, "name": "Борисов Андрей", "aliases": ["борисов", "борисова"]}],
    "projects": [{"id": 44, "name": "Борисов Андрей", "aliases": ["борисов андрей"]}],
    "warehouses": [{"id": 7, "name": "Борисов", "aliases": ["борисов", "борисова"]}],
}


def _args(
    request: str,
    tool: str,
    args: dict,
    *,
    now: datetime | None = None,
    entity_catalog: dict | None = None,
) -> dict:
    plan, task, constraints = _plan(request, tool, args)
    normalized = normalize_plan(
        plan,
        task=task,
        constraints=constraints,
        now=now,
        entity_catalog=_ENTITY_CATALOG if entity_catalog is None else entity_catalog,
    )
    return normalized.subtasks[0].structured_command.arguments


def test_warehouse_search_verbs_have_the_same_contents_semantics():
    shown = _args("Покажи склад Борисова", "bitrix_warehouse_search", {"query": "Борисова"})
    found = _args("Найди склад Борисова", "bitrix_warehouse_search", {"query": "Борисова"})
    listed = _args("Выведи склад Борисова", "bitrix_warehouse_search", {"query": "Борисова"})
    advised = _args("Подскажи склад Борисова", "bitrix_warehouse_search", {"query": "Борисова"})

    for result in (shown, found, listed, advised):
        assert result["store_id"] == 7
        assert result["include_products"] is True
        assert result["product_limit"] == 50


def test_warehouse_address_request_is_a_direct_orchestrator_response():
    task = AgentTask(task_id="t-address", request="Покажи адрес склада Борисова")
    canonical = canonicalize_plan(
        None,
        plan_id="p-address",
        task=task,
        constraints={"capability_catalog": {"bitrix24": {"registry_version": "CURRENT"}}},
        entity_catalog=_ENTITY_CATALOG,
    )

    assert canonical.state == "DIRECT_RESPONSE"
    assert canonical.subtasks == []


def test_product_on_warehouse_is_canonical_and_not_left_to_specialist_reasoning():
    result = _args(
        "Найди амортизатор на складе Борисова",
        "bitrix_warehouse_search",
        {"query": "амортизатор"},
    )

    assert result["query"] == "Борисов"
    assert result["product_query"] == "амортизатор"
    assert result["include_products"] is True
    assert result["product_limit"] == 50


def test_default_task_template_is_current_user_and_three_working_days_at_1900():
    now = datetime(2026, 7, 24, 9, 0, tzinfo=MOSCOW_TZ)  # Friday
    result = _args("Создай задачу проверить договор", "task_create_draft", {"title": "Проверить договор"}, now=now)

    assert result["responsible_self"] is True
    assert result["deadline_iso"] == "2026-07-29T19:00:00+03:00"


def test_default_calendar_template_is_three_working_days_at_noon():
    now = datetime(2026, 7, 24, 9, 0, tzinfo=MOSCOW_TZ)
    result = _args("Напомни позвонить Борисову", "calendar_event_draft", {"title": "Позвонить Борисову"}, now=now)

    assert result["start_iso"] == "2026-07-29T12:00:00+03:00"
    assert result["end_iso"] == "2026-07-29T12:30:00+03:00"


def test_explicit_calendar_values_are_never_overwritten_by_defaults():
    result = _args(
        "Напомни завтра в 16:00 позвонить Борисову",
        "calendar_event_draft",
        {"title": "Позвонить Борисову", "start_iso": "2026-07-23T16:00:00+03:00"},
    )

    assert result["start_iso"] == "2026-07-23T16:00:00+03:00"
    assert result["end_iso"] == "2026-07-23T16:30:00+03:00"


def test_task_close_is_routed_before_generic_task_search_and_gets_full_template_defaults():
    result = _args("Закрой задачу 123", "task_close_draft", {"task_id": 123})

    assert result["close_now"] is True
    assert result["overall_status"] == "unconfirmed"
    assert len(result["missing_fields"]) == 3


def test_task_close_command_phrase_from_model_is_not_treated_as_completion_result():
    result = _args(
        "Закрой задачу 123",
        "task_close_draft",
        {
            "task_id": 123,
            "completion_summary": "Пожалуйста, закрой задачу №123 в Битрикс",
        },
    )

    assert "completion_summary" not in result
    assert result["overall_status"] == "unconfirmed"
    assert len(result["missing_fields"]) == 3


def test_task_close_model_aliases_are_normalized_by_orchestrator():
    result = _args(
        "Закрой задачу 123, работа выполнена полностью",
        "task_close_draft",
        {
            "task_id": 123,
            "completion_summary": "Работа выполнена",
            "overall_status": "готово",
            "action": "утвердить",
        },
    )

    assert result["completion_summary"] == "Работа выполнена"
    assert result["overall_status"] == "completed"
    assert result["action"] == "approve"


def test_task_close_report_incident_choice_is_normalized_by_orchestrator():
    result = _args(
        "Первый вариант",
        "task_close_report_incident",
        {"task_id": 123, "action": "первый вариант"},
    )

    assert result["action"] == "restore"


def test_semantically_wrong_tool_is_rejected_before_dispatch():
    plan, task, constraints = _plan("Покажи склад Борисова", "portal_search", {"query": "Борисова"})

    with pytest.raises(SemanticPolicyViolation, match="SEMANTIC_TOOL_MISMATCH"):
        normalize_plan(plan, task=task, constraints=constraints)


def test_generic_tasks_are_current_users_open_tasks():
    result = _args("Покажи задачи", "bitrix_my_tasks", {})

    assert result == {"status": "open", "limit": 10, "offset": 0}


def test_unjustified_all_tasks_scope_fails_safe_to_current_user():
    result = _args("Какие задачи сейчас активны", "bitrix_task_search", {"scope": "all"})

    assert result["scope"] == "my"
    assert result["status"] == "active"


def test_document_lookup_has_focused_scope():
    result = _args("Найди договор на диске", "portal_search", {"query": "договор"})

    assert result["scope"] == "documents"
    assert result["limit"] == 50


def test_plain_find_uses_global_bitrix_search_with_fifty_result_limit():
    result = _args("Найди сертификат", "portal_search", {"query": "сертификат"})

    assert result["scope"] == "all"
    assert result["query"] == "сертификат"
    assert result["limit"] == 50


def test_bitrix_marker_routes_broad_search_but_is_not_part_of_query():
    result = _args("Битрикс, найди сертификат", "portal_search", {"query": "Битрикс сертификат"})

    assert result["scope"] == "all"
    assert result["query"] == "сертификат"


def test_named_warehouse_never_falls_back_to_bitrix_when_catalog_is_unavailable():
    with pytest.raises(SemanticPolicyViolation, match="ENTITY_CATALOG_UNAVAILABLE"):
        _args(
            "Покажи склад Борисова",
            "bitrix_warehouse_search",
            {"query": "Борисова"},
            entity_catalog={"status": "error"},
        )


def test_named_task_user_must_resolve_to_exact_catalog_id():
    result = _args(
        "Покажи задачи Борисова",
        "bitrix_task_search",
        {"scope": "responsible", "target_user_name": "Борисова"},
    )

    assert result["target_user_id"] == 22
    assert result["target_user_name"] == "Борисов Андрей"


def test_task_close_control_user_is_resolved_by_orchestrator():
    result = _args(
        "Добавь Борисова в контролируемые пользователи",
        "task_close_control_update",
        {"action": "add_controlled_user", "target_user_name": "Борисова"},
    )

    assert result["operation"] == "prepare"
    assert result["target_user_id"] == 22
    assert result["target_user_name"] == "Борисов Андрей"


def test_task_defaults_replace_unnecessary_pro_clarification_without_second_model_call():
    plan = Plan("p1", "CLARIFICATION_REQUIRED", "Кто ответственный и какой срок?", [], 3)
    task = AgentTask(
        task_id="t1",
        request="Создать задачу сделать тестовый тест",
        user=UserContext(id="1", display_name="Кулинич Валерий Васильевич"),
    )
    entity_catalog = {
        "status": "ready",
        "users": [{"id": 1, "name": "Кулинич Валерий Васильевич", "aliases": ["кулинич валерий"]}],
        "projects": [{"id": 71, "name": "Кулинич Валерий", "aliases": ["кулинич валерий"]}],
        "warehouses": [],
    }
    constraints = {
        "capability_catalog": {
            "bitrix24": {
                "registry_version": "registry-v1",
                "tools": [
                    {
                        "id": "task_create_draft",
                        "parameters": {"type": "object"},
                        "structured_command": True,
                    }
                ],
            }
        }
    }

    canonical = canonicalize_plan(
        plan,
        plan_id="p1",
        task=task,
        constraints=constraints,
        entity_catalog=entity_catalog,
    )
    normalized = normalize_plan(
        canonical,
        task=task,
        constraints=constraints,
        now=datetime(2026, 7, 24, 9, 0, tzinfo=MOSCOW_TZ),
        entity_catalog=entity_catalog,
    )
    arguments = normalized.subtasks[0].structured_command.arguments

    assert normalized.state == "EXECUTE"
    assert normalized.max_rounds == 1
    assert arguments["title"] == "Сделать тестовый тест"
    assert arguments["responsible_id"] == 1
    assert arguments["group_id"] == 71
    assert arguments["deadline_iso"] == "2026-07-29T19:00:00+03:00"


def test_task_template_discards_model_only_project_and_no_deadline_values():
    now = datetime(2026, 7, 24, 9, 0, tzinfo=MOSCOW_TZ)
    task = AgentTask(
        task_id="t1",
        request="Создай задачу поменять амортизаторы на Ларгус 2",
        user=UserContext(id="1", display_name="Кулинич Валерий Васильевич"),
    )
    entity_catalog = {
        "status": "ready",
        "users": [{"id": 1, "name": "Кулинич Валерий Васильевич", "aliases": ["кулинич валерий"]}],
        "projects": [
            {"id": 71, "name": "Кулинич Валерий", "aliases": ["кулинич валерий"]},
            {"id": 99, "name": "Ларгус 2", "aliases": ["ларгус 2"]},
        ],
        "warehouses": [],
    }
    plan, _, constraints = _plan(
        task.request,
        "task_create_draft",
        {"title": "Поменять амортизаторы на Ларгус 2", "project_name": "Ларгус 2", "group_id": 99, "no_deadline": True},
    )
    result = normalize_plan(
        plan,
        task=task,
        constraints=constraints,
        now=now,
        entity_catalog=entity_catalog,
    ).subtasks[0].structured_command.arguments

    assert result["responsible_id"] == 1
    assert result["responsible_name"] == "Кулинич Валерий Васильевич"
    assert result["group_id"] == 71
    assert result["project_name"] == "Кулинич Валерий"
    assert result["deadline_iso"] == "2026-07-29T19:00:00+03:00"
    assert "no_deadline" not in result


def test_explicit_no_deadline_is_preserved_only_when_the_user_says_so():
    result = _args(
        "Создай задачу без срока проверить договор",
        "task_create_draft",
        {"title": "Проверить договор", "no_deadline": True},
    )

    assert result["no_deadline"] is True


def test_calendar_action_wins_over_warehouse_word_in_its_title():
    assert _expected_tool("Создай напоминание проверить склад") == "calendar_event_draft"


def test_orchestrator_rebuilds_composite_and_all_warehouse_plans_from_exact_catalog_ids():
    entity_catalog = {
        "status": "ready",
        "users": [],
        "projects": [],
        "warehouses": [
            {"id": 7, "name": "Борисов", "aliases": ["борисов", "борисова"]},
            {"id": 8, "name": "Карасев", "aliases": ["карасев", "карасева"]},
            {"id": 9, "name": "Гараж", "aliases": ["гараж"]},
        ],
    }
    constraints = {
        "capability_catalog": {
            "bitrix24": {
                "registry_version": "registry-v1",
                "tools": [
                    {
                        "id": "bitrix_warehouse_search",
                        "parameters": {"type": "object"},
                        "structured_command": True,
                    }
                ],
            }
        }
    }
    incomplete = Plan(
        "p1",
        "EXECUTE",
        None,
        [
            Subtask(
                "wrong",
                None,
                "bitrix24",
                "bitrix_warehouse_search",
                "Покажи склад Гараж",
                StructuredCommand("CURRENT", "bitrix_warehouse_search", {"store_id": 9}),
            )
        ],
        3,
    )
    task = AgentTask(task_id="t1", request="Покажи склад Борисова Карасева и Гараж")

    canonical = canonicalize_plan(
        incomplete,
        plan_id="p1",
        task=task,
        constraints=constraints,
        entity_catalog=entity_catalog,
    )
    normalized = normalize_plan(
        canonical,
        task=task,
        constraints=constraints,
        entity_catalog=entity_catalog,
    )

    assert [item.structured_command.arguments["store_id"] for item in normalized.subtasks] == [7, 8, 9]
    assert all(item.structured_command.arguments["product_limit"] == 10 for item in normalized.subtasks)
    assert normalized.max_rounds == 1

    all_task = AgentTask(task_id="t2", request="Найди амортизатор на складах и покажи")
    all_plan = canonicalize_plan(
        incomplete,
        plan_id="p2",
        task=all_task,
        constraints=constraints,
        entity_catalog=entity_catalog,
    )
    normalized_all = normalize_plan(
        all_plan,
        task=all_task,
        constraints=constraints,
        entity_catalog=entity_catalog,
    )

    assert [item.structured_command.arguments["store_id"] for item in normalized_all.subtasks] == [7, 8, 9]
    assert all(
        item.structured_command.arguments["product_query"] == "амортизатор"
        for item in normalized_all.subtasks
    )
    assert all(item.structured_command.arguments["product_limit"] == 10 for item in normalized_all.subtasks)


def test_canonical_warehouse_plan_keeps_reverse_order_product_after_exact_store_name():
    entity_catalog = {
        "status": "ready",
        "users": [],
        "projects": [],
        "warehouses": [
            {"id": 9, "name": "Гараж", "aliases": ["гараж"]},
            {"id": 33, "name": "Гараж Смородин", "aliases": ["гараж смородин"]},
        ],
    }
    constraints = {
        "capability_catalog": {
            "bitrix24": {
                "registry_version": "registry-v1",
                "tools": [{"id": "bitrix_warehouse_search", "parameters": {"type": "object"}, "structured_command": True}],
            }
        }
    }
    task = AgentTask(task_id="t1", request="Найди на складе Гараж Смородина втулки")
    canonical = canonicalize_plan(None, plan_id="p1", task=task, constraints=constraints, entity_catalog=entity_catalog)
    normalized = normalize_plan(canonical, task=task, constraints=constraints, entity_catalog=entity_catalog)
    arguments = normalized.subtasks[0].structured_command.arguments

    assert arguments["store_id"] == 33
    assert arguments["query"] == "Гараж Смородин"
    assert arguments["product_query"] == "втулки"
