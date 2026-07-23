from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from ai_server.models import AgentManifest, AgentTask, ToolStatus, UserContext
from ai_server.orchestrators.bitrix_semantics import SemanticPolicyViolation, normalize_plan
from ai_server.orchestrators.entity_catalog import (
    OrchestratorEntityCatalog,
    find_entities_in_text,
    normalize_entity_text,
)
from ai_server.orchestrators.plan_authoritative import Plan, StructuredCommand, Subtask
from ai_server.orchestrators.tools.call_specialist import CallSpecialistTool
from ai_server.utils import MOSCOW_TZ


class _DirectoryBitrix:
    async def list_all_users(self, *, limit: int):
        return [
            {"ID": 13, "NAME": "Валерий", "LAST_NAME": "Кулинич", "SECOND_NAME": "Васильевич"},
            {"ID": 21, "NAME": "Марина", "LAST_NAME": "Кулинич"},
            {"ID": 35, "NAME": "Андрей", "LAST_NAME": "Борисов"},
        ]

    async def search_projects(self, query: str, *, limit: int):
        return [
            {"ID": 71, "NAME": "Кулинич Валерий", "OPENED": "Y"},
            {"ID": 77, "NAME": "Борисов Андрей", "OPENED": "Y"},
            {"ID": 99, "NAME": "Закрытый", "OPENED": "N"},
        ]

    async def list_catalog_stores(self, *, limit: int):
        return [
            {"ID": 501, "TITLE": "Склад Борисова", "ADDRESS": "Борисов"},
            {"ID": 601, "TITLE": "Гараж", "ADDRESS": "Российская, 8"},
            {"ID": 602, "TITLE": "Гараж Смородин", "ADDRESS": "Кагальницкое шоссе"},
        ]


def _catalog() -> dict:
    service = OrchestratorEntityCatalog(_DirectoryBitrix())
    return asyncio.run(service.refresh())


def _normalize(
    request: str,
    tool: str,
    arguments: dict,
    *,
    user_id: str = "13",
    display_name: str = "Кулинич Валерий Васильевич",
    now: datetime | None = None,
) -> dict:
    command = StructuredCommand("registry", tool, arguments)
    plan = Plan("p1", "EXECUTE", None, [Subtask("s1", None, "bitrix24", tool, request, command)])
    task = AgentTask(
        task_id="t1",
        request=request,
        user=UserContext(id=user_id, display_name=display_name),
        context={"orchestrator_entity_catalog": _catalog()},
    )
    constraints = {
        "capability_catalog": {
            "bitrix24": {
                "tools": [
                    {
                        "id": tool,
                        "parameters": {"type": "object"},
                        "structured_command": True,
                    }
                ]
            }
        }
    }
    normalized = normalize_plan(plan, task=task, constraints=constraints, now=now)
    return normalized.subtasks[0].structured_command.arguments


def test_entity_catalog_normalizes_case_and_russian_surname_case():
    catalog = _catalog()

    assert normalize_entity_text("БОРИСЁВА") == "борисева"
    assert [item["id"] for item in find_entities_in_text(catalog, "users", "Покажи задачи борисова")] == [35]
    assert [item["id"] for item in find_entities_in_text(catalog, "warehouses", "склад БОРИСОВА")] == [501]
    assert [item["id"] for item in catalog["projects"]] == [77, 71]


def test_warehouse_name_becomes_one_exact_store_id():
    result = _normalize(
        "Найди амортизатор на складе борисова",
        "bitrix_warehouse_search",
        {"query": "борисова"},
    )

    assert result["store_id"] == 501
    assert result["query"] == "Склад Борисова"
    assert result["product_query"] == "амортизатор"
    assert result["include_products"] is True


def test_named_employee_task_search_becomes_one_exact_target_id():
    result = _normalize(
        "Покажи активные задачи борисова",
        "bitrix_task_search",
        {"scope": "my"},
    )

    assert result["target_user_id"] == 35
    assert result["target_user_name"] == "Борисов Андрей"
    assert result["status"] == "active"


def test_wrong_employee_id_is_rejected_before_bitrix_dispatch():
    with pytest.raises(SemanticPolicyViolation, match="ENTITY_ID_MISMATCH"):
        _normalize(
            "Покажи активные задачи Борисова",
            "bitrix_task_search",
            {"scope": "my", "target_user_id": 13},
        )


def test_task_template_for_named_employee_uses_their_open_personal_project():
    now = datetime(2026, 7, 24, 9, 0, tzinfo=MOSCOW_TZ)
    result = _normalize(
        "Создай задачу для Борисова проверить машину",
        "task_create_draft",
        {"title": "Проверить машину"},
        now=now,
    )

    assert result["responsible_id"] == 35
    assert result["responsible_name"] == "Борисов Андрей"
    assert result["group_id"] == 77
    assert result["deadline_iso"] == "2026-07-29T19:00:00+03:00"


def test_unqualified_task_template_uses_requester_and_requester_project():
    result = _normalize(
        "Создай задачу проверить договор",
        "task_create_draft",
        {"title": "Проверить договор"},
    )

    assert result["responsible_id"] == 13
    assert result["group_id"] == 71


def test_calendar_template_is_orchestrator_owned():
    now = datetime(2026, 7, 24, 9, 0, tzinfo=MOSCOW_TZ)
    result = _normalize(
        "Напомни позвонить клиенту",
        "calendar_event_draft",
        {"title": "Позвонить клиенту"},
        now=now,
    )

    assert result["start_iso"] == "2026-07-29T12:00:00+03:00"
    assert result["end_iso"] == "2026-07-29T12:30:00+03:00"
    assert result["owner_name"] == "Кулинич Валерий Васильевич"


def test_exact_warehouse_name_wins_over_longer_partial_name():
    catalog = _catalog()

    assert [item["id"] for item in find_entities_in_text(catalog, "warehouses", "Покажи склад гараж")] == [601]
    assert [
        item["id"]
        for item in find_entities_in_text(catalog, "warehouses", "Покажи склад гараж смородин")
    ] == [602]
    assert [
        item["id"]
        for item in find_entities_in_text(catalog, "warehouses", "Покажи склад гараж смородина")
    ] == [602]


def test_warehouse_address_is_not_a_semantic_alias():
    catalog = _catalog()

    assert find_entities_in_text(catalog, "warehouses", "Покажи склад Российская 8") == []


def test_full_employee_name_and_explicit_id_win_over_shared_surname():
    catalog = _catalog()

    assert [
        item["id"]
        for item in find_entities_in_text(catalog, "users", "Создай задачу на кулинич валерия")
    ] == [13]
    assert [
        item["id"]
        for item in find_entities_in_text(catalog, "users", "кулинич валерий айди 13")
    ] == [13]
    assert sorted(item["id"] for item in find_entities_in_text(catalog, "users", "кулинич")) == [13, 21]


def test_live_bitrix_specialist_cannot_be_called_without_structured_command():
    class _StructuredOnlySpecialist:
        async def handle(self, task):
            raise AssertionError("legacy semantic path must not be called")

        def capability_registry(self):
            return {
                "schema_version": "specialist.capabilities.v1",
                "registry_version": "v1",
                "tools": [
                    {
                        "id": "bitrix_warehouse_search",
                        "structured_command": True,
                        "parameters": {"type": "object"},
                    }
                ],
            }

    manifest = AgentManifest(
        id="bitrix24",
        name="Bitrix",
        kind="specialist",
        reasoning_mode="executor",
        description="executor",
        capabilities=["bitrix_warehouse_search"],
    )
    tool = CallSpecialistTool({"bitrix24": _StructuredOnlySpecialist()}, [manifest])
    result = asyncio.run(
        tool.execute_with_task(
            {"specialist_id": "bitrix24", "request": "Покажи склад"},
            task=AgentTask(task_id="t1", request="Покажи склад"),
        )
    )

    assert result.status == ToolStatus.INVALID_TOOL_CALL
    assert result.data["reason"] == "ORCHESTRATOR_STRUCTURED_COMMAND_REQUIRED"
