from ai_server.capability_registry import build_capability_registry, registry_tool, validate_tool_arguments
from ai_server.models import AgentManifest
from ai_server.registry import get_agent_manifest


def _manifest() -> AgentManifest:
    return AgentManifest(id="bitrix24", name="Bitrix", kind="specialist", description="test")


def test_registry_hash_is_stable_and_changes_with_tool_contract():
    first_tool = {
        "name": "search",
        "description": "Search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
    registry_1 = build_capability_registry(_manifest(), [first_tool], structured_tool_names={"search"})
    registry_2 = build_capability_registry(_manifest(), [first_tool], structured_tool_names={"search"})
    changed_tool = {
        **first_tool,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    }
    registry_3 = build_capability_registry(_manifest(), [changed_tool], structured_tool_names={"search"})

    assert registry_1["registry_version"] == registry_2["registry_version"]
    assert registry_tool(registry_1, "search")["structured_command"] is True
    assert registry_1["registry_version"] != registry_3["registry_version"]
    assert registry_tool(registry_1, "search")["version"] != registry_tool(registry_3, "search")["version"]


def test_argument_validation_fails_closed_for_required_type_range_enum_and_unknown():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            "mode": {"type": "string", "enum": ["brief", "full"]},
        },
        "required": ["query", "limit"],
    }

    assert validate_tool_arguments(schema, {"query": "Borisov", "limit": 10, "mode": "brief"}) == []
    errors = validate_tool_arguments(
        schema,
        {"limit": 0, "mode": "unexpected", "extra": True},
    )

    assert "arguments.query: required" in errors
    assert "arguments.limit: below minimum 1" in errors
    assert "arguments.mode: value is not in enum" in errors
    assert "arguments.extra: unknown argument" in errors
    assert validate_tool_arguments(schema, {"query": "x", "limit": True}) == ["arguments.limit: expected integer"]


def test_live_bitrix_registry_contains_search_matrix_and_orchestrator_contract():
    manifest = get_agent_manifest("bitrix24")
    registry = build_capability_registry(manifest, [], structured_tool_names=set())
    contracts = {item["id"]: item["content"] for item in registry["contracts"]}
    skills = {item["id"]: item["content"] for item in registry["skills"]}

    matrix = contracts["search_intents"]
    assert {"найди", "покажи", "поищи", "выведи", "ищи"} <= set(matrix["verbs"]["search"])
    assert matrix["warehouse"]["warehouse_card"]["product_limit"] == 10
    assert matrix["warehouse"]["list_warehouses"]["include_products"] is False
    assert "orchestrator_command_contract" in skills
    assert "structured_command" in skills["orchestrator_command_contract"]
    assert matrix["warehouse"]["list_warehouses"]["list_all"] is True
    assert "search_product is not a tool" in matrix["rules"][0]["notes"][-1]
