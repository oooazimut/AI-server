import asyncio
import json

import pytest

from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.models import AgentManifest, AgentResult, AgentTask, ModelUsageRecord, ToolDefinition, ToolResult
from ai_server.orchestrators.plan_authoritative import (
    PLAN_SCHEMA,
    PlanAuthoritativeOrchestrator,
    PlanRejected,
    _constraints,
    _decode_plan,
    _hash,
)
from ai_server.orchestrators.tools.call_specialist import CallSpecialistTool
from ai_server.registry import get_agent_manifest
from tests.fakes import FakeBitrixLLM


def _structured_catalog():
    return {
        "bitrix24": {
            "capabilities": ["bitrix_api"],
            "registry_version": "registry-v1",
            "tools": [
                {
                    "id": "bitrix_api",
                    "version": "tool-v1",
                    "description": "Exact Bitrix method call",
                    "structured_command": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string"},
                            "params": {"type": "object"},
                        },
                        "required": ["method", "params"],
                    },
                }
            ],
        }
    }


def _raw_plan(request: str, command, *, max_rounds: int = 1):
    return json.dumps(
        {
            "schema_version": PLAN_SCHEMA,
            "plan_id": "p1",
            "request_hash": _hash(request),
            "state": "EXECUTE",
            "clarification": None,
            "max_rounds": max_rounds,
            "subtasks": [
                {
                    "subtask_id": "s1",
                    "segment_id": None,
                    "specialist_id": "bitrix24",
                    "capability": "bitrix_api",
                    "request": request,
                    "structured_command": command,
                }
            ],
        },
        ensure_ascii=False,
    )


def test_plan_accepts_exact_versioned_command_and_requires_it_for_structured_tool():
    request = "Find employee Borisov"
    catalog = _structured_catalog()
    constraints = _constraints(request, catalog)
    command = {
        "registry_version": "registry-v1",
        "tool_name": "bitrix_api",
        "arguments": {"method": "user.search", "params": {"NAME": "Borisov"}},
    }

    plan = _decode_plan(_raw_plan(request, command), plan_id="p1", request=request, constraints=constraints)

    assert plan.subtasks[0].structured_command.tool_name == "bitrix_api"
    assert plan.subtasks[0].structured_command.arguments == command["arguments"]

    without_command = json.loads(_raw_plan(request, command))
    without_command["subtasks"][0].pop("structured_command")
    with pytest.raises(PlanRejected, match="STRUCTURED_COMMAND_REQUIRED"):
        _decode_plan(
            json.dumps(without_command),
            plan_id="p1",
            request=request,
            constraints=constraints,
        )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"registry_version": "stale"}, "CAPABILITY_REGISTRY_VERSION_MISMATCH"),
        ({"tool_name": "unknown"}, "STRUCTURED_COMMAND_TOOL_INVALID"),
        ({"arguments": {"method": "user.search", "params": {}, "extra": True}}, "STRUCTURED_COMMAND_ARGUMENTS_INVALID"),
    ],
)
def test_plan_rejects_stale_wrong_or_invalid_structured_command(change, reason):
    request = "Find employee Borisov"
    command = {
        "registry_version": "registry-v1",
        "tool_name": "bitrix_api",
        "arguments": {"method": "user.search", "params": {}},
        **change,
    }
    with pytest.raises(PlanRejected, match=reason):
        _decode_plan(
            _raw_plan(request, command),
            plan_id="p1",
            request=request,
            constraints=_constraints(request, _structured_catalog()),
        )


class _RecordingWarehouseTool:
    name = "bitrix_warehouse_search"

    def __init__(self):
        self.calls = []

    def definition(self):
        return ToolDefinition(
            name=self.name,
            description="Warehouse search",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "include_products": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["query", "include_products", "limit"],
            },
        )

    async def execute(self, args, **kwargs):
        self.calls.append(dict(args))
        return ToolResult(
            status="ok",
            tool=self.name,
            data={"query": args["query"], "matches": [{"id": 1, "title": "Borisov", "address": "A"}]},
        )


def _direct_specialist(tool, llm):
    return Bitrix24Specialist(
        get_agent_manifest("bitrix24"),
        agent_tools=[tool],
        llm=llm,
    )


def test_specialist_executes_exact_arguments_once_without_bitrix_llm_reinterpretation():
    tool = _RecordingWarehouseTool()
    llm = FakeBitrixLLM()
    specialist = _direct_specialist(tool, llm)
    registry = specialist.capability_registry()
    arguments = {"query": "Borisov", "include_products": False, "limit": 7}

    result = asyncio.run(
        specialist.execute_structured_command(
            AgentTask(task_id="t1", request="show warehouse Borisov"),
            {
                "registry_version": registry["registry_version"],
                "tool_name": tool.name,
                "arguments": arguments,
            },
        )
    )

    assert result.status == "completed"
    assert tool.calls == [arguments]
    assert llm.decide_calls == []
    assert llm.compose_calls == []
    assert result.metadata["structured_command"] is True
    assert result.metadata["command_arguments"] == arguments


def test_specialist_rejects_stale_registry_before_tool_or_llm_call():
    tool = _RecordingWarehouseTool()
    llm = FakeBitrixLLM()
    specialist = _direct_specialist(tool, llm)

    result = asyncio.run(
        specialist.execute_structured_command(
            AgentTask(task_id="t1", request="show warehouse Borisov"),
            {
                "registry_version": "stale",
                "tool_name": tool.name,
                "arguments": {"query": "Borisov", "include_products": True, "limit": 10},
            },
        )
    )

    assert result.status == "failed"
    assert result.metadata["reason"] == "CAPABILITY_REGISTRY_VERSION_MISMATCH"
    assert tool.calls == []
    assert llm.decide_calls == []
    assert llm.compose_calls == []


class _StructuredSpecialist:
    def __init__(self):
        self.commands = []
        self.registry = _structured_catalog()["bitrix24"]
        self.registry.update(
            {
                "schema_version": "specialist.capabilities.v1",
                "specialist_id": "bitrix24",
                "specialist_version": "1",
                "skills": [],
                "contracts": [],
                "allowed_actions": [],
                "approval_required": [],
            }
        )

    def capability_registry(self):
        return self.registry

    async def handle(self, task):  # pragma: no cover - direct path must bypass it
        raise AssertionError("legacy Bitrix LLM path was called")

    async def execute_structured_command(self, task, command):
        self.commands.append(command)
        method = command["arguments"]["method"]
        payload = [{"ID": 17, "NAME": "Borisov"}] if method == "user.search" else [{"ID": 101}]
        tool_result = ToolResult(status="ok", tool="bitrix_api", data={"method": method, "result": payload})
        return AgentResult(
            status="completed",
            agent_id="bitrix24",
            answer="executed",
            confidence=1.0,
            metadata={
                "terminal": True,
                "answer_is_final": True,
                "safe_to_send": True,
                "structured_command": True,
                "registry_version": self.registry["registry_version"],
                "command_arguments": command["arguments"],
                "tool_result": tool_result.model_dump(),
            },
        )


class _TwoRoundPlanner:
    def __init__(self):
        self.plan_calls = []

    async def plan(self, *, task, constraints, **kwargs):
        self.plan_calls.append(task)
        history = task.context.get("orchestrator_execution_history") or []
        if not history:
            arguments = {"method": "user.search", "params": {"NAME": "Borisov"}}
            max_rounds = 2
        else:
            assert history[-1]["result"]["data"]["result"][0]["ID"] == 17
            arguments = {"method": "tasks.task.list", "params": {"filter": {"RESPONSIBLE_ID": 17}}}
            max_rounds = 1
        raw = {
            "schema_version": PLAN_SCHEMA,
            "plan_id": constraints["plan_id"],
            "request_hash": constraints["request_hash"],
            "state": "EXECUTE",
            "clarification": None,
            "max_rounds": max_rounds,
            "subtasks": [
                {
                    "subtask_id": f"s{len(self.plan_calls)}",
                    "segment_id": None,
                    "specialist_id": "bitrix24",
                    "capability": "bitrix_api",
                    "request": task.request,
                    "structured_command": {
                        "registry_version": "registry-v1",
                        "tool_name": "bitrix_api",
                        "arguments": arguments,
                    },
                }
            ],
        }
        return json.dumps(raw), ModelUsageRecord(agent_id="planner", provider="test", model="test")

    async def finalize(self, **kwargs):  # pragma: no cover - terminal facts are rendered deterministically
        raise AssertionError("final model must not be called")


def test_orchestrator_can_use_first_result_for_one_bounded_followup_command():
    specialist = _StructuredSpecialist()
    manifest = AgentManifest(id="bitrix24", name="Bitrix", kind="specialist", description="test")
    manifest.capabilities = ["bitrix_api"]
    call = CallSpecialistTool({"bitrix24": specialist}, [manifest])
    planner = _TwoRoundPlanner()
    orchestrator = PlanAuthoritativeOrchestrator(
        AgentManifest(id="internal_orchestrator", name="Orchestrator", kind="orchestrator", description="test"),
        agent_tools=[call],
        planner=planner,
        llm=planner,
    )

    result = asyncio.run(orchestrator.handle(AgentTask(task_id="t1", request="Find Borisov tasks")))

    assert result.status == "completed"
    assert [item["arguments"]["method"] for item in specialist.commands] == ["user.search", "tasks.task.list"]
    assert len(planner.plan_calls) == 2
    assert result.metadata["structured_command_rounds"] == 2
