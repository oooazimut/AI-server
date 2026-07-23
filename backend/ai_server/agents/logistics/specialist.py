from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from ai_server.agents.base import BaseSpecialist, _trace_now_iso
from ai_server.agents.logistics.tools import (
    VehicleCancelReportTool,
    VehicleContextTool,
    VehicleGetEmployeePeriodReportTool,
    VehicleGetOperatorsTool,
    VehicleGetReportTool,
    VehicleGetVehiclePeriodReportTool,
    VehicleReferenceTool,
    VehicleSaveDraftTool,
    VehicleSaveReportTool,
    VehicleSetOperatorsTool,
    VehicleStartDayTool,
    VehicleUpdateReportTool,
)
from ai_server.agents.ports import SchedulerPort
from ai_server.agents.tool import AgentTool
from ai_server.capability_registry import build_capability_registry, registry_tool, validate_tool_arguments
from ai_server.models import (
    ActionRecord,
    AgentManifest,
    AgentResult,
    AgentTask,
    ModelUsageRecord,
    ToolResult,
    ToolStatus,
)
from ai_server.settings import Settings, get_settings
from ai_server.tools.vehicle_usage import VehicleUsageStorePort

_STRUCTURED_TOOL_NAMES = frozenset(
    {
        "vehicle_usage_context",
        "vehicle_usage_reference",
        "vehicle_usage_get_operators",
        "vehicle_usage_set_operators",
        "vehicle_usage_start_day",
        "vehicle_usage_get_report",
        "vehicle_usage_get_employee_period_report",
        "vehicle_usage_get_vehicle_period_report",
        "vehicle_usage_save_draft",
        "vehicle_usage_save_report",
        "vehicle_usage_update_report",
        "vehicle_usage_cancel_day",
    }
)


class LogisticsSpecialist(BaseSpecialist):
    """Version-bound Logistics executor with no model or free-text reasoning."""

    action_prefix = "logistics"

    def __init__(
        self,
        manifest: AgentManifest,
        *,
        agent_tools: list[AgentTool] | None = None,
        scheduler: SchedulerPort | None = None,
        store: Any | None = None,
        settings: Settings | None = None,
        result_publisher: Any | None = None,
        conversation_trace: Any | None = None,
    ) -> None:
        self._settings = settings
        super().__init__(
            manifest,
            agent_tools=agent_tools,
            scheduler=scheduler,
            store=store,
            result_publisher=result_publisher,
            conversation_trace=conversation_trace,
        )

    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        *,
        vehicle_usage_store: VehicleUsageStorePort | None = None,
        scheduler: SchedulerPort | None = None,
        settings: Settings | None = None,
        specialist_result_publisher: Any | None = None,
        conversation_trace: Any | None = None,
        **_: object,
    ) -> LogisticsSpecialist:
        current_settings = settings or get_settings()
        allowed_user_ids = frozenset(current_settings.resolved_vehicle_usage_allowed_user_ids)
        admin_user_ids = frozenset(current_settings.resolved_vehicle_usage_admin_user_ids)
        tools: list[AgentTool] = [
            VehicleContextTool(vehicle_usage_store),
            VehicleReferenceTool(vehicle_usage_store),
            VehicleGetOperatorsTool(vehicle_usage_store),
            VehicleSetOperatorsTool(vehicle_usage_store, admin_user_ids=admin_user_ids),
            VehicleStartDayTool(vehicle_usage_store, allowed_user_ids=allowed_user_ids),
            VehicleGetReportTool(vehicle_usage_store),
            VehicleGetEmployeePeriodReportTool(vehicle_usage_store),
            VehicleGetVehiclePeriodReportTool(vehicle_usage_store),
            VehicleSaveDraftTool(vehicle_usage_store, allowed_user_ids=allowed_user_ids),
            VehicleSaveReportTool(vehicle_usage_store, allowed_user_ids=allowed_user_ids),
            VehicleUpdateReportTool(vehicle_usage_store, allowed_user_ids=allowed_user_ids),
            VehicleCancelReportTool(vehicle_usage_store, allowed_user_ids=allowed_user_ids),
        ]
        return cls(
            manifest,
            agent_tools=tools,
            scheduler=scheduler,
            store=vehicle_usage_store,
            settings=current_settings,
            result_publisher=specialist_result_publisher,
            conversation_trace=conversation_trace,
        )

    def capability_registry(self) -> dict[str, Any]:
        definitions = self.tool_definitions()
        available = {str(item.get("name") or "") for item in definitions if isinstance(item, dict)}
        return build_capability_registry(self.manifest, definitions, structured_tool_names=available)

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            tool.definition().model_dump()
            for tool in self._tool_registry.values()
            if tool.name in _STRUCTURED_TOOL_NAMES
        ]

    async def execute_structured_command(self, task: AgentTask, command: dict[str, Any]) -> AgentResult:
        registry = self.capability_registry()
        failure = self._structured_command_failure(command, registry)
        if failure is not None:
            return failure

        tool_name = str(command["tool_name"])
        arguments = dict(command["arguments"])
        call = SimpleNamespace(name=tool_name, args=arguments, summary="orchestrator structured command")
        started_at = _trace_now_iso()
        started = time.monotonic()
        result, action, approvals = await self._execute_tool_call(call, task)
        if result is None:
            result = ToolResult(
                status=ToolStatus.ERROR,
                tool=tool_name,
                error="structured command returned no result",
            )
        await self._record_timing(
            task,
            stage="structured_tool_execute",
            started_at=started_at,
            elapsed_ms=(time.monotonic() - started) * 1000,
            status=str(result.status),
            tool=tool_name,
            details={"registry_version": registry["registry_version"]},
        )

        status = _terminal_status(result)
        actions = [
            ActionRecord(
                name="logistics_capability_contract",
                status="ok",
                details={
                    "registry_version": registry["registry_version"],
                    "tool": tool_name,
                    "mode": "structured_command",
                },
            )
        ]
        if action is not None:
            actions.append(action)
        return AgentResult(
            status="needs_human" if approvals else status,
            agent_id=self.manifest.id,
            answer="",
            actions_taken=actions,
            actions_requiring_approval=list(approvals),
            model_usage=[
                ModelUsageRecord(
                    agent_id=self.manifest.id,
                    provider="internal",
                    model="structured-logistics-executor",
                    status="not_used",
                    notes=["No Logistics model call and no specialist-side response rendering."],
                )
            ],
            confidence=1.0 if result.status == ToolStatus.OK else 0.0,
            logs=["Logistics executed one exact orchestrator command."],
            metadata={
                "terminal": True,
                "answer_is_final": False,
                "safe_to_send": True,
                "fast_return": True,
                "fast_return_reason": "orchestrator_structured_command",
                "terminal_tool": tool_name,
                "terminal_status": status,
                "registry_version": registry["registry_version"],
                "structured_command": True,
                "formatter_domain": "logistics",
                "command_arguments": arguments,
                "tool_result": result.model_dump(),
            },
        )

    def _structured_command_failure(
        self,
        command: dict[str, Any],
        registry: dict[str, Any],
    ) -> AgentResult | None:
        reason = ""
        details: dict[str, Any] = {}
        if not isinstance(command, dict) or set(command) != {"registry_version", "tool_name", "arguments"}:
            reason = "STRUCTURED_COMMAND_SCHEMA_MISMATCH"
        elif str(command.get("registry_version") or "") != str(registry.get("registry_version") or ""):
            reason = "CAPABILITY_REGISTRY_VERSION_MISMATCH"
        elif not isinstance(command.get("arguments"), dict):
            reason = "STRUCTURED_COMMAND_ARGUMENTS_INVALID"
        else:
            tool = registry_tool(registry, str(command.get("tool_name") or ""))
            if tool is None or not tool.get("structured_command"):
                reason = "STRUCTURED_COMMAND_TOOL_DISABLED"
            else:
                errors = validate_tool_arguments(dict(tool.get("parameters") or {}), command["arguments"])
                if errors:
                    reason = "STRUCTURED_COMMAND_ARGUMENTS_INVALID"
                    details["validation_errors"] = errors
        if not reason:
            return None
        return AgentResult(
            status="failed",
            agent_id=self.manifest.id,
            answer="Logistics отклонил несовместимую команду оркестратора.",
            actions_taken=[
                ActionRecord(
                    name="logistics_capability_contract",
                    status="rejected",
                    details={"reason": reason, **details},
                )
            ],
            confidence=0.0,
            metadata={
                "terminal": True,
                "answer_is_final": True,
                "safe_to_send": True,
                "fast_return": True,
                "fast_return_reason": "structured_command_rejected",
                "reason": reason,
                "registry_version": registry.get("registry_version"),
            },
        )

    async def handle(self, task: AgentTask) -> AgentResult:
        return AgentResult(
            status="failed",
            agent_id=self.manifest.id,
            answer="Logistics принимает только точную структурированную команду оркестратора.",
            model_usage=[
                ModelUsageRecord(
                    agent_id=self.manifest.id,
                    provider="internal",
                    model="logistics-executor-guard",
                    status="not_used",
                    notes=["All Logistics free-text reasoning paths are disabled."],
                )
            ],
            actions_taken=[
                ActionRecord(
                    name="logistics_executor_guard",
                    status="rejected",
                    details={"reason": "ORCHESTRATOR_STRUCTURED_COMMAND_REQUIRED"},
                )
            ],
            metadata={"reason": "ORCHESTRATOR_STRUCTURED_COMMAND_REQUIRED"},
        )

    def _logs(self) -> list[str]:
        return ["Logistics is a structured executor with no model surface."]

    def _llm_failure_result(self, message: str) -> Any:
        raise RuntimeError(message)


def _terminal_status(result: ToolResult) -> str:
    if result.status in {ToolStatus.OK, ToolStatus.NOT_FOUND, ToolStatus.DRY_RUN}:
        if bool(result.data.get("needs_clarification")):
            return "needs_clarification"
        return "completed"
    if result.status in {ToolStatus.INVALID_TOOL_CALL, ToolStatus.AMBIGUOUS, ToolStatus.DENIED}:
        return "needs_clarification"
    return "failed"


__all__ = ["LogisticsSpecialist"]
