"""Orchestrator-owned rendering of raw Bitrix executor results."""

from __future__ import annotations

from typing import Any

from ai_server.models import AgentResult, ToolResult
from ai_server.orchestrators.bitrix_formatter import direct_tool_results_response


def render_bitrix_tool_results(
    *,
    agent_id: str,
    tool_results: list[ToolResult],
    portal_base_url: str = "",
    command_arguments: dict[str, Any] | None = None,
) -> AgentResult:
    """Render raw executor data; structured specialists never call this layer."""

    return direct_tool_results_response(
        agent_id=agent_id,
        tool_results=tool_results,
        portal_base_url=portal_base_url,
        command_arguments=command_arguments,
    )


__all__ = ["render_bitrix_tool_results"]
