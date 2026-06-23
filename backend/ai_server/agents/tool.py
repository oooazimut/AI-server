from __future__ import annotations

from typing import Any, Protocol

from ai_server.models import ToolDefinition, ToolResult


class AgentTool(Protocol):
    """A single callable tool owned by a specialist agent.

    Singleton dependencies (clients, stores) go in __init__.
    Per-request data (user_id, dialog_key, dialog_id) come in execute().
    """

    name: str

    def definition(self) -> ToolDefinition: ...

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult: ...
