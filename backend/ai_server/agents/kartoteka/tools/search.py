from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_server.models import ToolDefinition, ToolResult, ToolStatus

if TYPE_CHECKING:
    from ai_server.integrations.postgres.kartoteka_agent import PostgresKartotekaStore


class KartotekaSearchTool:
    name = "kartoteka_search"

    def __init__(self, store: PostgresKartotekaStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="kartoteka_search",
            description=(
                "Ищет в локальном каталоге организации по ключевым словам. "
                "Возвращает путь, имя файла и сниппет из текста документа."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "limit": {
                        "type": "integer",
                        "description": "Максимальное количество результатов (1–20)",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if self._store is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="KartotekaStore не настроен")
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="query обязателен")
        limit = min(max(int(args.get("limit") or 5), 1), 20)
        results = await self._store.search(query, user_id=user_id, limit=limit)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={"results": results, "count": len(results)})


class KartotekaContextTool:
    name = "kartoteka_context"

    def __init__(self, store: PostgresKartotekaStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="kartoteka_context",
            description="Возвращает статистику локального файлового каталога: число документов и чанков.",
            parameters={"type": "object", "properties": {}},
        )

    async def execute(
        self,
        args: dict[str, Any],
        *,
        user_id: int | None = None,
        dialog_key: str | None = None,
        dialog_id: str | None = None,
    ) -> ToolResult:
        if self._store is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="KartotekaStore не настроен")
        stats = await self._store.stats()
        return ToolResult(status=ToolStatus.OK, tool=self.name, data=stats)
