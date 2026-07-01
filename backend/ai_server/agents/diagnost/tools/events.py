from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_server.models import ToolDefinition, ToolResult, ToolStatus

if TYPE_CHECKING:
    from ai_server.integrations.postgres.diagnost_agent import PostgresDiagnostStore


class SearchEventsTool:
    name = "diagnost_search_events"

    def __init__(self, store: PostgresDiagnostStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Ищет диалоги оркестратора по ключевым словам в запросе или ответе.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Ключевые слова для поиска"},
                    "limit": {
                        "type": "integer",
                        "description": "Максимум результатов (1–50)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(self, args: dict[str, Any], *, user_id: int | None = None, **_: Any) -> ToolResult:
        if self._store is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="DiagnostStore не настроен")
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="query обязателен")
        limit = min(max(int(args.get("limit") or 10), 1), 50)
        results = await self._store.search_events(query, limit=limit)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={"results": results, "count": len(results)})


class GetIncidentTool:
    name = "diagnost_get_incident"

    def __init__(self, store: PostgresDiagnostStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Возвращает инцидент по его ID вместе с исходным событием.",
            parameters={
                "type": "object",
                "properties": {
                    "incident_id": {"type": "string", "description": "Идентификатор инцидента"},
                },
                "required": ["incident_id"],
            },
        )

    async def execute(self, args: dict[str, Any], *, user_id: int | None = None, **_: Any) -> ToolResult:
        if self._store is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="DiagnostStore не настроен")
        incident_id = str(args.get("incident_id") or "").strip()
        if not incident_id:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="incident_id обязателен")
        incident = await self._store.get_incident(incident_id)
        if incident is None:
            return ToolResult(status=ToolStatus.NOT_FOUND, tool=self.name, error=f"Инцидент не найден: {incident_id}")
        return ToolResult(status=ToolStatus.OK, tool=self.name, data=incident)


class ListIncidentsTool:
    name = "diagnost_list_incidents"

    def __init__(self, store: PostgresDiagnostStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Возвращает список инцидентов. Можно фильтровать по статусу (open/resolved).",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Фильтр по статусу: open, resolved или пусто (все)",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимум результатов (1–100)",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
            },
        )

    async def execute(self, args: dict[str, Any], *, user_id: int | None = None, **_: Any) -> ToolResult:
        if self._store is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="DiagnostStore не настроен")
        status = str(args.get("status") or "").strip()
        limit = min(max(int(args.get("limit") or 20), 1), 100)
        incidents = await self._store.list_incidents(status=status, limit=limit)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={"incidents": incidents, "count": len(incidents)})


class CreateIncidentTool:
    name = "diagnost_create_incident"

    def __init__(self, store: PostgresDiagnostStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Создаёт инцидент вручную по ID события с комментарием.",
            parameters={
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "ID события (task_id оркестратора)"},
                    "comment": {"type": "string", "description": "Описание проблемы", "default": ""},
                },
                "required": ["event_id"],
            },
        )

    async def execute(self, args: dict[str, Any], *, user_id: int | None = None, **_: Any) -> ToolResult:
        if self._store is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="DiagnostStore не настроен")
        event_id = str(args.get("event_id") or "").strip()
        if not event_id:
            return ToolResult(status=ToolStatus.INVALID_TOOL_CALL, tool=self.name, error="event_id обязателен")
        comment = str(args.get("comment") or "").strip()
        incident_id = await self._store.save_incident(event_id, reason="manual", comment=comment)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data={"incident_id": incident_id, "event_id": event_id})


class ErrorReportTool:
    name = "diagnost_error_report"

    def __init__(self, store: PostgresDiagnostStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description="Формирует сводный отчёт по инцидентам за указанный период (без LLM, чистая агрегация).",
            parameters={
                "type": "object",
                "properties": {
                    "since_hours": {
                        "type": "integer",
                        "description": "За сколько часов назад (1–720)",
                        "default": 24,
                        "minimum": 1,
                        "maximum": 720,
                    },
                },
            },
        )

    async def execute(self, args: dict[str, Any], *, user_id: int | None = None, **_: Any) -> ToolResult:
        if self._store is None:
            return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error="DiagnostStore не настроен")
        since_hours = min(max(int(args.get("since_hours") or 24), 1), 720)
        report = await self._store.error_report(since_hours=since_hours)
        return ToolResult(status=ToolStatus.OK, tool=self.name, data=report)
