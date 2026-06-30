from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_server.models import ToolDefinition, ToolResult, ToolStatus

if TYPE_CHECKING:
    from ai_server.integrations.postgres.kartoteka_agent import PostgresKartotekaStore

_NOT_CONFIGURED_MSG = "Операция временно недоступна: файловый сервер не подключён"


class FileAddTool:
    name = "kartoteka_file_add"

    def __init__(self, store: PostgresKartotekaStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="kartoteka_file_add",
            description="Добавляет файл в каталог по указанному пути.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Полный путь к файлу"},
                    "filename": {"type": "string", "description": "Имя файла"},
                    "tags": {"type": "string", "description": "Теги через запятую"},
                },
                "required": ["path", "filename"],
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
        return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error=_NOT_CONFIGURED_MSG)


class FileDeleteTool:
    name = "kartoteka_file_delete"

    def __init__(self, store: PostgresKartotekaStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="kartoteka_file_delete",
            description="Удаляет файл из каталога по пути.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Полный путь к файлу"},
                },
                "required": ["path"],
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
        return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error=_NOT_CONFIGURED_MSG)


class FileMoveTool:
    name = "kartoteka_file_move"

    def __init__(self, store: PostgresKartotekaStore | None = None) -> None:
        self._store = store

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="kartoteka_file_move",
            description="Перемещает или переименовывает файл в каталоге.",
            parameters={
                "type": "object",
                "properties": {
                    "old_path": {"type": "string", "description": "Текущий путь к файлу"},
                    "new_path": {"type": "string", "description": "Новый путь к файлу"},
                },
                "required": ["old_path", "new_path"],
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
        return ToolResult(status=ToolStatus.NOT_CONFIGURED, tool=self.name, error=_NOT_CONFIGURED_MSG)
