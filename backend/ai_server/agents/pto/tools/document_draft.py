from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from ai_server.models import ToolDefinition, ToolResult, ToolStatus
from ai_server.settings import Settings
from ai_server.utils import optional_int


class DocumentDraftCreateTool:
    name = "document_draft_create"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="document_draft_create",
            description="Create a local PTO document draft from explicit LLM-provided content.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "extension": {"type": "string", "enum": [".txt", ".md"]},
                },
                "required": ["title", "content"],
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
        title = str(args.get("title") or "").strip()
        content = str(args.get("content") or "").strip()
        extension = str(args.get("extension") or ".md").strip().lower()
        if extension not in {".txt", ".md"}:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="document_draft_create",
                error="extension must be .txt or .md",
            )
        if not title or not content:
            return ToolResult(
                status=ToolStatus.INVALID_TOOL_CALL,
                tool="document_draft_create",
                error="title and content are required",
            )
        drafts_dir = self._settings.document_drafts_dir
        drafts_dir.mkdir(parents=True, exist_ok=True)
        path = (
            drafts_dir
            / f"{datetime.now(UTC).astimezone().strftime('%Y%m%d-%H%M%S')}-{_safe_draft_name(title, extension)}"
        )
        path.write_text(content + "\n", encoding="utf-8")
        return ToolResult(
            status=ToolStatus.OK,
            tool="document_draft_create",
            data={
                "title": title,
                "path": str(path),
                "bytes": path.stat().st_size,
                "message": "Draft was created locally; uploading/sending requires a separate approved write action.",
            },
        )


class DocumentDraftListTool:
    name = "document_draft_list"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="document_draft_list",
            description="List recent local PTO document drafts.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                },
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
        limit = max(1, min(optional_int(args.get("limit")) or 10, 30))
        drafts_dir = self._settings.document_drafts_dir
        drafts_dir.mkdir(parents=True, exist_ok=True)
        drafts = sorted(
            [path for path in drafts_dir.iterdir() if path.is_file()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:limit]
        return ToolResult(
            status=ToolStatus.OK,
            tool="document_draft_list",
            data={
                "drafts": [
                    {
                        "name": path.name,
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).astimezone().isoformat(),
                    }
                    for path in drafts
                ],
                "total": len(drafts),
            },
        )


def _safe_draft_name(title: str, extension: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9а-яА-Я._-]+", "_", title).strip("._")
    if not name:
        name = "draft"
    if not name.lower().endswith(extension):
        name += extension
    return name[:120]
