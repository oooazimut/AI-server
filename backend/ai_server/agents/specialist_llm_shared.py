from __future__ import annotations

import logging
from typing import Any

from ai_server.models import AgentManifest, ToolResult
from ai_server.registry import resolve_project_path
from ai_server.retrieval import RetrievalHit
from ai_server.skills import Skill

logger = logging.getLogger(__name__)

DECISION_STATUSES = {"completed", "needs_clarification", "needs_human"}
RESULT_STATUSES = {"completed", "needs_clarification", "needs_human", "failed"}

SKILLS_PROMPT_FRAGMENT = (
    "В payload есть available_skills — пошаговые how-to для типовых сценариев. "
    "Читай их перед выбором инструментов: они описывают правильную последовательность действий. "
)

DIALOG_HISTORY_PROMPT_FRAGMENT = (
    "В payload есть dialog_history — последние сообщения этого диалога (роли user/assistant). "
    "Текущий request может быть продолжением (уточнением имени, местоимением 'он/она/этот', "
    "ответом на твой предыдущий уточняющий вопрос). Используй dialog_history, чтобы понять, "
    "о какой сущности (сотрудник, задача, проект) идёт речь, и не задавай уточняющий вопрос повторно, "
    "если ответ на него уже есть в dialog_history. "
)


def load_instructions(manifest: AgentManifest) -> str:
    if not manifest.instructions_file:
        return ""
    try:
        path = resolve_project_path(manifest.instructions_file)
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Failed to load instructions from %s: %s", manifest.instructions_file, exc)
        return ""


def decision_status(value: object) -> str:
    status = str(value or "completed").strip()
    return status if status in DECISION_STATUSES else "completed"


def result_status(value: object) -> str:
    status = str(value or "completed").strip()
    return status if status in RESULT_STATUSES else "completed"


def retrieval_context(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    context = []
    for hit in hits[:5]:
        context.append(
            {
                "topic": hit.chunk.topic,
                "section": hit.chunk.section,
                "score": hit.score,
                "text": hit.chunk.text[:1200],
            }
        )
    return context


def allowed_tool_definitions(definitions: list[dict[str, Any]], allowed_tool_names: set[str]) -> list[dict[str, Any]]:
    return [definition for definition in definitions if definition.get("name") in allowed_tool_names]


def safe_context(context: dict[str, Any]) -> dict[str, Any]:
    """Strip internal per-request keys (toolset objects) before passing context to json.dumps."""
    return {k: v for k, v in context.items() if not k.startswith("_")}


def skills_context(skills: list[Skill]) -> list[dict[str, Any]]:
    return [{"id": s.id, "title": s.title, "content": s.content or s.preview} for s in skills]


def compact_tool_result(result: ToolResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "tool": result.tool,
        "data": result.data,
        "error": result.error,
    }
