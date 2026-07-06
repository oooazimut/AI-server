from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_server.registry import PROJECT_ROOT

DEFAULT_BITRIX_AGENT_VAR = Path(r"C:\Users\office3pc\PyProjects\BitrixAIAgent\var")
DEFAULT_AI_SERVER_VAR = PROJECT_ROOT / "var"


@dataclass(frozen=True)
class VarMigrationItem:
    path: str
    item_type: str
    category: str
    description: str
    profiles: tuple[str, ...] = ("cutover",)
    sensitive: bool = False
    required_for_cutover: bool = False


MIGRATION_ITEMS: tuple[VarMigrationItem, ...] = (
    VarMigrationItem(
        path="search_index.sqlite",
        item_type="file",
        category="portal_search",
        description="Локальный индекс портала Битрикс24.",
        profiles=("cutover", "index_only"),
        required_for_cutover=True,
        sensitive=True,
    ),
    VarMigrationItem(
        path="search_content",
        item_type="directory",
        category="portal_search",
        description="Кэш извлечённого содержимого документов для поиска/RAG.",
        profiles=("cutover", "index_only"),
        sensitive=True,
    ),
    VarMigrationItem(
        path="search_indexer_state.json",
        item_type="file",
        category="portal_search",
        description="Состояние периодического индексатора.",
        profiles=("cutover", "index_only"),
    ),
    VarMigrationItem(
        path="webhook_event_queue.sqlite",
        item_type="file",
        category="events",
        description="Очередь webhook-событий с dedupe/retry состоянием.",
        required_for_cutover=True,
        sensitive=True,
    ),
    VarMigrationItem(
        path="dialog_state.sqlite",
        item_type="file",
        category="conversation_state",
        description="Состояние старых диалогов Bitrix-агента.",
        required_for_cutover=True,
        sensitive=True,
    ),
    VarMigrationItem(
        path="bitrix_oauth.sqlite",
        item_type="file",
        category="credentials",
        description="OAuth-токены Bitrix. Переносить только при cutover и совместимых настройках приложения.",
        required_for_cutover=True,
        sensitive=True,
    ),
    VarMigrationItem(
        path="bitrix_write_audit.jsonl",
        item_type="file",
        category="audit",
        description="Журнал write-действий старого агента.",
        sensitive=True,
    ),
    VarMigrationItem(
        path="quality_control_state.json",
        item_type="file",
        category="quality_control",
        description="State контроля качества закрытия задач.",
        sensitive=True,
    ),
    VarMigrationItem(
        path="supervisor_state.json",
        item_type="file",
        category="task_supervisor",
        description="State cooldown-уведомлений по просроченным задачам.",
        sensitive=True,
    ),
    VarMigrationItem(
        path="vehicle_usage.sqlite",
        item_type="file",
        category="vehicle_usage",
        description="Legacy SQLite export for vehicle usage cutover; target runtime storage is PostgreSQL schema logistics.",
        sensitive=True,
    ),
    VarMigrationItem(
        path="attachments",
        item_type="directory",
        category="attachments",
        description="Скачанные вложения и голосовые файлы старых диалогов.",
        profiles=("cutover", "attachments"),
        sensitive=True,
    ),
    VarMigrationItem(
        path="document_drafts",
        item_type="directory",
        category="document_drafts",
        description="Сгенерированные или подготовленные черновики документов.",
        profiles=("cutover", "documents"),
        sensitive=True,
    ),
    VarMigrationItem(
        path="learning_events.jsonl",
        item_type="file",
        category="audit",
        description="История learning/events старого агента.",
        sensitive=True,
    ),
)

EXCLUDED_RUNTIME_ITEMS: tuple[str, ...] = (
    "search_indexer.lock",
    "dev_outgoing_webhook_setup.txt",
    "ngrok-dev.err.log",
    "ngrok-dev.out.log",
    "tmp*",
)


def selected_migration_items(profile: str = "cutover") -> list[VarMigrationItem]:
    return [item for item in MIGRATION_ITEMS if profile in item.profiles]


def build_var_migration_plan(
    source_var: Path | str = DEFAULT_BITRIX_AGENT_VAR,
    target_var: Path | str = DEFAULT_AI_SERVER_VAR,
    *,
    profile: str = "cutover",
) -> list[dict[str, Any]]:
    source_root = Path(source_var)
    target_root = Path(target_var)
    return [
        _plan_item(item, source_root=source_root, target_root=target_root) for item in selected_migration_items(profile)
    ]


def migrate_var(
    source_var: Path | str = DEFAULT_BITRIX_AGENT_VAR,
    target_var: Path | str = DEFAULT_AI_SERVER_VAR,
    *,
    profile: str = "cutover",
    execute: bool = False,
    backup_existing: bool = True,
) -> list[dict[str, Any]]:
    source_root = Path(source_var).resolve()
    target_root = Path(target_var).resolve()
    if source_root == target_root:
        raise ValueError("source_var and target_var must be different directories")

    plan = build_var_migration_plan(source_root, target_root, profile=profile)
    if not execute:
        return plan

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = target_root / "legacy" / "backups" / timestamp
    target_root.mkdir(parents=True, exist_ok=True)

    executed: list[dict[str, Any]] = []
    for entry in plan:
        if not entry["exists"]:
            executed.append({**entry, "result": "missing"})
            continue

        source_path = Path(entry["source"])
        target_path = Path(entry["target"])
        _assert_inside(target_path, target_root)
        if target_path.exists() and backup_existing:
            backup_path = backup_root / entry["path"]
            _assert_inside(backup_path, backup_root)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.is_dir():
                shutil.copytree(target_path, backup_path)
            else:
                shutil.copy2(target_path, backup_path)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if entry["item_type"] == "directory":
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.copytree(source_path, target_path)
            (target_path / ".gitkeep").touch(exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)
        executed.append({**entry, "result": "copied"})
    return executed


def _plan_item(
    item: VarMigrationItem,
    *,
    source_root: Path,
    target_root: Path,
) -> dict[str, Any]:
    source_path = source_root / item.path
    target_path = target_root / item.path
    exists = source_path.exists()
    return {
        "path": item.path,
        "item_type": item.item_type,
        "category": item.category,
        "description": item.description,
        "source": str(source_path),
        "target": str(target_path),
        "exists": exists,
        "sensitive": item.sensitive,
        "required_for_cutover": item.required_for_cutover,
        "action": "copy" if exists else "missing",
    }


def _assert_inside(path: Path, root: Path) -> None:
    path.resolve().relative_to(root.resolve())
