from __future__ import annotations

from datetime import datetime

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search.content_indexer import sync_portal_content_index
from ai_server.integrations.bitrix.portal_search.entity_syncs import (
    _delta_folder_id,
    _delta_folder_path,
    _delta_storage_name,
    _sync_catalog,
    _sync_disk,
    _sync_disk_folder_delta,
    _sync_projects,
    _sync_tasks,
)
from ai_server.integrations.bitrix.portal_search.search_index import PortalSearchIndex
from ai_server.integrations.bitrix.portal_search.types import (
    PortalDeltaSyncStats,
    PortalSyncStats,
)
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ


async def sync_portal_index(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    include_content: bool = False,
    settings: Settings,
) -> PortalSyncStats:
    sync_started_at = datetime.now(MOSCOW_TZ).isoformat()
    stats = PortalSyncStats(errors=[], prune_skipped=[])

    stats.projects = await _sync_projects(bitrix, index, settings)
    if stats.projects < settings.search_index_max_projects:
        stats.stale_deleted += index.delete_stale_items(
            entity_types={"project"},
            seen_before=sync_started_at,
        )
    else:
        stats.prune_skipped.append("projects: reached configured limit")

    task_sync = await _sync_tasks(bitrix, index, settings)
    stats.tasks = int(task_sync["tasks"])
    stats.task_attachments = int(task_sync["attachments"])
    task_prune_types = set()
    if bool(task_sync["tasks_complete"]):
        task_prune_types.add("task")
    else:
        stats.prune_skipped.append("tasks: reached configured limit")
    if bool(task_sync["attachments_complete"]):
        task_prune_types.add("task_attachment")
    elif settings.search_index_include_task_attachments:
        stats.prune_skipped.append("task attachments: reached configured limit")
    if task_prune_types:
        stats.stale_deleted += index.delete_stale_items(
            entity_types=task_prune_types,
            seen_before=sync_started_at,
        )

    if settings.search_index_include_catalog:
        try:
            catalog_stats = await _sync_catalog(bitrix, index, settings)
            stats.catalog_products = catalog_stats["products"]
            stats.catalog_stores = catalog_stats["stores"]
            if stats.catalog_products < settings.search_index_max_catalog_products:
                stats.stale_deleted += index.delete_stale_items(
                    entity_types={"catalog_product", "catalog_store"},
                    seen_before=sync_started_at,
                )
            else:
                stats.prune_skipped = (stats.prune_skipped or []) + ["catalog: reached configured limit"]
        except Exception as exc:
            stats.errors = (stats.errors or []) + [f"catalog: {type(exc).__name__}: {exc}"]

    if settings.search_index_include_disk:
        try:
            disk_stats = await _sync_disk(bitrix, index, settings)
            stats.storages = int(disk_stats["storages"])
            stats.disk_items = int(disk_stats["items"])
            if bool(disk_stats["complete"]):
                stats.stale_deleted += index.delete_stale_items(
                    entity_types={"disk_storage", "disk_folder", "disk_file"},
                    seen_before=sync_started_at,
                )
            else:
                stats.prune_skipped.append("disk: reached configured limit")
        except Exception as exc:
            stats.errors.append(f"disk: {type(exc).__name__}: {exc}")

    if include_content and settings.search_content_enabled:
        try:
            stats.content = await sync_portal_content_index(bitrix, index, settings=settings)
        except Exception as exc:
            stats.errors.append(f"content: {type(exc).__name__}: {exc}")
    if not stats.prune_skipped:
        stats.prune_skipped = None
    if not stats.errors:
        stats.errors = None
    return stats


async def sync_disk_delta_index(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    cursor_type: str | None,
    cursor_id: str | None,
    folder_limit: int,
    child_limit: int,
    settings: Settings,
) -> PortalDeltaSyncStats:
    stats = PortalDeltaSyncStats(errors=[])
    folders, next_type, next_id, wrapped = index.disk_delta_folder_candidates(
        cursor_type=cursor_type,
        cursor_id=cursor_id,
        limit=folder_limit,
    )
    stats.cursor_type = next_type
    stats.cursor_id = next_id
    stats.wrapped = wrapped

    for folder in folders:
        folder_id = _delta_folder_id(folder)
        if folder_id is None:
            continue
        try:
            folder_stats = await _sync_disk_folder_delta(
                bitrix,
                index,
                folder_id=folder_id,
                storage_name=_delta_storage_name(folder),
                path=_delta_folder_path(folder),
                child_limit=child_limit,
                settings=settings,
            )
            stats.folders_scanned += 1
            stats.items_seen += int(folder_stats["items_seen"])
            stats.items_changed += int(folder_stats["items_changed"])
            stats.files_changed += int(folder_stats["files_changed"])
            stats.folders_changed += int(folder_stats["folders_changed"])
            stats.deleted += int(folder_stats["deleted"])
        except Exception as exc:
            stats.errors.append(f"{folder.entity_type} #{folder.entity_id}: {type(exc).__name__}: {exc}")
    if not stats.errors:
        stats.errors = None
    return stats
