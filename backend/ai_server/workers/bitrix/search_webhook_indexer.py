from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search import (
    PortalSearchIndex,
    delete_portal_file_cache_path,
    format_portal_content_sync_stats,
    portal_file_cache_path,
    sync_catalog_product_item,
    sync_disk_file_item,
    sync_portal_content_item,
    sync_task_item,
)
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)

DISK_FILE_EVENT_MARKERS = ("DISK", "FILE")
DELETE_MARKERS = ("DELETE", "TRASH", "MARKDELETED")
UPDATE_MARKERS = ("ADD", "CREATE", "UPDATE", "RESTORE", "MOVE", "RENAME")


@dataclass(frozen=True)
class SearchWebhookJob:
    event: str
    action: str
    entity_type: str
    entity_id: int

    @property
    def file_id(self) -> int:
        return self.entity_id


def prepare_search_webhook_job(
    payload: dict[str, Any], *, settings: Settings
) -> tuple[SearchWebhookJob | None, dict[str, Any]]:
    event = _payload_event_type(payload)
    if not settings.search_webhook_indexer_enabled:
        return None, {"handled": False, "reason": "disabled", "event": event}
    entity_type = _event_entity_type(event)
    if entity_type is None:
        return None, {"handled": False, "reason": "unsupported_event", "event": event}
    entity_id = _extract_event_entity_id(payload, entity_type=entity_type, event=event)
    if entity_id is None:
        return None, {"handled": False, "reason": f"{entity_type}_id_not_found", "event": event}

    action = _event_action(event)
    if action is None:
        return None, {"handled": False, "reason": "unsupported_action", "event": event, "entity_id": entity_id}

    return SearchWebhookJob(event=event, action=action, entity_type=entity_type, entity_id=entity_id), {
        "handled": True,
        "queued": True,
        "event": event,
        "action": action,
        "entity_type": entity_type,
        "entity_id": entity_id,
    }


async def process_search_webhook_job(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    job: SearchWebhookJob,
    *,
    status: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    _record_seen(status, job, settings=settings)
    try:
        if job.entity_type == "disk_file" and job.action == "delete":
            result = await _delete_indexed_file(index, job, settings=settings)
        elif job.action == "delete":
            deleted = index.delete_item(entity_type=job.entity_type, entity_id=job.entity_id)
            result = {
                "handled": True,
                "reason": "deleted" if deleted else "already_absent",
                "event": job.event,
                "action": job.action,
                "entity_type": job.entity_type,
                "entity_id": job.entity_id,
                "deleted": deleted,
            }
        elif job.entity_type == "disk_file":
            result = await _upsert_indexed_file(bitrix, index, job, settings=settings)
        elif job.entity_type == "task":
            result = await _upsert_indexed_task(bitrix, index, job, settings=settings)
        elif job.entity_type == "catalog_product":
            result = await _upsert_indexed_product(bitrix, index, job, settings=settings)
        else:
            result = {"handled": False, "reason": "unsupported_entity_type", "entity_type": job.entity_type}
        status["last_reason"] = result.get("reason")
        status["last_result"] = result
        status["processed"] = int(status.get("processed") or 0) + 1
        return result
    except Exception as exc:
        logger.exception("Search webhook indexing failed")
        status["last_error"] = f"{type(exc).__name__}: {exc}"
        status["last_reason"] = "error"
        status["errors"] = int(status.get("errors") or 0) + 1
        return {
            "handled": False,
            "reason": "error",
            "event": job.event,
            "action": job.action,
            "entity_type": job.entity_type,
            "entity_id": job.entity_id,
            "error": type(exc).__name__,
        }


async def _delete_indexed_file(
    index: PortalSearchIndex, job: SearchWebhookJob, *, settings: Settings
) -> dict[str, Any]:
    existing = index.get_item(entity_type="disk_file", entity_id=job.file_id)
    if existing:
        delete_portal_file_cache_path(portal_file_cache_path(existing, settings), settings)
    deleted = index.delete_item(entity_type="disk_file", entity_id=job.file_id)
    return {
        "handled": True,
        "reason": "deleted" if deleted else "already_absent",
        "event": job.event,
        "action": job.action,
        "file_id": job.file_id,
        "deleted": deleted,
    }


async def _upsert_indexed_file(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    job: SearchWebhookJob,
    *,
    settings: Settings,
) -> dict[str, Any]:
    item = await sync_disk_file_item(
        bitrix,
        index,
        file_id=job.file_id,
        preserve_content=True,
        settings=settings,
    )
    if not item:
        return {
            "handled": False,
            "reason": "file_not_loaded",
            "event": job.event,
            "action": job.action,
            "file_id": job.file_id,
        }

    result: dict[str, Any] = {
        "handled": True,
        "reason": "metadata_indexed",
        "event": job.event,
        "action": job.action,
        "file_id": job.file_id,
        "title": item.title,
    }
    if not settings.search_webhook_content_enabled:
        result["content"] = {"handled": False, "reason": "disabled"}
        return result

    extension = _file_extension(item.title)
    if extension not in settings.resolved_search_content_allowed_extensions:
        result["content"] = {
            "handled": False,
            "reason": "unsupported_extension",
            "extension": extension or "<none>",
        }
        return result

    stats = await sync_portal_content_item(bitrix, index, item, extensions={extension}, settings=settings)
    result["content"] = {
        "handled": True,
        "summary": format_portal_content_sync_stats(stats),
        "downloaded": stats.downloaded,
        "indexed": stats.indexed,
        "failed": stats.failed,
        "unsupported": stats.unsupported,
        "skipped": stats.skipped,
    }
    result["reason"] = "metadata_and_content_indexed"
    return result


async def _upsert_indexed_task(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    job: SearchWebhookJob,
    *,
    settings: Settings,
) -> dict[str, Any]:
    item = await sync_task_item(bitrix, index, task_id=job.entity_id, settings=settings)
    return {
        "handled": item is not None,
        "reason": "task_indexed" if item is not None else "task_not_loaded",
        "event": job.event,
        "action": job.action,
        "task_id": job.entity_id,
        "title": item.title if item is not None else "",
    }


async def _upsert_indexed_product(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    job: SearchWebhookJob,
    *,
    settings: Settings,
) -> dict[str, Any]:
    item = await sync_catalog_product_item(bitrix, index, product_id=job.entity_id, settings=settings)
    return {
        "handled": item is not None,
        "reason": "catalog_product_indexed" if item is not None else "catalog_product_not_loaded",
        "event": job.event,
        "action": job.action,
        "product_id": job.entity_id,
        "title": item.title if item is not None else "",
    }


def _record_seen(status: dict[str, Any], job: SearchWebhookJob, *, settings: Settings) -> None:
    status["enabled"] = settings.search_webhook_indexer_enabled
    status["last_received_at"] = datetime.now(MOSCOW_TZ).isoformat()
    status["last_event"] = job.event
    status["last_entity_type"] = job.entity_type
    status["last_entity_id"] = job.entity_id
    if job.entity_type == "disk_file":
        status["last_file_id"] = job.file_id
    status["last_action"] = job.action
    status["last_error"] = None
    status["events_seen"] = int(status.get("events_seen") or 0) + 1


def _payload_event_type(payload: dict[str, Any]) -> str:
    return str(payload.get("event") or payload.get("EVENT") or payload.get("type") or "").upper()


def _is_disk_file_event(event: str) -> bool:
    return all(marker in event for marker in DISK_FILE_EVENT_MARKERS)


def _event_entity_type(event: str) -> str | None:
    if _is_disk_file_event(event):
        return "disk_file"
    if "TASK" in event and ("COMMENT" in event or event.startswith("ONTASK")):
        return "task"
    if "CATALOG" in event and "PRODUCT" in event:
        return "catalog_product"
    return None


def _event_action(event: str) -> str | None:
    if any(marker in event for marker in DELETE_MARKERS):
        return "delete"
    if any(marker in event for marker in UPDATE_MARKERS):
        return "upsert"
    return None


def _extract_file_id(payload: dict[str, Any]) -> int | None:
    candidates = [
        payload.get("file_id"),
        payload.get("fileId"),
        payload.get("FILE_ID"),
        payload.get("id"),
        payload.get("ID"),
    ]
    data = payload.get("data")
    if isinstance(data, dict):
        candidates.extend(_nested_candidates(data))
    return _first_int(candidates)


def _extract_event_entity_id(payload: dict[str, Any], *, entity_type: str, event: str = "") -> int | None:
    if entity_type == "disk_file":
        return _extract_file_id(payload)
    preferred = (
        ("task_id", "taskId", "TASK_ID", "TASKID")
        if entity_type == "task"
        else ("product_id", "productId", "PRODUCT_ID")
    )
    candidates = [payload.get(key) for key in preferred]
    data = payload.get("data")
    if isinstance(data, dict):
        candidates.extend(
            _nested_candidates_for_keys(
                data,
                preferred,
                allow_generic_id=entity_type != "task" or "COMMENT" not in event,
            )
        )
    if entity_type != "task" or "COMMENT" not in event:
        # Entity update events frequently expose the entity as ID inside FIELDS_AFTER.
        candidates.extend([payload.get("id"), payload.get("ID")])
    return _first_int(candidates)


def _nested_candidates_for_keys(
    data: dict[str, Any],
    keys: tuple[str, ...],
    *,
    allow_generic_id: bool,
) -> list[Any]:
    candidates = [data.get(key) for key in keys]
    for nested_key in ("FIELDS_AFTER", "FIELDS_BEFORE", "fields", "PARAMS", "params"):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            candidates.extend(_nested_candidates_for_keys(nested, keys, allow_generic_id=allow_generic_id))
            if allow_generic_id:
                candidates.extend([nested.get("ID"), nested.get("id")])
    return candidates


def _nested_candidates(data: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for key in (
        "ID",
        "id",
        "FILE_ID",
        "fileId",
        "OBJECT_ID",
        "objectId",
        "file_id",
        "object_id",
    ):
        candidates.append(data.get(key))
    for nested_key in (
        "FIELDS_AFTER",
        "FIELDS_BEFORE",
        "fields",
        "file",
        "object",
        "PARAMS",
        "params",
    ):
        nested = data.get(nested_key)
        if isinstance(nested, dict):
            candidates.extend(_nested_candidates(nested))
    return candidates


def _first_int(candidates: list[Any]) -> int | None:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, int):
            return candidate
        cleaned = str(candidate).strip()
        if cleaned.isdigit():
            return int(cleaned)
    return None


def _file_extension(name: str) -> str:
    from pathlib import Path

    return Path(name).suffix.lower()
