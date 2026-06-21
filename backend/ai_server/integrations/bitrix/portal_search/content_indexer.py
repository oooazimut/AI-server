from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from ai_server.document_text import extract_text_from_file
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search.file_cache import delete_portal_file_cache_path, portal_file_cache_path
from ai_server.integrations.bitrix.portal_search.search_index import PortalSearchIndex
from ai_server.integrations.bitrix.portal_search.text_utils import (
    body_with_content,
    file_extension,
    normalize_extensions,
    safe_int,
    to_str,
)
from ai_server.integrations.bitrix.portal_search.types import (
    CONTENT_INDEX_VERSION,
    PortalContentSyncStats,
    PortalSearchResult,
)
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ


async def sync_portal_content_index(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    extensions: set[str] | None = None,
    settings: Settings,
) -> PortalContentSyncStats:
    stats = PortalContentSyncStats(errors=[])
    allowed_extensions = normalize_extensions(settings.resolved_search_content_allowed_extensions)
    if extensions:
        allowed_extensions &= normalize_extensions(extensions)
    candidate_limit = max(settings.search_content_max_files * 50, settings.search_content_max_files)
    processed_downloads = 0

    for item in index.content_candidates(limit=candidate_limit):
        metadata = dict(item.metadata)
        status = str(metadata.get("content_index_status") or "")
        content_version = str(metadata.get("content_index_version") or "")
        extension = file_extension(item.title)
        if extensions and extension not in allowed_extensions:
            continue
        if status == "indexed" and (content_version == CONTENT_INDEX_VERSION or not extensions):
            continue
        if content_version == CONTENT_INDEX_VERSION and status in {
            "unsupported",
            "too_large",
            "empty",
            "failed",
            "no_download_url",
        }:
            continue

        stats.candidates += 1
        if extension not in allowed_extensions:
            _mark_content_status(
                index,
                item,
                metadata,
                status="unsupported",
                reason=f"extension {extension or '<none>'} is not enabled",
            )
            stats.unsupported += 1
            continue

        size = safe_int(metadata.get("size"))
        if size and size > settings.search_content_max_bytes:
            _mark_content_status(
                index,
                item,
                metadata,
                status="too_large",
                reason=f"file exceeds {settings.search_content_max_bytes} bytes",
            )
            stats.skipped += 1
            continue

        if processed_downloads >= settings.search_content_max_files:
            break
        processed_downloads += 1

        item_stats = await sync_portal_content_item(
            bitrix,
            index,
            item,
            extensions={extension},
            settings=settings,
        )
        stats.downloaded += item_stats.downloaded
        stats.indexed += item_stats.indexed
        stats.skipped += item_stats.skipped
        stats.unsupported += item_stats.unsupported
        stats.failed += item_stats.failed
        if item_stats.errors:
            stats.errors.extend(item_stats.errors)

    if not stats.errors:
        stats.errors = None
    return stats


async def sync_portal_content_item(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    item: PortalSearchResult,
    *,
    extensions: set[str] | None = None,
    settings: Settings,
) -> PortalContentSyncStats:
    stats = PortalContentSyncStats(errors=[])
    allowed_extensions = normalize_extensions(settings.resolved_search_content_allowed_extensions)
    if extensions:
        allowed_extensions &= normalize_extensions(extensions)

    metadata = dict(item.metadata)
    extension = file_extension(item.title)
    if extensions and extension not in allowed_extensions:
        return stats

    stats.candidates = 1
    if extension not in allowed_extensions:
        _mark_content_status(
            index,
            item,
            metadata,
            status="unsupported",
            reason=f"extension {extension or '<none>'} is not enabled",
        )
        stats.unsupported += 1
        return stats

    size = safe_int(metadata.get("size"))
    if size and size > settings.search_content_max_bytes:
        _mark_content_status(
            index,
            item,
            metadata,
            status="too_large",
            reason=f"file exceeds {settings.search_content_max_bytes} bytes",
        )
        stats.skipped += 1
        return stats

    target_path = None
    downloaded_for_indexing = False
    try:
        download_url = await _resolve_download_url(bitrix, item)
        if not download_url:
            _mark_content_status(
                index,
                item,
                metadata,
                status="no_download_url",
                reason="Bitrix did not return a download URL",
            )
            stats.failed += 1
            return stats

        target_path = portal_file_cache_path(item, settings)
        downloaded_bytes = await bitrix.download_file_from_url(
            download_url,
            target_path,
            max_bytes=settings.search_content_max_bytes,
        )
        downloaded_for_indexing = True
        stats.downloaded += 1

        extracted = await asyncio.to_thread(
            extract_text_from_file,
            target_path,
            original_name=item.title,
            max_chars=settings.search_content_max_chars,
        )
        metadata.update(
            {
                "content_index_status": extracted.status,
                "content_index_version": CONTENT_INDEX_VERSION,
                "content_index_reason": extracted.reason,
                "content_indexed_at": datetime.now(MOSCOW_TZ).isoformat(),
                "content_bytes": downloaded_bytes,
                "content_extension": extension,
                "content_text_length": len(extracted.text),
            }
        )
        if extracted.status == "indexed":
            index.update_item_body_metadata(
                entity_type=item.entity_type,
                entity_id=item.entity_id,
                body=body_with_content(item.body, extracted.text),
                metadata=metadata,
            )
            stats.indexed += 1
        else:
            index.update_item_body_metadata(
                entity_type=item.entity_type,
                entity_id=item.entity_id,
                body=item.body,
                metadata=metadata,
            )
            if extracted.status == "unsupported":
                stats.unsupported += 1
            elif extracted.status == "failed":
                stats.failed += 1
            else:
                stats.skipped += 1
    except Exception as exc:
        metadata.update(
            {
                "content_index_status": "failed",
                "content_index_version": CONTENT_INDEX_VERSION,
                "content_index_reason": type(exc).__name__,
                "content_indexed_at": datetime.now(MOSCOW_TZ).isoformat(),
                "content_extension": extension,
            }
        )
        index.update_item_body_metadata(
            entity_type=item.entity_type,
            entity_id=item.entity_id,
            body=item.body,
            metadata=metadata,
        )
        stats.failed += 1
        stats.errors = [f"{item.entity_type} #{item.entity_id} {item.title}: {type(exc).__name__}"]
    finally:
        if downloaded_for_indexing and target_path is not None and not settings.search_content_keep_local_files:
            delete_portal_file_cache_path(target_path, settings)

    return stats


async def _resolve_download_url(bitrix: BitrixClient, item: PortalSearchResult) -> str | None:
    if item.entity_type == "task_attachment":
        attached_object_id = safe_int(item.metadata.get("attached_object_id")) or safe_int(item.entity_id)
        if not attached_object_id:
            return None
        attached = await bitrix.get_attached_object(attached_object_id)
        if isinstance(attached, dict):
            return to_str(_first(attached, "DOWNLOAD_URL", "downloadUrl"))
        return None

    if item.entity_type == "disk_file":
        disk_file_id = safe_int(item.metadata.get("disk_object_id")) or safe_int(item.entity_id)
        if not disk_file_id:
            return None
        return await bitrix.get_disk_file_download_url(disk_file_id)

    return None


def _mark_content_status(
    index: PortalSearchIndex,
    item: PortalSearchResult,
    metadata: dict[str, Any],
    *,
    status: str,
    reason: str,
) -> None:
    metadata.update(
        {
            "content_index_status": status,
            "content_index_version": CONTENT_INDEX_VERSION,
            "content_index_reason": reason,
            "content_indexed_at": datetime.now(MOSCOW_TZ).isoformat(),
            "content_extension": file_extension(item.title),
        }
    )
    index.update_item_body_metadata(
        entity_type=item.entity_type,
        entity_id=item.entity_id,
        body=item.body,
        metadata=metadata,
    )


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None
