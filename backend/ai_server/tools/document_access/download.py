from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_server.integrations.bitrix.portal_search import (
    PortalSearchResult,
    portal_file_cache_path,
)
from ai_server.settings import Settings
from ai_server.tools.bitrix_ports import BitrixFileDownloadPort
from ai_server.utils import optional_int


async def resolve_portal_file_download_url(bitrix: BitrixFileDownloadPort, item: PortalSearchResult) -> str | None:
    if item.entity_type == "task_attachment":
        attached_object_id = optional_int(item.metadata.get("attached_object_id")) or optional_int(item.entity_id)
        if attached_object_id is None:
            return None
        attached = await bitrix.get_attached_object(attached_object_id)
        if isinstance(attached, dict):
            return str(_first(attached, "DOWNLOAD_URL", "downloadUrl") or "")
        return None

    if item.entity_type == "disk_file":
        disk_file_id = optional_int(item.metadata.get("disk_object_id")) or optional_int(item.entity_id)
        if disk_file_id is None:
            return None
        return await bitrix.get_disk_file_download_url(disk_file_id)
    return None


async def _ensure_local_document(
    bitrix: BitrixFileDownloadPort, item: PortalSearchResult, *, max_bytes: int, settings: Settings
) -> Path:
    path = portal_file_cache_path(item, settings)
    if path.exists() and path.stat().st_size > 0:
        return path
    download_url = await resolve_portal_file_download_url(bitrix, item)
    if not download_url:
        raise ValueError(f"Bitrix не вернул ссылку на скачивание: {item.title}")
    await bitrix.download_file_from_url(download_url, path, max_bytes=max_bytes)
    return path


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None
