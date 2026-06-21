from __future__ import annotations

from typing import Any

from ai_server.settings import Settings


def user_has_private_disk_restrictions(user_id: int | None, *, settings: Settings) -> bool:
    return user_id is not None and user_id in settings.resolved_agent_private_disk_restricted_user_ids


def is_private_disk_item(item: Any, *, settings: Settings) -> bool:
    entity_type = str(getattr(item, "entity_type", "") or _dict_value(item, "entity_type") or "").lower()
    if entity_type not in {"disk_file", "disk_folder", "task_attachment"}:
        return False
    metadata = getattr(item, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = _dict_value(item, "metadata") or {}
    path = str(metadata.get("path") or "")
    if _path_has_private_marker(path, settings=settings):
        return True
    title = str(getattr(item, "title", "") or _dict_value(item, "title") or "")
    return entity_type == "disk_folder" and _matches_private_marker(title, settings=settings)


def can_user_see_portal_item(item: Any, *, user_id: int | None, settings: Settings) -> bool:
    return not user_has_private_disk_restrictions(user_id, settings=settings) or not is_private_disk_item(
        item, settings=settings
    )


def filter_portal_items_for_user(items: list[Any], *, user_id: int | None, settings: Settings) -> list[Any]:
    return (
        items
        if not user_has_private_disk_restrictions(user_id, settings=settings)
        else [item for item in items if not is_private_disk_item(item, settings=settings)]
    )


def _path_has_private_marker(path: str, *, settings: Settings) -> bool:
    components = [part.strip().casefold() for part in path.replace("\\", "/").split("/") if part.strip()]
    markers = settings.resolved_agent_private_disk_path_markers
    return any(marker.strip().casefold() in components for marker in markers if marker.strip())


def _matches_private_marker(value: str, *, settings: Settings) -> bool:
    normalized = value.strip().casefold()
    return bool(normalized) and any(
        normalized == marker.strip().casefold()
        for marker in settings.resolved_agent_private_disk_path_markers
        if marker.strip()
    )


def _dict_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None
