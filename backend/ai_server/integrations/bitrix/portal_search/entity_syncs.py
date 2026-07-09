from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search.file_cache import delete_portal_file_cache_path, portal_file_cache_path
from ai_server.integrations.bitrix.portal_search.search_index import PortalSearchIndex
from ai_server.integrations.bitrix.portal_search.text_utils import (
    normalize_url,
    safe_int,
    to_str,
)
from ai_server.integrations.bitrix.portal_search.types import PortalSearchResult
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ

_BITRIX_PAIRED_TAG_RE = re.compile(
    r"\[(USER|URL|B|I|U|S|QUOTE|CODE|COLOR|SIZE)[^\]]*\](.*?)\[/\1\]", re.IGNORECASE | re.DOTALL
)
_BITRIX_SINGLE_TAG_RE = re.compile(r"\[/?[A-Z][A-Z0-9_]*(?:=[^\]]*)?\]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _attachment_ids(value: object) -> list[int]:
    if value is None or value == "":
        return []
    raw_items = value if isinstance(value, list) else [value]
    ids = []
    for item in raw_items:
        normalized = str(item).strip().removeprefix("n")
        if normalized.isdigit():
            ids.append(int(normalized))
    return ids


async def _task_comment_texts(bitrix: BitrixClient, task_id: object, *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    try:
        result = await bitrix.result("task.commentitem.getlist", {"TASKID": task_id})
    except Exception:
        return []
    texts: list[str] = []
    for comment in _extract_comments(result):
        text = _comment_text(comment)
        if not text or _is_system_comment_text(text):
            continue
        texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _extract_comments(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ("comments", "COMMENTS", "items", "result"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _extract_comments(value)
                if nested:
                    return nested
    return []


def _comment_text(comment: dict[str, Any]) -> str:
    return _clean_bitrix_text(
        to_str(_first(comment, "POST_MESSAGE", "POST_MESSAGE_HTML", "POST_MESSAGE_TEXT", "text", "message")) or ""
    )


def _clean_bitrix_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    previous = None
    while previous != text:
        previous = text
        text = _BITRIX_PAIRED_TAG_RE.sub(r"\2", text)
    text = _BITRIX_SINGLE_TAG_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    return " ".join(text.split())


def _is_system_comment_text(text: str) -> bool:
    normalized = text.casefold().strip().rstrip(".")
    if not normalized:
        return True
    if normalized.startswith("крайний срок изменен на:"):
        return True
    if normalized in {
        "задача завершена",
        "задача возвращена в работу",
        "задача почти просрочена",
    }:
        return True
    system_fragments = (
        "вы добавлены наблюдателем",
        "вы назначены исполнителем",
        "задача почти просрочена",
        "завершите задачу или передвиньте срок",
    )
    return any(fragment in normalized for fragment in system_fragments)


def _person_label(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    name = to_str(_first(value, "name", "NAME"))
    if name:
        return name
    parts = [
        to_str(_first(value, "lastName", "LAST_NAME", "last_name")),
        to_str(_first(value, "name", "NAME", "firstName", "FIRST_NAME")),
        to_str(_first(value, "secondName", "SECOND_NAME", "second_name")),
    ]
    return " ".join(part for part in parts if part)


async def sync_disk_file_item(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    file_id: int,
    preserve_content: bool = True,
    settings: Settings,
) -> PortalSearchResult | None:
    file_data = await bitrix.get_disk_file(file_id)
    if not isinstance(file_data, dict):
        return None

    item_id = _first(file_data, "ID", "id") or file_id
    name = str(_first(file_data, "NAME", "name") or f"Файл #{item_id}")
    item_type = str(_first(file_data, "TYPE", "type") or "file").lower()
    detail_url = normalize_url(to_str(_first(file_data, "DETAIL_URL", "detailUrl")))
    storage_name = to_str(_first(file_data, "STORAGE_NAME", "storageName"))
    path = to_str(_first(file_data, "PATH", "path"))
    storage_id = _first(file_data, "STORAGE_ID", "storageId")
    parent_id = _first(file_data, "PARENT_ID", "parentId")
    update_time = to_str(_first(file_data, "UPDATE_TIME", "updateTime", "UPDATED_TIME", "updatedTime"))

    body_parts = [
        f"Диск: {storage_name}" if storage_name else "",
        f"Путь: {path}" if path else "",
        f"Тип: {item_type}",
        f"Хранилище ID: {storage_id}" if storage_id else "",
        f"Папка ID: {parent_id}" if parent_id else "",
    ]
    index.upsert_item(
        entity_type="disk_file",
        entity_id=item_id,
        title=name,
        body="\n".join(part for part in body_parts if part),
        url=detail_url or _disk_object_url(item_id, settings),
        metadata={
            "type": item_type,
            "path": path,
            "storage_name": storage_name,
            "storage_id": storage_id,
            "parent_id": parent_id,
            "disk_object_id": item_id,
            "detail_url": detail_url,
            "size": _first(file_data, "SIZE", "size"),
            "created_by": _first(file_data, "CREATED_BY", "createdBy"),
            "updated_by": _first(file_data, "UPDATED_BY", "updatedBy"),
            "webhook_synced_at": datetime.now(MOSCOW_TZ).isoformat(),
        },
        source_updated_at=update_time,
        preserve_content=preserve_content,
    )
    return index.get_item(entity_type="disk_file", entity_id=item_id)


async def _sync_catalog(bitrix: BitrixClient, index: PortalSearchIndex, settings: Settings) -> dict[str, int]:
    products_count = 0
    stores_count = 0

    try:
        catalogs = await bitrix.list_catalogs()
    except Exception:
        catalogs = []

    for catalog in catalogs:
        iblock_id = _first(catalog, "iblockId", "IBLOCK_ID")
        if iblock_id is None:
            continue
        try:
            products = await bitrix.list_catalog_products(
                int(iblock_id), limit=settings.search_index_max_catalog_products
            )
        except Exception:
            continue
        for product in products:
            product_id = _first(product, "id", "ID")
            if product_id is None:
                continue
            name = str(_first(product, "name", "NAME") or f"Товар #{product_id}")
            body_parts = [
                str(_first(product, "previewText", "PREVIEW_TEXT") or ""),
                str(_first(product, "detailText", "DETAIL_TEXT") or ""),
                f"Каталог iblockId:{iblock_id}",
            ]
            index.upsert_item(
                entity_type="catalog_product",
                entity_id=product_id,
                title=name,
                body="\n".join(p for p in body_parts if p.strip()),
                url=_catalog_product_url(iblock_id, product_id, settings),
                metadata={"iblock_id": iblock_id},
            )
            products_count += 1

    try:
        stores = await bitrix.list_catalog_stores()
    except Exception:
        stores = []

    for store in stores:
        store_id = _first(store, "id", "ID")
        if store_id is None:
            continue
        title = str(_first(store, "title", "TITLE") or f"Склад #{store_id}")
        address = str(_first(store, "address", "ADDRESS") or "")
        description = str(_first(store, "description", "DESCRIPTION") or "")
        index.upsert_item(
            entity_type="catalog_store",
            entity_id=store_id,
            title=title,
            body="\n".join(p for p in [address, description] if p.strip()),
            url="",
            metadata={
                "active": _first(store, "active", "ACTIVE"),
                "is_default": _first(store, "isDefault", "IS_DEFAULT"),
            },
        )
        stores_count += 1

    return {"products": products_count, "stores": stores_count}


async def _sync_tasks(bitrix: BitrixClient, index: PortalSearchIndex, settings: Settings) -> dict[str, object]:
    tasks = await bitrix.list_all_tasks(
        select=[
            "ID",
            "TITLE",
            "DESCRIPTION",
            "STATUS",
            "RESPONSIBLE_ID",
            "RESPONSIBLE",
            "CREATED_BY",
            "CREATOR",
            "GROUP_ID",
            "DEADLINE",
            "CREATED_DATE",
            "CHANGED_DATE",
            "CLOSED_DATE",
            "ACCOMPLICES",
            "AUDITORS",
            "UF_TASK_WEBDAV_FILES",
        ],
        order={"CHANGED_DATE": "DESC"},
        limit=settings.search_index_max_tasks,
    )
    indexed_attachments = 0
    seen_attachments: set[int] = set()
    for task in tasks:
        task_id = _first(task, "id", "ID")
        if task_id is None:
            continue
        title = str(_first(task, "title", "TITLE") or "Без названия")
        comments = (
            await _task_comment_texts(bitrix, task_id, limit=settings.search_index_task_comment_limit)
            if settings.search_index_include_task_comments
            else []
        )
        responsible_label = _person_label(_first(task, "responsible", "RESPONSIBLE"))
        creator_label = _person_label(_first(task, "creator", "CREATOR"))
        responsible = responsible_label or to_str(_first(task, "responsibleId", "RESPONSIBLE_ID"))
        creator = creator_label or to_str(_first(task, "createdBy", "CREATED_BY"))
        body_parts = [
            str(_first(task, "description", "DESCRIPTION") or ""),
            f"Статус: {_first(task, 'status', 'STATUS')}",
            f"Исполнитель: {responsible}" if responsible else "",
            f"Постановщик: {creator}" if creator else "",
            f"Проект: {_first(task, 'groupId', 'GROUP_ID')}",
            f"Срок: {_first(task, 'deadline', 'DEADLINE')}",
            f"Дата создания: {_first(task, 'createdDate', 'CREATED_DATE')}",
            f"Дата закрытия: {_first(task, 'closedDate', 'CLOSED_DATE')}",
            "Комментарии:\n" + "\n".join(f"- {comment}" for comment in comments) if comments else "",
        ]
        index.upsert_item(
            entity_type="task",
            entity_id=task_id,
            title=title,
            body="\n".join(part for part in body_parts if str(part).strip()),
            url=_task_url(task_id, settings),
            metadata={
                "status": _first(task, "status", "STATUS"),
                "responsible_id": _first(task, "responsibleId", "RESPONSIBLE_ID"),
                "responsible_label": responsible_label,
                "created_by": _first(task, "createdBy", "CREATED_BY"),
                "creator_label": creator_label,
                "group_id": _first(task, "groupId", "GROUP_ID"),
                "deadline": _first(task, "deadline", "DEADLINE"),
                "created_date": _first(task, "createdDate", "CREATED_DATE"),
                "changed_date": _first(task, "changedDate", "CHANGED_DATE"),
                "closed_date": _first(task, "closedDate", "CLOSED_DATE"),
                "accomplices": _first(task, "accomplices", "ACCOMPLICES"),
                "auditors": _first(task, "auditors", "AUDITORS"),
                "comments_indexed": bool(comments),
                "comments_count": len(comments),
            },
            source_updated_at=to_str(_first(task, "changedDate", "CHANGED_DATE")),
        )

        if not settings.search_index_include_task_attachments:
            continue
        if indexed_attachments >= settings.search_index_max_task_attachments:
            continue
        attachment_ids_list = _attachment_ids(_first(task, "ufTaskWebdavFiles", "UF_TASK_WEBDAV_FILES"))
        for attached_object_id in attachment_ids_list:
            if attached_object_id in seen_attachments:
                continue
            seen_attachments.add(attached_object_id)
            try:
                attached = await bitrix.get_attached_object(attached_object_id)
            except Exception:
                continue
            if isinstance(attached, dict):
                _index_task_attachment(
                    index,
                    attached=attached,
                    task_id=task_id,
                    task_title=title,
                    task_updated_at=to_str(_first(task, "changedDate", "CHANGED_DATE")),
                    settings=settings,
                )
                indexed_attachments += 1
            if indexed_attachments >= settings.search_index_max_task_attachments:
                break
    return {
        "tasks": len(tasks),
        "attachments": indexed_attachments,
        "tasks_complete": len(tasks) < settings.search_index_max_tasks,
        "attachments_complete": (
            not settings.search_index_include_task_attachments
            or indexed_attachments < settings.search_index_max_task_attachments
        ),
    }


async def _sync_projects(bitrix: BitrixClient, index: PortalSearchIndex, settings: Settings) -> int:
    projects = await bitrix.search_projects("", limit=settings.search_index_max_projects)
    if not isinstance(projects, list):
        return 0
    for project in projects:
        project_id = _first(project, "ID", "id")
        if project_id is None:
            continue
        name = str(_first(project, "NAME", "name") or f"Проект #{project_id}")
        index.upsert_item(
            entity_type="project",
            entity_id=project_id,
            title=name,
            body="\n".join(
                str(value)
                for value in (
                    _first(project, "DESCRIPTION", "description"),
                    f"Проект: {name}",
                    f"Владелец: {_first(project, 'OWNER_ID', 'ownerId')}",
                )
                if value
            ),
            url=_project_url(project_id, settings),
            metadata={
                "owner_id": _first(project, "OWNER_ID", "ownerId"),
                "active": _first(project, "ACTIVE", "active"),
                "project": _first(project, "PROJECT", "project"),
            },
            source_updated_at=to_str(_first(project, "DATE_UPDATE", "dateUpdate")),
        )
    return len(projects)


async def _sync_disk(bitrix: BitrixClient, index: PortalSearchIndex, settings: Settings) -> dict[str, object]:
    storages = await bitrix.list_disk_storages(limit=settings.search_index_max_storages)
    indexed_items = 0
    for storage in storages:
        storage_id = _first(storage, "ID", "id")
        root_id = _first(storage, "ROOT_OBJECT_ID", "rootObjectId")
        name = str(_first(storage, "NAME", "name") or f"Диск #{storage_id}")
        if storage_id is None:
            continue

        index.upsert_item(
            entity_type="disk_storage",
            entity_id=storage_id,
            title=name,
            body=f"Хранилище Bitrix Disk: {name}",
            url="",
            metadata={
                "storage_id": storage_id,
                "root_object_id": root_id,
                "entity_type": _first(storage, "ENTITY_TYPE", "entityType"),
                "entity_id": _first(storage, "ENTITY_ID", "entityId"),
            },
        )
        indexed_items += 1

        if root_id is None or indexed_items >= settings.search_index_max_disk_items:
            continue
        indexed_items += await _sync_disk_folder(
            bitrix,
            index,
            folder_id=int(root_id),
            storage_name=name,
            path=name,
            depth=0,
            remaining=settings.search_index_max_disk_items - indexed_items,
            settings=settings,
        )
        if indexed_items >= settings.search_index_max_disk_items:
            break
    return {
        "storages": len(storages),
        "items": indexed_items,
        "complete": (
            len(storages) < settings.search_index_max_storages and indexed_items < settings.search_index_max_disk_items
        ),
    }


async def _sync_disk_folder(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    folder_id: int,
    storage_name: str,
    path: str,
    depth: int,
    remaining: int,
    settings: Settings,
) -> int:
    if remaining <= 0 or depth > settings.search_index_disk_max_depth:
        return 0

    children = await bitrix.list_disk_folder_children_all(folder_id=folder_id, limit=remaining)
    count = 0
    for child in children:
        item_id = _first(child, "ID", "id")
        if item_id is None:
            continue
        name = str(_first(child, "NAME", "name") or f"Объект #{item_id}")
        item_type = str(_first(child, "TYPE", "type") or "").lower()
        child_path = f"{path}/{name}"
        entity_type = "disk_folder" if item_type == "folder" else "disk_file"
        detail_url = normalize_url(to_str(_first(child, "DETAIL_URL", "detailUrl")))
        index.upsert_item(
            entity_type=entity_type,
            entity_id=item_id,
            title=name,
            body="\n".join(
                str(value)
                for value in (
                    f"Диск: {storage_name}",
                    f"Путь: {child_path}",
                    f"Тип: {item_type}",
                )
                if value
            ),
            url=detail_url or _disk_object_url(item_id, settings),
            metadata={
                "type": item_type,
                "path": child_path,
                "storage_name": storage_name,
                "parent_id": folder_id,
                "disk_object_id": item_id,
                "detail_url": detail_url,
                "size": _first(child, "SIZE", "size"),
                "created_by": _first(child, "CREATED_BY", "createdBy"),
                "updated_by": _first(child, "UPDATED_BY", "updatedBy"),
            },
            source_updated_at=to_str(_first(child, "UPDATE_TIME", "updateTime")),
        )
        count += 1
        if entity_type == "disk_folder" and depth < settings.search_index_disk_max_depth:
            count += await _sync_disk_folder(
                bitrix,
                index,
                folder_id=int(item_id),
                storage_name=storage_name,
                path=child_path,
                depth=depth + 1,
                remaining=remaining - count,
                settings=settings,
            )
        if count >= remaining:
            break
    return count


async def _sync_disk_folder_delta(
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    *,
    folder_id: int,
    storage_name: str,
    path: str,
    child_limit: int,
    settings: Settings,
) -> dict[str, int]:
    children = await bitrix.list_disk_folder_children_all(folder_id=folder_id, limit=child_limit)
    seen_ids: set[str] = set()
    items_seen = 0
    items_changed = 0
    files_changed = 0
    folders_changed = 0

    for child in children:
        item_id = _first(child, "ID", "id")
        if item_id is None:
            continue
        seen_ids.add(str(item_id))
        items_seen += 1
        name = str(_first(child, "NAME", "name") or f"Объект #{item_id}")
        item_type = str(_first(child, "TYPE", "type") or "").lower()
        child_path = f"{path}/{name}" if path else name
        entity_type = "disk_folder" if item_type == "folder" else "disk_file"
        detail_url = normalize_url(to_str(_first(child, "DETAIL_URL", "detailUrl")))
        source_updated_at = to_str(_first(child, "UPDATE_TIME", "updateTime"))
        metadata = {
            "type": item_type,
            "path": child_path,
            "storage_name": storage_name,
            "parent_id": folder_id,
            "disk_object_id": item_id,
            "detail_url": detail_url,
            "size": _first(child, "SIZE", "size"),
            "created_by": _first(child, "CREATED_BY", "createdBy"),
            "updated_by": _first(child, "UPDATED_BY", "updatedBy"),
            "delta_synced_at": datetime.now(MOSCOW_TZ).isoformat(),
        }
        snapshot = index.item_snapshot(entity_type=entity_type, entity_id=item_id)
        changed = _disk_delta_item_changed(
            snapshot=snapshot,
            new_metadata=metadata,
            new_source_updated_at=source_updated_at,
        )
        index.upsert_item(
            entity_type=entity_type,
            entity_id=item_id,
            title=name,
            body="\n".join(
                str(value)
                for value in (
                    f"Диск: {storage_name}",
                    f"Путь: {child_path}",
                    f"Тип: {item_type}",
                )
                if value
            ),
            url=detail_url or _disk_object_url(item_id, settings),
            metadata=metadata,
            source_updated_at=source_updated_at,
        )
        if changed:
            items_changed += 1
            if entity_type == "disk_file":
                files_changed += 1
            else:
                folders_changed += 1

    deleted = 0
    if not child_limit or len(children) < child_limit:
        deleted = _delete_missing_delta_children(index, parent_id=folder_id, seen_ids=seen_ids, settings=settings)
    return {
        "items_seen": items_seen,
        "items_changed": items_changed,
        "files_changed": files_changed,
        "folders_changed": folders_changed,
        "deleted": deleted,
    }


def _index_task_attachment(
    index: PortalSearchIndex,
    *,
    attached: dict[str, Any],
    task_id: object,
    task_title: str,
    task_updated_at: str | None,
    settings: Settings,
) -> None:
    attached_id = _first(attached, "ID", "id")
    object_id = _first(attached, "OBJECT_ID", "objectId")
    name = str(_first(attached, "NAME", "name") or f"Вложение #{attached_id}")
    index.upsert_item(
        entity_type="task_attachment",
        entity_id=attached_id,
        title=name,
        body="\n".join(
            str(value)
            for value in (
                f"Вложение задачи: {task_title}",
                f"Задача: #{task_id}",
                f"Имя файла: {name}",
                f"Размер: {_first(attached, 'SIZE', 'size')}",
            )
            if value
        ),
        url=_task_url(task_id, settings),
        metadata={
            "task_id": task_id,
            "task_title": task_title,
            "attached_object_id": attached_id,
            "disk_object_id": object_id,
            "size": _first(attached, "SIZE", "size"),
            "created_by": _first(attached, "CREATED_BY", "createdBy"),
            "create_time": _first(attached, "CREATE_TIME", "createTime"),
            "download_available": bool(_first(attached, "DOWNLOAD_URL", "downloadUrl")),
        },
        source_updated_at=to_str(_first(attached, "CREATE_TIME", "createTime")) or task_updated_at,
    )


def _delete_missing_delta_children(
    index: PortalSearchIndex,
    *,
    parent_id: int,
    seen_ids: set[str],
    settings: Settings,
) -> int:
    deleted = 0
    for existing in index.children_by_parent_id(parent_id):
        if existing.entity_id in seen_ids:
            continue
        if existing.entity_type == "disk_file":
            delete_portal_file_cache_path(portal_file_cache_path(existing, settings), settings)
        if index.delete_item(entity_type=existing.entity_type, entity_id=existing.entity_id):
            deleted += 1
    return deleted


def _disk_delta_item_changed(
    *,
    snapshot: dict[str, Any] | None,
    new_metadata: dict[str, Any],
    new_source_updated_at: object,
) -> bool:
    if not snapshot:
        return True
    existing_source = to_str(snapshot.get("source_updated_at"))
    new_source = to_str(new_source_updated_at)
    if existing_source or new_source:
        return existing_source != new_source
    existing_metadata = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), dict) else {}
    return (
        safe_int(existing_metadata.get("size")) != safe_int(new_metadata.get("size"))
        or to_str(existing_metadata.get("path")) != to_str(new_metadata.get("path"))
        or to_str(existing_metadata.get("detail_url")) != to_str(new_metadata.get("detail_url"))
    )


def _delta_folder_id(folder: PortalSearchResult) -> int | None:
    if folder.entity_type == "disk_storage":
        return safe_int(folder.metadata.get("root_object_id"))
    return safe_int(folder.metadata.get("disk_object_id")) or safe_int(folder.entity_id)


def _delta_storage_name(folder: PortalSearchResult) -> str:
    return to_str(folder.metadata.get("storage_name")) or folder.title


def _delta_folder_path(folder: PortalSearchResult) -> str:
    return to_str(folder.metadata.get("path")) or folder.title


# ---------------------------------------------------------------------------
# URL builders (private, used only in this module)
# ---------------------------------------------------------------------------


def _portal_domain(settings: Settings) -> str:
    candidates = (
        settings.bitrix_domain,
        settings.bitrix_rest_webhook_url,
        settings.bitrix_projects_webhook_url,
    )
    for candidate in candidates:
        domain = _domain_from_value(candidate)
        if domain:
            return domain
    return ""


def _domain_from_value(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if "://" not in cleaned:
        cleaned = "https://" + cleaned
    parts = urlsplit(cleaned)
    return parts.netloc.strip().rstrip("/")


def _task_url(task_id: object, settings: Settings) -> str:
    domain = _portal_domain(settings)
    if not domain:
        return f"/company/personal/user/0/tasks/task/view/{task_id}/"
    return f"https://{domain}/company/personal/user/0/tasks/task/view/{task_id}/"


def _project_url(project_id: object, settings: Settings) -> str:
    domain = _portal_domain(settings)
    if not domain:
        return f"/workgroups/group/{project_id}/"
    return f"https://{domain}/workgroups/group/{project_id}/"


def _catalog_product_url(iblock_id: object, product_id: object, settings: Settings) -> str:
    domain = _portal_domain(settings)
    if not domain:
        return f"/crm/catalog/{iblock_id}/product/{product_id}/"
    return f"https://{domain}/crm/catalog/{iblock_id}/product/{product_id}/"


def _disk_object_url(object_id: object, settings: Settings) -> str:
    domain = _portal_domain(settings)
    if not domain:
        return f"/docs/file/{object_id}/"
    return f"https://{domain}/docs/file/{object_id}/"
