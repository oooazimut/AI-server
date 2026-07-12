from __future__ import annotations

import html
import re
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode, urlsplit

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search.file_cache import delete_portal_file_cache_path, portal_file_cache_path
from ai_server.integrations.bitrix.portal_search.search_index import PortalSearchIndex
from ai_server.integrations.bitrix.portal_search.text_utils import (
    normalize_url,
    safe_int,
    to_str,
)
from ai_server.integrations.bitrix.portal_search.types import PortalSearchResult
from ai_server.integrations.bitrix.task_close_reports import (
    TASK_CLOSE_INCOMPLETE_MARKER as _TASK_CLOSE_INCOMPLETE_MARKER,
)
from ai_server.integrations.bitrix.task_close_reports import (
    TASK_CLOSE_REPORT_FILE_RE as _TASK_CLOSE_REPORT_FILE_RE,
)
from ai_server.integrations.bitrix.task_close_reports import (
    restore_task_close_report_file,
    task_close_report_key,
    task_close_report_problem_types,
    task_close_report_records,
    task_close_report_state_key,
)
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ

_BITRIX_PAIRED_TAG_RE = re.compile(
    r"\[(USER|URL|B|I|U|S|QUOTE|CODE|COLOR|SIZE)[^\]]*\](.*?)\[/\1\]", re.IGNORECASE | re.DOTALL
)
_BITRIX_SINGLE_TAG_RE = re.compile(r"\[/?[A-Z][A-Z0-9_]*(?:=[^\]]*)?\]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BITRIX_BATCH_COMMENT_SIZE = 50
_BITRIX_TASK_RESULT_LIMIT = 10
_TASK_CLOSE_REPORT_INCIDENT_STATUS_PENDING = "pending"
_TASK_CLOSE_REPORT_INCIDENT_STATUS_RESTORED = "restored"
_TASK_CLOSE_REPORT_INCIDENT_STATUS_ACCEPTED_MISSING = "accepted_missing"


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
    return _comment_texts_from_result(result, limit=limit)


async def _task_comments_by_id(
    bitrix: BitrixClient,
    task_ids: list[object],
    *,
    limit: int,
) -> dict[str, list[str]]:
    if limit <= 0:
        return {}
    unique_ids = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id is not None))
    comments_by_id: dict[str, list[str]] = {}
    for offset in range(0, len(unique_ids), _BITRIX_BATCH_COMMENT_SIZE):
        chunk = unique_ids[offset : offset + _BITRIX_BATCH_COMMENT_SIZE]
        cmd = {
            f"task_{index}": "task.commentitem.getlist?" + urlencode({"TASKID": task_id})
            for index, task_id in enumerate(chunk)
        }
        try:
            batch_result = await bitrix.result("batch", {"halt": 0, "cmd": cmd})
        except Exception:
            for task_id in chunk:
                comments_by_id[task_id] = await _task_comment_texts(bitrix, task_id, limit=limit)
            continue

        results = batch_result.get("result") if isinstance(batch_result, dict) else None
        if not isinstance(results, dict):
            for task_id in chunk:
                comments_by_id[task_id] = []
            continue
        for index, task_id in enumerate(chunk):
            comments_by_id[task_id] = _comment_texts_from_result(results.get(f"task_{index}"), limit=limit)
    return comments_by_id


async def _task_results_by_id(
    bitrix: BitrixClient,
    task_ids: list[object],
    *,
    limit: int,
) -> dict[str, list[str]]:
    if limit <= 0:
        return {}
    unique_ids = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id is not None))
    results_by_id: dict[str, list[str]] = {}
    for offset in range(0, len(unique_ids), _BITRIX_BATCH_COMMENT_SIZE):
        chunk = unique_ids[offset : offset + _BITRIX_BATCH_COMMENT_SIZE]
        cmd = {
            f"task_result_{index}": "tasks.task.result.list?" + urlencode({"taskId": task_id})
            for index, task_id in enumerate(chunk)
        }
        try:
            batch_result = await bitrix.result("batch", {"halt": 0, "cmd": cmd})
        except Exception:
            for task_id in chunk:
                results_by_id[task_id] = await _task_result_texts(bitrix, task_id, limit=limit)
            continue

        results = batch_result.get("result") if isinstance(batch_result, dict) else None
        if not isinstance(results, dict):
            for task_id in chunk:
                results_by_id[task_id] = []
            continue
        for index, task_id in enumerate(chunk):
            results_by_id[task_id] = _task_result_texts_from_result(results.get(f"task_result_{index}"), limit=limit)
    return results_by_id


async def _task_result_texts(bitrix: BitrixClient, task_id: object, *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    try:
        result = await bitrix.result("tasks.task.result.list", {"taskId": task_id})
    except Exception:
        return []
    return _task_result_texts_from_result(result, limit=limit)


def _task_result_texts_from_result(result: Any, *, limit: int) -> list[str]:
    texts: list[str] = []
    for item in _extract_task_results(result):
        text = _task_result_text(item)
        if not text:
            continue
        texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _extract_task_results(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        for key in ("results", "RESULTS", "items", "result"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
            if isinstance(value, dict):
                nested = _extract_task_results(value)
                if nested:
                    return nested
    return []


def _task_result_text(item: dict[str, Any]) -> str:
    return _clean_bitrix_comment_text(
        to_str(_first(item, "TEXT", "text", "POST_MESSAGE", "message", "comment", "COMMENT")) or ""
    )


def _comment_texts_from_result(result: Any, *, limit: int) -> list[str]:
    texts: list[str] = []
    for comment in _extract_comments(result):
        text = _comment_text(comment)
        if not text:
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
    return _clean_bitrix_comment_text(
        to_str(_first(comment, "POST_MESSAGE", "POST_MESSAGE_HTML", "POST_MESSAGE_TEXT", "text", "message")) or ""
    )


def _task_close_marker_metadata(
    *,
    task_results: list[str],
    comments: list[str],
    attachment_names: list[str] | None = None,
) -> dict[str, Any]:
    texts = [*task_results, *comments]
    marker_texts = [text for text in texts if _TASK_CLOSE_INCOMPLETE_MARKER in text]
    attachment_problem_types = _task_close_report_problem_types(attachment_names or [])
    problem_types: list[str] = []
    marker_blob = "\n".join(marker_texts).casefold()
    if marker_texts and any(marker in marker_blob for marker in ("невыполн", "не выполн", "not_done", "not done")):
        problem_types.append("not_done")
    if marker_texts and (
        any(marker in marker_blob for marker in ("неподтверж", "не подтверж", "непровер", "unknown", "unconfirmed"))
        or not problem_types
    ):
        problem_types.append("unconfirmed")
    problem_types = _unique_problem_types([*problem_types, *attachment_problem_types])
    has_marker = bool(marker_texts) or bool(problem_types)
    return {
        "ai_close_incomplete": has_marker,
        "ai_close_marker": _TASK_CLOSE_INCOMPLETE_MARKER if has_marker else "",
        "ai_close_problem_types": problem_types,
        "ai_close_has_not_done": "not_done" in problem_types,
        "ai_close_has_unconfirmed": "unconfirmed" in problem_types,
        "ai_close_marker_source": _task_close_marker_source(
            task_results=task_results,
            comments=comments,
            attachment_names=attachment_names or [],
        ),
    }


def _task_close_marker_source(*, task_results: list[str], comments: list[str], attachment_names: list[str]) -> str:
    if any(_TASK_CLOSE_INCOMPLETE_MARKER in text for text in task_results):
        return "task_result"
    if any(_TASK_CLOSE_INCOMPLETE_MARKER in text for text in comments):
        return "comment"
    if _task_close_report_problem_types(attachment_names):
        return "task_attachment"
    return ""


def _task_close_report_problem_types(attachment_names: list[str]) -> list[str]:
    return task_close_report_problem_types(attachment_names)


def _unique_problem_types(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in {"not_done", "unconfirmed"} or value in result:
            continue
        result.append(value)
    return result


def _task_close_report_is_accepted_missing(
    index: PortalSearchIndex, *, task_id: int, file_record: dict[str, Any]
) -> bool:
    state = _task_close_report_state(index, task_id=task_id, file_record=file_record)
    return str((state or {}).get("status") or "") == _TASK_CLOSE_REPORT_INCIDENT_STATUS_ACCEPTED_MISSING


def _task_close_report_state(
    index: PortalSearchIndex, *, task_id: int, file_record: dict[str, Any]
) -> dict[str, Any] | None:
    getter = getattr(index, "get_task_close_processing_state", None)
    if not callable(getter):
        return None
    state = getter(task_id=task_id, state_key=task_close_report_state_key(file_record))
    return state if isinstance(state, dict) else None


def _upsert_task_close_report_state(
    index: PortalSearchIndex,
    *,
    task_id: int,
    file_record: dict[str, Any],
    status: str,
    payload: dict[str, Any] | None = None,
    actor_user_id: int | None = None,
) -> None:
    upsert = getattr(index, "upsert_task_close_processing_state", None)
    if not callable(upsert):
        return
    upsert(
        task_id=task_id,
        state_key=task_close_report_state_key(file_record),
        status=status,
        payload=payload,
        actor_user_id=actor_user_id,
    )


async def _task_close_report_integrity_metadata(
    *,
    bitrix: BitrixClient,
    index: PortalSearchIndex,
    settings: Settings,
    task_id: object,
    task_title: str,
    current_report_files: list[dict[str, Any]],
) -> dict[str, Any]:
    task_id_int = safe_int(task_id)
    if task_id_int is None:
        return {"ai_close_report_files": current_report_files}

    previous = index.item_snapshot(entity_type="task", entity_id=task_id_int)
    previous_metadata = previous.get("metadata") if isinstance(previous, dict) else {}
    if not isinstance(previous_metadata, dict):
        previous_metadata = {}
    previous_report_files = _dict_list(previous_metadata.get("ai_close_report_files"))
    if not previous_report_files:
        return {"ai_close_report_files": current_report_files}

    previous_report_files = [
        record
        for record in previous_report_files
        if not _task_close_report_is_accepted_missing(index, task_id=task_id_int, file_record=record)
    ]
    if not previous_report_files:
        return {"ai_close_report_files": current_report_files}

    current_keys = {task_close_report_key(record) for record in current_report_files}
    missing_files = [record for record in previous_report_files if task_close_report_key(record) not in current_keys]
    if not missing_files:
        return {"ai_close_report_files": current_report_files}

    now = datetime.now(MOSCOW_TZ)
    now_iso = now.isoformat(timespec="seconds")
    detected_at = str(previous_metadata.get("ai_close_report_detected_at") or now_iso)
    auto_restore_after = str(
        previous_metadata.get("ai_close_report_auto_restore_after")
        or _task_close_report_auto_restore_after(detected_at, settings)
    )
    incident_key = str(
        previous_metadata.get("ai_close_report_incident_key")
        or _task_close_report_incident_key(task_id=task_id_int, missing_files=missing_files)
    )
    metadata = {
        "ai_close_report_files": previous_report_files,
        "ai_close_report_missing": True,
        "ai_close_report_missing_files": missing_files,
        "ai_close_report_incident_status": _TASK_CLOSE_REPORT_INCIDENT_STATUS_PENDING,
        "ai_close_report_incident_key": incident_key,
        "ai_close_report_detected_at": detected_at,
        "ai_close_report_auto_restore_after": auto_restore_after,
    }
    for file_record in missing_files:
        _upsert_task_close_report_state(
            index,
            task_id=task_id_int,
            file_record=file_record,
            status=_TASK_CLOSE_REPORT_INCIDENT_STATUS_PENDING,
            payload={
                "task_title": task_title,
                "detected_at": detected_at,
                "auto_restore_after": auto_restore_after,
                "file": file_record,
            },
        )

    if _should_auto_restore_task_close_report(auto_restore_after, now=now):
        restored_files: list[dict[str, Any]] = []
        restore_errors: list[str] = []
        for file_record in missing_files:
            try:
                restore_result = await restore_task_close_report_file(
                    bitrix,
                    task_id=task_id_int,
                    file_record=file_record,
                    max_bytes=settings.attachment_max_bytes,
                )
                restored_file = restore_result.get("restored_file")
                if isinstance(restored_file, dict):
                    restored_files.append(restored_file)
                    _upsert_task_close_report_state(
                        index,
                        task_id=task_id_int,
                        file_record=file_record,
                        status=_TASK_CLOSE_REPORT_INCIDENT_STATUS_RESTORED,
                        payload={"task_title": task_title, "restored_at": now_iso, "restored_file": restored_file},
                    )
            except Exception as exc:
                restore_errors.append(f"{file_record.get('name')}: {type(exc).__name__}: {exc}")
        if restored_files and not restore_errors:
            return {
                "ai_close_report_files": [*current_report_files, *restored_files],
                "ai_close_report_missing": False,
                "ai_close_report_missing_files": [],
                "ai_close_report_incident_status": _TASK_CLOSE_REPORT_INCIDENT_STATUS_RESTORED,
                "ai_close_report_incident_key": incident_key,
                "ai_close_report_detected_at": detected_at,
                "ai_close_report_restored_at": now_iso,
            }
        metadata["ai_close_report_restore_error"] = "; ".join(restore_errors) if restore_errors else "not restored"

    if not previous_metadata.get("ai_close_report_alert_sent_at"):
        alert_error = await _notify_task_close_report_missing(
            bitrix,
            settings=settings,
            task_id=task_id_int,
            task_title=task_title,
            missing_files=missing_files,
            auto_restore_after=auto_restore_after,
        )
        if alert_error:
            metadata["ai_close_report_alert_error"] = alert_error
        else:
            metadata["ai_close_report_alert_sent_at"] = now_iso
    else:
        metadata["ai_close_report_alert_sent_at"] = previous_metadata.get("ai_close_report_alert_sent_at")
    return metadata


def _task_close_marker_metadata_from_report_files(report_files: list[dict[str, Any]]) -> dict[str, Any] | None:
    problem_types = _unique_problem_types(
        [
            problem_type
            for record in report_files
            for problem_type in _string_list_for_metadata(record.get("problem_types"))
        ]
    )
    if not problem_types:
        return None
    return {
        "ai_close_incomplete": True,
        "ai_close_marker": _TASK_CLOSE_INCOMPLETE_MARKER,
        "ai_close_problem_types": problem_types,
        "ai_close_has_not_done": "not_done" in problem_types,
        "ai_close_has_unconfirmed": "unconfirmed" in problem_types,
        "ai_close_marker_source": "task_attachment",
    }


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _string_list_for_metadata(value: object) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _task_close_report_incident_key(*, task_id: int, missing_files: list[dict[str, Any]]) -> str:
    keys = ",".join(sorted(task_close_report_key(record) for record in missing_files))
    return f"task:{task_id}:ai-close-report-missing:{keys}"


def _task_close_report_auto_restore_after(detected_at: str, settings: Settings) -> str:
    detected_dt = _parse_datetime(detected_at) or datetime.now(MOSCOW_TZ)
    hours = max(int(settings.bitrix_task_close_report_auto_restore_hours), 0)
    return (detected_dt + timedelta(hours=hours)).isoformat(timespec="seconds")


def _should_auto_restore_task_close_report(auto_restore_after: str, *, now: datetime) -> bool:
    due_at = _parse_datetime(auto_restore_after)
    return bool(due_at and now >= due_at)


def _parse_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=MOSCOW_TZ)
    return parsed.astimezone(MOSCOW_TZ)


async def _notify_task_close_report_missing(
    bitrix: BitrixClient,
    *,
    settings: Settings,
    task_id: int,
    task_title: str,
    missing_files: list[dict[str, Any]],
    auto_restore_after: str,
) -> str:
    message = _task_close_report_missing_message(
        task_id=task_id,
        task_title=task_title,
        missing_files=missing_files,
        auto_restore_after=auto_restore_after,
    )
    errors: list[str] = []
    for admin_id in settings.resolved_task_close_report_admin_user_ids:
        try:
            await bitrix.send_bot_message(str(admin_id), message)
            continue
        except Exception as send_exc:
            send_error = f"send_bot_message:{admin_id}:{type(send_exc).__name__}: {send_exc}"
        try:
            await bitrix.notify_user(
                user_id=admin_id,
                message=message,
                tag="ai_server",
                sub_tag=f"task_close_report_missing_{task_id}",
            )
            continue
        except Exception as exc:
            errors.append(f"{send_error}; notify_user:{admin_id}:{type(exc).__name__}: {exc}")
    return "; ".join(errors)


def _task_close_report_missing_message(
    *,
    task_id: int,
    task_title: str,
    missing_files: list[dict[str, Any]],
    auto_restore_after: str,
) -> str:
    file_names = ", ".join(str(record.get("name") or "") for record in missing_files if record.get("name"))
    return "\n".join(
        [
            "Контроль AI-закрытия: файл отчёта исчез из задачи.",
            f"Задача #{task_id}: {task_title or 'без названия'}",
            f"Файл: {file_names or 'AI-close report'}",
            "",
            "Выберите текстом:",
            "1 — восстановить файл в задаче",
            "2 — всё в порядке, удалить эти данные из индекса",
            f"Если ответа не будет, файл будет восстановлен автоматически после {auto_restore_after}.",
        ]
    )


def _clean_bitrix_comment_text(value: object) -> str:
    text = html.unescape(str(value or "")).replace("\r\n", "\n").replace("\r", "\n")
    previous = None
    while previous != text:
        previous = text
        text = _BITRIX_PAIRED_TAG_RE.sub(r"\2", text)
    text = _BITRIX_SINGLE_TAG_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line and not _is_system_comment_text(line))


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
        "вы назначены соисполнителем",
        "задача просрочена",
        "задача почти просрочена",
        "завершите задачу или передвиньте срок",
        "необходимо указать крайний срок",
        "необходимо принять задачу или отправить на доработку",
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
    stock_rows_count = 0
    products_by_id: dict[str, dict[str, Any]] = {}

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
            product_iblock_id = _first(product, "iblockId", "IBLOCK_ID", "iblock_id") or iblock_id
            name = str(_first(product, "name", "NAME") or f"Товар #{product_id}")
            product_url = _catalog_product_url(product_iblock_id, product_id, settings)
            products_by_id[str(product_id)] = {
                "id": product_id,
                "name": name,
                "iblock_id": product_iblock_id,
                "url": product_url,
            }
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
                url=product_url,
                metadata={"iblock_id": product_iblock_id},
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

        if stock_rows_count >= settings.search_index_max_catalog_stock_rows:
            continue
        remaining = settings.search_index_max_catalog_stock_rows - stock_rows_count
        try:
            stock_rows = await bitrix.list_catalog_store_products(store_id, limit=remaining)
        except Exception:
            continue
        for row in stock_rows:
            product_id = _first(row, "productId", "PRODUCT_ID", "product_id")
            if product_id is None:
                continue
            amount = _first(row, "amount", "AMOUNT", "quantity", "QUANTITY")
            if not _is_positive_number(amount):
                continue
            product = products_by_id.get(str(product_id))
            if not product:
                continue
            product_name = str(product.get("name") or "").strip()
            if not product_name:
                continue
            product_url = str(product.get("url") or "")
            stock_body = "\n".join(
                part
                for part in (
                    f"Store: {title}",
                    f"Address: {address}" if address else "",
                    f"Product: {product_name}",
                    f"Amount: {amount}",
                )
                if part
            )
            index.upsert_item(
                entity_type="catalog_store_stock",
                entity_id=f"{store_id}:{product_id}",
                title=f"{product_name} - {title}",
                body=stock_body,
                url=product_url,
                metadata={
                    "store_id": store_id,
                    "store_title": title,
                    "store_address": address,
                    "product_id": product_id,
                    "product_name": product_name,
                    "iblock_id": product.get("iblock_id"),
                    "amount": amount,
                    "product_url": product_url,
                    "positive_amount": True,
                },
            )
            stock_rows_count += 1
            if stock_rows_count >= settings.search_index_max_catalog_stock_rows:
                break

    return {"products": products_count, "stores": stores_count, "stock_rows": stock_rows_count}


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
    comments_by_task_id = (
        await _task_comments_by_id(
            bitrix,
            [_first(task, "id", "ID") for task in tasks],
            limit=settings.search_index_task_comment_limit,
        )
        if settings.search_index_include_task_comments
        else {}
    )
    results_by_task_id = await _task_results_by_id(
        bitrix,
        [_first(task, "id", "ID") for task in tasks],
        limit=_BITRIX_TASK_RESULT_LIMIT,
    )
    for task in tasks:
        task_id = _first(task, "id", "ID")
        if task_id is None:
            continue
        title = str(_first(task, "title", "TITLE") or "Без названия")
        comments = comments_by_task_id.get(str(task_id), [])
        task_results = results_by_task_id.get(str(task_id), [])
        task_attachments: list[dict[str, Any]] = []
        if (
            settings.search_index_include_task_attachments
            and indexed_attachments < settings.search_index_max_task_attachments
        ):
            attachment_ids_list = _attachment_ids(_first(task, "ufTaskWebdavFiles", "UF_TASK_WEBDAV_FILES"))
            for attached_object_id in attachment_ids_list:
                if indexed_attachments + len(task_attachments) >= settings.search_index_max_task_attachments:
                    break
                if attached_object_id in seen_attachments:
                    continue
                seen_attachments.add(attached_object_id)
                try:
                    attached = await bitrix.get_attached_object(attached_object_id)
                except Exception:
                    continue
                if isinstance(attached, dict):
                    task_attachments.append(attached)
        attachment_names = [str(_first(attached, "NAME", "name") or "") for attached in task_attachments]
        close_marker_metadata = _task_close_marker_metadata(
            task_results=task_results,
            comments=comments,
            attachment_names=attachment_names,
        )
        current_report_files = task_close_report_records(task_attachments)
        report_integrity_metadata = await _task_close_report_integrity_metadata(
            bitrix=bitrix,
            index=index,
            settings=settings,
            task_id=task_id,
            task_title=title,
            current_report_files=current_report_files,
        )
        effective_report_files = _dict_list(report_integrity_metadata.get("ai_close_report_files"))
        report_marker_metadata = _task_close_marker_metadata_from_report_files(effective_report_files)
        if report_marker_metadata:
            close_marker_metadata = {**close_marker_metadata, **report_marker_metadata}
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
            "Результаты:\n" + "\n".join(f"- {result}" for result in task_results) if task_results else "",
            "Комментарии:\n" + "\n".join(f"- {comment}" for comment in comments) if comments else "",
            f"AI close marker: {_TASK_CLOSE_INCOMPLETE_MARKER}" if close_marker_metadata["ai_close_incomplete"] else "",
            "AI close problem types: " + ", ".join(close_marker_metadata["ai_close_problem_types"])
            if close_marker_metadata["ai_close_problem_types"]
            else "",
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
                "task_results_indexed": bool(task_results),
                "task_results_count": len(task_results),
                **report_integrity_metadata,
                **close_marker_metadata,
            },
            source_updated_at=to_str(_first(task, "changedDate", "CHANGED_DATE")),
        )

        for attached in task_attachments:
            _index_task_attachment(
                index,
                attached=attached,
                task_id=task_id,
                task_title=title,
                task_updated_at=to_str(_first(task, "changedDate", "CHANGED_DATE")),
                settings=settings,
            )
            indexed_attachments += 1
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
    visited_folder_ids: set[int] = set()
    seen_disk_object_ids: set[int] = set()
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

        root_folder_id = safe_int(root_id)
        if root_folder_id is None or indexed_items >= settings.search_index_max_disk_items:
            continue
        if root_folder_id in visited_folder_ids:
            continue
        visited_folder_ids.add(root_folder_id)
        indexed_items += await _sync_disk_folder(
            bitrix,
            index,
            folder_id=root_folder_id,
            storage_name=name,
            path=name,
            depth=0,
            remaining=settings.search_index_max_disk_items - indexed_items,
            settings=settings,
            visited_folder_ids=visited_folder_ids,
            seen_disk_object_ids=seen_disk_object_ids,
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
    visited_folder_ids: set[int],
    seen_disk_object_ids: set[int],
) -> int:
    if remaining <= 0 or depth > settings.search_index_disk_max_depth:
        return 0

    children = await bitrix.list_disk_folder_children_all(folder_id=folder_id, limit=remaining)
    count = 0
    for child in children:
        item_id = _first(child, "ID", "id")
        if item_id is None:
            continue
        disk_object_id = safe_int(item_id)
        if disk_object_id is not None and disk_object_id in seen_disk_object_ids:
            continue
        if disk_object_id is not None:
            seen_disk_object_ids.add(disk_object_id)
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
            child_folder_id = safe_int(item_id)
            if child_folder_id is None or child_folder_id in visited_folder_ids:
                continue
            visited_folder_ids.add(child_folder_id)
            count += await _sync_disk_folder(
                bitrix,
                index,
                folder_id=child_folder_id,
                storage_name=storage_name,
                path=child_path,
                depth=depth + 1,
                remaining=remaining - count,
                settings=settings,
                visited_folder_ids=visited_folder_ids,
                seen_disk_object_ids=seen_disk_object_ids,
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
    close_problem_types = _task_close_report_problem_types([name])
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
            "ai_close_report": bool(_TASK_CLOSE_REPORT_FILE_RE.match(name)),
            "ai_close_incomplete": bool(close_problem_types),
            "ai_close_marker": _TASK_CLOSE_INCOMPLETE_MARKER if close_problem_types else "",
            "ai_close_problem_types": close_problem_types,
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


def _is_positive_number(value: object) -> bool:
    if isinstance(value, bool) or value in (None, ""):
        return False
    try:
        return float(str(value).replace(",", ".")) > 0
    except (TypeError, ValueError):
        return False


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
        return f"/shop/documents-catalog/{iblock_id}/product/{product_id}/"
    return f"https://{domain}/shop/documents-catalog/{iblock_id}/product/{product_id}/"


def _disk_object_url(object_id: object, settings: Settings) -> str:
    domain = _portal_domain(settings)
    if not domain:
        return f"/docs/file/{object_id}/"
    return f"https://{domain}/docs/file/{object_id}/"
