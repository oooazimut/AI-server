from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import Any

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixConfigError
from ai_server.integrations.bitrix.portal_search.text_utils import safe_int, to_str

TASK_CLOSE_INCOMPLETE_MARKER = "AI_SERVER_TASK_CLOSE_INCOMPLETE"
TASK_CLOSE_REPORT_FILE_RE = re.compile(
    r"^(?P<stem>AI-close-\d+-(?P<status>ok|partial|unconfirmed|failed))(?: \(\d+\))?\.txt$",
    re.IGNORECASE,
)


def is_task_close_report_file_name(name: object) -> bool:
    return bool(TASK_CLOSE_REPORT_FILE_RE.match(str(name or "").strip()))


def canonical_task_close_report_file_name(name: object) -> str:
    match = TASK_CLOSE_REPORT_FILE_RE.match(str(name or "").strip())
    if not match:
        return ""
    return f"{match.group('stem')}.txt"


def task_close_report_problem_types(attachment_names: list[str]) -> list[str]:
    problem_types: list[str] = []
    for name in attachment_names:
        match = TASK_CLOSE_REPORT_FILE_RE.match(str(name or "").strip())
        if not match:
            continue
        status = match.group("status").casefold()
        if status in {"partial", "failed"}:
            problem_types.append("not_done")
        if status == "unconfirmed":
            problem_types.append("unconfirmed")
    return _unique_problem_types(problem_types)


def task_close_report_record(attached: dict[str, Any]) -> dict[str, Any] | None:
    name = str(_first(attached, "NAME", "name") or "").strip()
    if not is_task_close_report_file_name(name):
        return None
    return {
        "name": name,
        "canonical_name": canonical_task_close_report_file_name(name),
        "attached_object_id": _first(attached, "ID", "id", "ATTACHMENT_ID", "attachmentId"),
        "disk_object_id": _first(attached, "OBJECT_ID", "objectId", "FILE_ID", "fileId"),
        "size": _first(attached, "SIZE", "size"),
        "created_by": _first(attached, "CREATED_BY", "createdBy"),
        "create_time": _first(attached, "CREATE_TIME", "createTime"),
        "problem_types": task_close_report_problem_types([name]),
    }


def task_close_report_records(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for attached in attachments:
        record = task_close_report_record(attached)
        if record is not None:
            records.append(record)
    return records


def task_close_report_key(record: dict[str, Any]) -> str:
    disk_object_id = to_str(record.get("disk_object_id"))
    if disk_object_id:
        return f"disk:{disk_object_id}"
    attached_object_id = to_str(record.get("attached_object_id"))
    if attached_object_id:
        return f"attached:{attached_object_id}"
    return f"name:{str(record.get('name') or '').casefold()}"


async def restore_task_close_report_file(
    bitrix: Any,
    *,
    task_id: int,
    file_record: dict[str, Any],
    max_bytes: int,
) -> dict[str, Any]:
    disk_object_id = safe_int(file_record.get("disk_object_id"))
    file_name = str(file_record.get("name") or "").strip()
    if disk_object_id is None:
        raise BitrixConfigError("Cannot restore AI close report: disk_object_id is missing.")
    if not is_task_close_report_file_name(file_name):
        raise BitrixConfigError("Cannot restore AI close report: file name is not an AI close report.")
    if not hasattr(bitrix, "get_disk_file_download_url") or not hasattr(bitrix, "download_file_from_url"):
        raise BitrixConfigError("Cannot restore AI close report: Bitrix file download client is not configured.")

    download_url = await bitrix.get_disk_file_download_url(disk_object_id)
    with tempfile.TemporaryDirectory(prefix="ai-close-report-") as tmp_dir:
        path = Path(tmp_dir) / file_name
        await bitrix.download_file_from_url(download_url, path, max_bytes=max_bytes)
        content = path.read_bytes()

    payload = {
        "taskId": task_id,
        "fileParameters": {
            "NAME": file_name,
            "CONTENT": base64.b64encode(content).decode("ascii"),
        },
    }
    add_result = await _bitrix_result(bitrix, "task.item.addfile", payload)
    return {
        "task_id": task_id,
        "file_name": file_name,
        "disk_object_id": disk_object_id,
        "add_result": add_result,
        "restored_file": _restored_record(file_record, add_result),
    }


async def _bitrix_result(bitrix: Any, method: str, payload: dict[str, Any]) -> Any:
    if hasattr(bitrix, "result"):
        return await bitrix.result(method, payload)
    if hasattr(bitrix, "call"):
        data = await bitrix.call(method, payload)
        return data.get("result") if isinstance(data, dict) and "result" in data else data
    raise BitrixApiError(method, "Bitrix REST client is not configured.")


def _restored_record(original: dict[str, Any], add_result: Any) -> dict[str, Any]:
    result = _first_dict(add_result)
    record = dict(original)
    if result:
        record["attached_object_id"] = _first(result, "ATTACHMENT_ID", "attachedObjectId", "ID", "id") or record.get(
            "attached_object_id"
        )
        record["disk_object_id"] = _first(result, "FILE_ID", "fileId", "OBJECT_ID", "objectId") or record.get(
            "disk_object_id"
        )
        record["name"] = _first(result, "NAME", "name") or record.get("name")
    return record


def _first_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        nested = value.get("result")
        if isinstance(nested, dict):
            return nested
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return None


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _unique_problem_types(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in {"not_done", "unconfirmed"} or value in result:
            continue
        result.append(value)
    return result
