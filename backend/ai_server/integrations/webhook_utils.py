"""Webhook event key and partition utilities.

Defined in integrations/ so that both workers/ and integrations/redis/
can import without cross-layer violations:
  workers → integrations ✓
  integrations/redis → integrations ✓
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

_SENSITIVE_WEBHOOK_KEYS = {
    "auth",
    "authorization",
    "access_token",
    "refresh_token",
    "application_token",
    "client_secret",
    "secret",
    "agent_secret",
    "token",
    "webhook_secret",
    "auth_id",
    "refresh_id",
}


def sanitize_webhook_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_value(payload)
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(str(key)):
                continue
            result[key] = _sanitize_value(item)
        return result
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in _SENSITIVE_WEBHOOK_KEYS


def webhook_event_partition_key(payload: dict[str, Any], *, event_type: str) -> str:
    normalized = str(event_type or "").upper()
    if normalized in {"ONIMBOTV2MESSAGEADD", "ONIMBOTMESSAGEADD"}:
        dialog_id = _extract_dialog_id(payload)
        if dialog_id:
            return f"dialog:{dialog_id}"
        message_id = _extract_message_id(payload)
        if message_id:
            return f"message:{message_id}"
        return "dialog:unknown"
    if normalized.startswith("ONTASK"):
        task_id = _extract_task_id(payload)
        return f"task:{task_id or 'unknown'}"
    if "DISK" in normalized:
        file_id = _extract_disk_file_id(payload)
        return f"disk-file:{file_id or 'unknown'}"
    return f"event:{normalized or 'unknown'}"


def webhook_event_key(payload: dict[str, Any], *, event_type: str, received_at: str) -> str:
    message_id = _extract_message_id(payload)
    if event_type in {"ONIMBOTV2MESSAGEADD", "ONIMBOTMESSAGEADD"} and message_id:
        return f"message:{message_id}"
    body = json.dumps(
        {"event": event_type, "payload": payload, "received_at": received_at},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _extract_dialog_id(payload: dict[str, Any]) -> str:
    data = _payload_data(payload)
    chat = data.get("chat") if isinstance(data.get("chat"), dict) else {}
    params = data.get("PARAMS") if isinstance(data.get("PARAMS"), dict) else {}
    value = (
        chat.get("dialogId")
        or chat.get("dialog_id")
        or params.get("DIALOG_ID")
        or params.get("TO_USER_ID")
        or payload.get("DIALOG_ID")
        or payload.get("dialog_id")
    )
    return str(value or "").strip()


def _extract_message_id(payload: dict[str, Any]) -> str:
    data = _payload_data(payload)
    message = data.get("message") if isinstance(data.get("message"), dict) else {}
    params = data.get("PARAMS") if isinstance(data.get("PARAMS"), dict) else {}
    value = message.get("id") or params.get("MESSAGE_ID")
    return str(value or "").strip()


def _extract_task_id(payload: dict[str, Any]) -> str:
    data = _payload_data(payload)
    fields = data.get("FIELDS") if isinstance(data.get("FIELDS"), dict) else {}
    fields_lower = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    task = data.get("task") if isinstance(data.get("task"), dict) else {}
    params = data.get("PARAMS") if isinstance(data.get("PARAMS"), dict) else {}
    value = (
        data.get("TASK_ID")
        or data.get("taskId")
        or fields.get("ID")
        or fields.get("TASK_ID")
        or fields_lower.get("id")
        or fields_lower.get("taskId")
        or task.get("id")
        or task.get("ID")
        or params.get("TASK_ID")
        or params.get("ID")
        or payload.get("TASK_ID")
    )
    return str(value or "").strip()


def _extract_disk_file_id(payload: dict[str, Any]) -> str:
    data = _payload_data(payload)
    fields = data.get("FIELDS") if isinstance(data.get("FIELDS"), dict) else {}
    fields_lower = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    file = data.get("file") if isinstance(data.get("file"), dict) else {}
    params = data.get("PARAMS") if isinstance(data.get("PARAMS"), dict) else {}
    value = (
        data.get("FILE_ID")
        or data.get("fileId")
        or fields.get("ID")
        or fields.get("FILE_ID")
        or fields_lower.get("id")
        or fields_lower.get("fileId")
        or file.get("id")
        or file.get("ID")
        or params.get("FILE_ID")
        or params.get("ID")
        or payload.get("FILE_ID")
    )
    return str(value or "").strip()


def _payload_data(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("data") if isinstance(payload.get("data"), dict) else payload
