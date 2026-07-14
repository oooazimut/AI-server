from __future__ import annotations

from typing import Any


def agent_queue_partition_key(message: dict[str, Any]) -> str:
    """Return a stable partition key for agent queue concurrency control."""
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    routing = message.get("routing") if isinstance(message.get("routing"), dict) else {}
    context = payload.get("context") if isinstance(payload.get("context"), dict) else {}

    dialog_key = str(context.get("dialog_key") or routing.get("dialog_key") or "").strip()
    if dialog_key:
        return f"dialog:{dialog_key}"

    dialog_id = str(context.get("dialog_id") or routing.get("recipient_id") or "").strip()
    user_id = _payload_user_id(payload)
    if dialog_id and user_id:
        return f"dialog_id:{dialog_id}:user:{user_id}"
    if dialog_id:
        return f"dialog_id:{dialog_id}"

    task_id = str(payload.get("task_id") or "").strip()
    if task_id:
        return f"task:{task_id}"
    return ""


def _payload_user_id(payload: dict[str, Any]) -> str:
    user = payload.get("user")
    if isinstance(user, dict):
        return str(user.get("id") or "").strip()
    return ""
