from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.models import AgentTask, UserContext
from ai_server.settings import get_settings
from ai_server.utils import MOSCOW_TZ

logger = logging.getLogger(__name__)
TASK_QUALITY_WEBHOOK_EVENTS = {"ONTASKUPDATE"}
_WEBHOOK_QUALITY_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class TaskResult:
    id: int | None
    task_id: int
    text: str
    created_by: int | None
    created_at: str | None
    status: str | None


def is_quality_exempt_responsible(responsible_id: int | None) -> bool:
    settings = get_settings()
    return (
        responsible_id is not None and responsible_id in settings.resolved_quality_control_exempt_responsible_user_ids
    )


async def handle_quality_control_webhook_event(
    bitrix: BitrixClient,
    *,
    payload: dict[str, Any],
    status: dict[str, Any] | None = None,
    specialist: Any | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    event_type = _payload_event_type(payload)
    if event_type not in TASK_QUALITY_WEBHOOK_EVENTS:
        return {"handled": False, "reason": "unsupported_event", "event": event_type}

    if status is not None:
        status["last_received_at"] = datetime.now(MOSCOW_TZ).isoformat()
        status["last_event"] = event_type
        status["events_seen"] = int(status.get("events_seen") or 0) + 1

    if not settings.quality_control_webhook_enabled:
        _record_ignored(status, "disabled")
        return {"handled": False, "reason": "disabled", "event": event_type}

    task_id = _extract_task_id_from_event(payload)
    if task_id is None:
        _record_ignored(status, "task_id_not_found")
        return {"handled": False, "reason": "task_id_not_found", "event": event_type}

    if status is not None:
        status["last_task_id"] = task_id

    async with _WEBHOOK_QUALITY_LOCK:
        try:
            result = await _handle_quality_control_webhook_task(
                bitrix,
                task_id=task_id,
                event_type=event_type,
                payload=payload,
                specialist=specialist,
            )
        except Exception as exc:
            if status is not None:
                status["last_error"] = f"{type(exc).__name__}: {exc}"
                status["errors"] = int(status.get("errors") or 0) + 1
            raise

    if status is not None:
        status["last_error"] = None
        status["last_reason"] = result.get("reason")
        if result.get("handled"):
            status["tasks_processed"] = int(status.get("tasks_processed") or 0) + 1
            status["last_actions"] = result.get("actions", [])
        elif result.get("duplicate"):
            status["duplicates_seen"] = int(status.get("duplicates_seen") or 0) + 1
        else:
            status["ignored"] = int(status.get("ignored") or 0) + 1
    return result


async def _handle_quality_control_webhook_task(
    bitrix: BitrixClient,
    *,
    task_id: int,
    event_type: str,
    payload: dict[str, Any],
    specialist: Any | None,
) -> dict[str, Any]:
    settings = get_settings()

    # Pre-fetch task for status check and deduplication
    task_detail = await _fetch_task_detail(bitrix, task_id)
    if not task_detail:
        return {"handled": False, "reason": "task_fetch_failed", "event": event_type, "task_id": task_id}

    task_status = _to_str(_first(task_detail, "status", "STATUS"))
    if task_status != "4":
        return {
            "handled": False,
            "reason": f"status_{task_status}_not_waiting_control",
            "event": event_type,
            "task_id": task_id,
        }

    # Exempt responsible check
    responsible_id = _to_int(_first(task_detail, "responsibleId", "RESPONSIBLE_ID"))
    if is_quality_exempt_responsible(responsible_id):
        return {"handled": False, "reason": "exempt_responsible", "event": event_type, "task_id": task_id}

    # Fetch results
    try:
        raw_results = await bitrix.list_task_results(task_id)
    except Exception as exc:
        logger.warning("quality_control: failed to list results for task %s: %s", task_id, exc)
        raw_results = []

    # Deduplication
    result_obj = _latest_task_result(task_id, raw_results)
    result_text = result_obj.text if result_obj else ""
    process_key = _webhook_quality_process_key(task_id, task_detail, result_obj, result_text)
    duplicate = _quality_duplicate_result(task_id, event_type, process_key)
    if duplicate:
        return duplicate

    _mark_quality_processing(task_id, event_type, process_key)
    try:
        if specialist is not None:
            agent_result = await specialist.handle(
                AgentTask(
                    task_id=f"qc_{task_id}",
                    source="quality_control_webhook",
                    request=(
                        f"Контроль качества: задача #{task_id} в статусе «ждёт контроля» (STATUS=4). "
                        "Проверь результат задачи и выполни необходимые действия согласно инструкции."
                    ),
                    user=UserContext(id=str(responsible_id) if responsible_id else ""),
                    context={
                        "event_type": event_type,
                        "webhook_task_id": task_id,
                        "task_detail": task_detail,
                        "task_results": _extract_results(raw_results),
                        "event": "quality_control_webhook",
                        "quality_control_dry_run": settings.quality_control_dry_run,
                    },
                )
            )
            actions = [a.name for a in agent_result.actions_taken]
        else:
            logger.warning("quality_control: no specialist provided for task %s, skipping", task_id)
            actions = []
    except Exception:
        _mark_quality_failed(task_id, event_type, process_key)
        raise

    return _mark_quality_done(task_id, event_type=event_type, process_key=process_key, actions=actions)


def _quality_duplicate_result(task_id: int, event_type: str, process_key: str) -> dict[str, Any] | None:
    state = _load_state(get_settings().quality_control_state_path)
    ledger = _webhook_quality_ledger(state)
    existing = ledger.get(process_key)
    if isinstance(existing, dict) and existing.get("status") == "done":
        return {
            "handled": False,
            "duplicate": True,
            "reason": "already_processed",
            "event": event_type,
            "task_id": task_id,
            "process_key": process_key,
        }
    return None


def _mark_quality_processing(task_id: int, event_type: str, process_key: str) -> None:
    settings = get_settings()
    state = _load_state(settings.quality_control_state_path)
    ledger = _webhook_quality_ledger(state)
    ledger[process_key] = {
        "status": "processing",
        "task_id": task_id,
        "event": event_type,
        "started_at": datetime.now(MOSCOW_TZ).isoformat(),
    }
    _prune_webhook_quality_ledger(ledger)
    _save_state(settings.quality_control_state_path, state)


def _mark_quality_failed(task_id: int, event_type: str, process_key: str) -> None:
    settings = get_settings()
    state = _load_state(settings.quality_control_state_path)
    ledger = _webhook_quality_ledger(state)
    ledger[process_key] = {
        "status": "failed",
        "task_id": task_id,
        "event": event_type,
        "failed_at": datetime.now(MOSCOW_TZ).isoformat(),
    }
    _save_state(settings.quality_control_state_path, state)


def _mark_quality_done(
    task_id: int,
    *,
    event_type: str,
    process_key: str,
    actions: list[str] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    state = _load_state(settings.quality_control_state_path)
    ledger = _webhook_quality_ledger(state)
    ledger[process_key] = {
        "status": "done",
        "task_id": task_id,
        "event": event_type,
        "processed_at": datetime.now(MOSCOW_TZ).isoformat(),
        "actions": actions or [],
    }
    _prune_webhook_quality_ledger(ledger)
    _save_state(settings.quality_control_state_path, state)
    return {
        "handled": True,
        "reason": "processed",
        "event": event_type,
        "task_id": task_id,
        "actions": actions or [],
        "process_key": process_key,
    }


def _record_ignored(status: dict[str, Any] | None, reason: str) -> None:
    if status is None:
        return
    status["ignored"] = int(status.get("ignored") or 0) + 1
    status["last_reason"] = reason


def _extract_task_detail(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        task = result.get("task")
        if isinstance(task, dict):
            return task
        return result
    return {}


async def _fetch_task_detail(bitrix: BitrixClient, task_id: int) -> dict[str, Any]:
    try:
        raw_detail = await bitrix.get_task(
            task_id,
            select=[
                "ID",
                "TITLE",
                "DESCRIPTION",
                "STATUS",
                "RESPONSIBLE_ID",
                "CREATED_BY",
                "GROUP_ID",
                "DEADLINE",
                "TASK_CONTROL",
                "CHANGED_DATE",
                "CLOSED_DATE",
                "CLOSED_BY",
                "STATUS_CHANGED_BY",
            ],
        )
    except Exception:
        logger.exception("Failed to fetch task detail for quality control: task_id=%s", task_id)
        return {}
    return _extract_task_detail(raw_detail)


def _latest_task_result(task_id: int, result: object) -> TaskResult | None:
    results = _extract_results(result)
    if not results:
        return None
    latest = sorted(
        results,
        key=lambda item: str(_first(item, "updatedAt", "UPDATED_AT", "createdAt", "CREATED_AT") or ""),
        reverse=True,
    )[0]
    return TaskResult(
        id=_to_int(_first(latest, "id", "ID")),
        task_id=task_id,
        text=str(_first(latest, "text", "TEXT", "formattedText", "FORMATTED_TEXT") or ""),
        created_by=_to_int(_first(latest, "createdBy", "CREATED_BY")),
        created_at=_to_str(_first(latest, "createdAt", "CREATED_AT")),
        status=_to_str(_first(latest, "status", "STATUS")),
    )


def _extract_results(result: object) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        items = result.get("results") or result.get("items") or result.get("result")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
    return []


def _clean_result_text(text: str) -> str:
    without_bbcode = re.sub(r"\[/?[a-zA-Z0-9_]+(?:=[^\]]*)?\]", "", text)
    without_html = re.sub(r"<[^>]+>", "", without_bbcode)
    return without_html.replace("\r\n", "\n").replace("\r", "\n").strip()


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load quality control state")
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _payload_event_type(payload: dict[str, Any]) -> str:
    return str(payload.get("event") or payload.get("EVENT") or payload.get("type") or "").upper()


def _extract_task_id_from_event(payload: dict[str, Any]) -> int | None:
    data = _dict_value(_first_ci(payload, "data", "DATA"))
    fields_after = _dict_value(_first_ci(data, "FIELDS_AFTER", "fieldsAfter"))
    fields_before = _dict_value(_first_ci(data, "FIELDS_BEFORE", "fieldsBefore"))
    for container in (fields_after, fields_before, data, payload):
        task_id = _to_int(_first_ci(container, "ID", "id", "TASK_ID", "taskId", "task_id"))
        if task_id is not None:
            return task_id
    return None


def _webhook_quality_process_key(
    task_id: int,
    task_data: dict[str, Any],
    result: TaskResult | None,
    result_text: str,
) -> str:
    payload = {
        "task_id": task_id,
        "status": _to_str(_first(task_data, "status", "STATUS")) or "",
        "changed_date": _to_str(_first(task_data, "changedDate", "CHANGED_DATE")) or "",
        "closed_date": _to_str(_first(task_data, "closedDate", "CLOSED_DATE")) or "",
        "result_id": result.id if result else None,
        "result_created_at": result.created_at if result else None,
        "description_hash": _short_hash(str(_first(task_data, "description", "DESCRIPTION") or "")),
        "result_hash": _short_hash(_clean_result_text(result_text)),
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return f"task_quality:{task_id}:{digest}"


def _webhook_quality_ledger(state: dict[str, Any]) -> dict[str, Any]:
    ledger = state.get("_webhook_quality")
    if isinstance(ledger, dict):
        return ledger
    ledger = {}
    state["_webhook_quality"] = ledger
    return ledger


def _prune_webhook_quality_ledger(ledger: dict[str, Any], *, limit: int = 2000) -> None:
    while len(ledger) > limit:
        ledger.pop(next(iter(ledger)), None)


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_ci(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    lowered = {str(key).lower(): value for key, value in data.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None:
            return value
    return None


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _to_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
