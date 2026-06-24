"""Quality control logic owned by Bitrix24Specialist.

Moved from workers/bitrix/quality_control.py so that the specialist
fully owns its domain responsibility. The worker layer only routes events.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_server.models import AgentResult, AgentTask, UserContext
from ai_server.settings import Settings
from ai_server.utils import MOSCOW_TZ

if TYPE_CHECKING:
    from ai_server.agents.bitrix24.specialist import Bitrix24Specialist
    from ai_server.integrations.bitrix.ports import BitrixTaskPort

logger = logging.getLogger(__name__)

TASK_QUALITY_WEBHOOK_EVENTS = {"ONTASKUPDATE"}
_QUALITY_LOCK = asyncio.Lock()


@dataclass(frozen=True)
class TaskResult:
    id: int | None
    task_id: int
    text: str
    created_by: int | None
    created_at: str | None
    status: str | None


def is_quality_exempt_responsible(responsible_id: int | None, *, settings: Settings) -> bool:
    return (
        responsible_id is not None and responsible_id in settings.resolved_quality_control_exempt_responsible_user_ids
    )


async def handle_quality_control_task(
    specialist: Bitrix24Specialist,
    task: AgentTask,
    *,
    bitrix: BitrixTaskPort,
    settings: Settings,
) -> AgentResult:
    """Entry point called by Bitrix24Specialist.handle() for quality control tasks."""
    event_type = str(task.context.get("bitrix_event_type") or "").upper()
    task_id = task.context.get("task_id")

    if not settings.quality_control_webhook_enabled:
        return _skip_result(specialist.manifest.id, "quality_control_disabled")

    if task_id is None:
        return _skip_result(specialist.manifest.id, "task_id_not_found")

    try:
        return await _run_quality_control(
            specialist,
            bitrix=bitrix,
            task_id=int(task_id),
            event_type=event_type,
            settings=settings,
        )
    except Exception as exc:
        logger.exception("quality_control: unhandled error for task %s", task_id)
        return AgentResult(
            status="failed",
            agent_id=specialist.manifest.id,
            answer=f"Контроль качества: ошибка обработки задачи #{task_id}: {type(exc).__name__}: {exc}",
        )


async def _run_quality_control(
    specialist: Bitrix24Specialist,
    *,
    bitrix: BitrixTaskPort,
    task_id: int,
    event_type: str,
    settings: Settings,
) -> AgentResult:
    # Slow external calls — outside lock
    task_detail = await _fetch_task_detail(bitrix, task_id)
    if not task_detail:
        return _skip_result(specialist.manifest.id, "task_fetch_failed")

    task_status = _to_str(_first(task_detail, "status", "STATUS"))
    if task_status != "4":
        return _skip_result(specialist.manifest.id, f"status_{task_status}_not_waiting_control")

    responsible_id = _to_int(_first(task_detail, "responsibleId", "RESPONSIBLE_ID"))
    if is_quality_exempt_responsible(responsible_id, settings=settings):
        return _skip_result(specialist.manifest.id, "exempt_responsible")

    try:
        raw_results = await bitrix.list_task_results(task_id)
    except Exception as exc:
        logger.warning("quality_control: failed to list results for task %s: %s", task_id, exc)
        raw_results = []

    result_obj = _latest_task_result(task_id, raw_results)
    result_text = result_obj.text if result_obj else ""
    process_key = _webhook_quality_process_key(task_id, task_detail, result_obj, result_text)

    # Lock only around dedup state read/write (fast JSON file operations)
    async with _QUALITY_LOCK:
        if _quality_duplicate_result(task_id, event_type, process_key, settings=settings):
            return _skip_result(specialist.manifest.id, "already_processed")
        _mark_quality_processing(task_id, event_type, process_key, settings=settings)

    actor_user_id = settings.quality_control_actor_user_id
    qc_task = AgentTask(
        task_id=f"qc_{task_id}",
        source="quality_control_webhook",
        request=(
            f"Контроль качества: задача #{task_id} в статусе «ждёт контроля» (STATUS=4). "
            "Проверь результат задачи и выполни необходимые действия согласно инструкции."
        ),
        user=UserContext(id=str(actor_user_id) if actor_user_id else ""),
        context={
            "event_type": event_type,
            "webhook_task_id": task_id,
            "task_detail": task_detail,
            "task_results": _extract_results(raw_results),
            "task_responsible_id": responsible_id,
            "event": "quality_control_webhook",
            "quality_control_dry_run": settings.quality_control_dry_run,
        },
    )
    try:
        from ai_server.agents.base import BaseSpecialist

        agent_result = await BaseSpecialist.handle(specialist, qc_task)
    except Exception:
        async with _QUALITY_LOCK:
            _mark_quality_failed(task_id, event_type, process_key, settings=settings)
        raise

    async with _QUALITY_LOCK:
        _mark_quality_done(task_id, event_type=event_type, process_key=process_key, settings=settings)
    return agent_result


def _skip_result(agent_id: str, reason: str) -> AgentResult:
    return AgentResult(
        status="completed",
        agent_id=agent_id,
        answer=f"quality_control: skipped ({reason})",
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Deduplication (JSON file, same as before — local to process)
# ---------------------------------------------------------------------------


def _quality_duplicate_result(task_id: int, event_type: str, process_key: str, *, settings: Settings) -> bool:
    state = _load_state(settings.quality_control_state_path)
    ledger = _webhook_quality_ledger(state)
    existing = ledger.get(process_key)
    return isinstance(existing, dict) and existing.get("status") == "done"


def _mark_quality_processing(task_id: int, event_type: str, process_key: str, *, settings: Settings) -> None:
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


def _mark_quality_failed(task_id: int, event_type: str, process_key: str, *, settings: Settings) -> None:
    state = _load_state(settings.quality_control_state_path)
    ledger = _webhook_quality_ledger(state)
    ledger[process_key] = {
        "status": "failed",
        "task_id": task_id,
        "event": event_type,
        "failed_at": datetime.now(MOSCOW_TZ).isoformat(),
    }
    _save_state(settings.quality_control_state_path, state)


def _mark_quality_done(task_id: int, *, event_type: str, process_key: str, settings: Settings) -> None:
    state = _load_state(settings.quality_control_state_path)
    ledger = _webhook_quality_ledger(state)
    ledger[process_key] = {
        "status": "done",
        "task_id": task_id,
        "event": event_type,
        "processed_at": datetime.now(MOSCOW_TZ).isoformat(),
    }
    _prune_webhook_quality_ledger(ledger)
    _save_state(settings.quality_control_state_path, state)


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


# ---------------------------------------------------------------------------
# Bitrix data helpers
# ---------------------------------------------------------------------------


async def _fetch_task_detail(bitrix: BitrixTaskPort, task_id: int) -> dict[str, Any]:
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


def _extract_task_detail(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        task = result.get("task")
        if isinstance(task, dict):
            return task
        return result
    return {}


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


def _webhook_quality_process_key(
    task_id: int,
    task_data: dict[str, Any],
    result: TaskResult | None,
    result_text: str,
) -> str:
    without_bbcode = re.sub(r"\[/?[a-zA-Z0-9_]+(?:=[^\]]*)?\]", "", result_text)
    without_html = re.sub(r"<[^>]+>", "", without_bbcode)
    clean_text = without_html.replace("\r\n", "\n").replace("\r", "\n").strip()
    payload = {
        "task_id": task_id,
        "status": _to_str(_first(task_data, "status", "STATUS")) or "",
        "changed_date": _to_str(_first(task_data, "changedDate", "CHANGED_DATE")) or "",
        "closed_date": _to_str(_first(task_data, "closedDate", "CLOSED_DATE")) or "",
        "result_id": result.id if result else None,
        "result_created_at": result.created_at if result else None,
        "description_hash": _short_hash(str(_first(task_data, "description", "DESCRIPTION") or "")),
        "result_hash": _short_hash(clean_text),
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
    return f"task_quality:{task_id}:{digest}"


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
