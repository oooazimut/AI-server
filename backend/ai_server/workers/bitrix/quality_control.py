from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.result_templates import active_result_templates_context
from ai_server.settings import get_settings
from ai_server.utils import MOSCOW_TZ, confidence

logger = logging.getLogger(__name__)
TASK_QUALITY_WEBHOOK_EVENTS = {"ONTASKUPDATE"}
_WEBHOOK_QUALITY_LOCK = asyncio.Lock()


class QualityReviewError(RuntimeError):
    pass


class QualityControlError(RuntimeError):
    pass


class QualityReviewer(Protocol):
    async def review(
        self,
        *,
        title: str,
        description: str,
        result_text: str,
    ) -> TemplateValidation:
        pass


class QualityControlLLM(Protocol):
    async def decide(
        self,
        *,
        task_id: int,
        event_type: str,
        payload: dict[str, Any],
        tool_results: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
    ) -> QualityControlDecision:
        pass


@dataclass(frozen=True)
class TaskResult:
    id: int | None
    task_id: int
    text: str
    created_by: int | None
    created_at: str | None
    status: str | None


@dataclass(frozen=True)
class TemplateValidation:
    template_id: str
    valid: bool
    outcome: str
    issues: list[str]
    fixes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QualityTask:
    id: int
    title: str
    description: str
    group_id: int | None
    responsible_id: int | None
    creator_id: int | None
    status: str | None
    task_control: bool
    result: TaskResult | None
    validation: TemplateValidation

    @property
    def is_invalid(self) -> bool:
        return not self.validation.valid

    @property
    def state_signature(self) -> str:
        result_id = self.result.id if self.result else "none"
        return f"{self.id}:{self.status}:{result_id}:{'valid' if self.validation.valid else 'invalid'}"


@dataclass
class QualityAction:
    task_id: int
    dry_run: bool
    action: str
    sent_to_responsible: bool = False
    sent_to_director: bool = False
    notified_director_user_ids: list[int] | None = None
    returned_to_work: bool = False
    approved: bool = False
    reason: str = ""


@dataclass(frozen=True)
class QualityReport:
    tasks: list[QualityTask]
    checked_at: datetime
    dry_run: bool
    actions: list[QualityAction] = field(default_factory=list)


@dataclass(frozen=True)
class QualityControlToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class QualityControlDecision:
    status: str
    answer: str
    tool_calls: list[QualityControlToolCall]
    confidence: float = 0.5
    raw: dict[str, Any] = field(default_factory=dict)


class FixedQualityReviewer:
    def __init__(self, validation: TemplateValidation) -> None:
        self.validation = validation

    async def review(
        self,
        *,
        title: str,
        description: str,
        result_text: str,
    ) -> TemplateValidation:
        return self.validation


class LLMQualityReviewer:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def review(
        self,
        *,
        title: str,
        description: str,
        result_text: str,
    ) -> TemplateValidation:
        settings = get_settings()
        if not settings.quality_control_smart_enabled:
            return _presence_validation(result_text)
        if not settings.llm_configured:
            raise QualityReviewError("quality_control_llm_not_configured")

        completion = await self.client.complete(
            agent_id="bitrix24_quality_control",
            messages=[
                {"role": "system", "content": SMART_QUALITY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task_title": title,
                            "task_description": _truncate(description, 7000),
                            "executor_result": _truncate(_clean_result_text(result_text), 5000),
                            "result_templates": active_result_templates_context(),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        parsed = completion.json_content()
        valid = _to_bool(parsed.get("valid"))
        outcome = str(parsed.get("outcome") or "unknown").strip() or "unknown"
        issues = _string_list(parsed.get("issues"))
        fixes = _string_list(parsed.get("fixes"))
        missing_items = _string_list(parsed.get("missing_items"))
        for item in missing_items:
            issues.append(f"Из результата не видно, что выполнено по пункту задачи: {item}")
        if missing_items:
            valid = False
        if not valid and not issues:
            issues.append("Модель считает результат недостаточным, но не указала конкретный пункт.")
        if not valid and not fixes:
            fixes.append("Уточните результат: что сделано, что не сделано, причины и что нужно для полного выполнения.")
        return TemplateValidation(
            template_id="llm_quality_review_v1",
            valid=valid,
            outcome=outcome,
            issues=_unique_non_empty(issues),
            fixes=_unique_non_empty(fixes),
        )


class LLMQualityControlService:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def decide(
        self,
        *,
        task_id: int,
        event_type: str,
        payload: dict[str, Any],
        tool_results: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
    ) -> QualityControlDecision:
        settings = get_settings()
        if not settings.llm_configured:
            raise QualityControlError("quality_control_llm_not_configured")

        completion = await self.client.complete(
            agent_id="bitrix24_quality_control",
            messages=[
                {"role": "system", "content": QUALITY_CONTROL_AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "event": {
                                "task_id": task_id,
                                "event_type": event_type,
                                "payload": _redact_payload(payload),
                                "current_datetime": datetime.now(MOSCOW_TZ).isoformat(),
                            },
                            "policy": _quality_policy_context(settings),
                            "result_templates": active_result_templates_context(),
                            "tools": tool_definitions,
                            "tool_results": tool_results,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return _parse_quality_control_decision(completion.json_content())


class LLMDrivenQualityControlAgent:
    def __init__(self, llm: QualityControlLLM | None = None) -> None:
        self.llm = llm or LLMQualityControlService()

    async def handle(
        self,
        bitrix: BitrixClient,
        *,
        task_id: int,
        event_type: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        tool_results: list[dict[str, Any]] = []
        context: dict[str, Any] = {}

        for step in range(1, 6):
            decision = await self.llm.decide(
                task_id=task_id,
                event_type=event_type,
                payload=payload,
                tool_results=tool_results,
                tool_definitions=_quality_control_tool_definitions(),
            )
            if not decision.tool_calls or all(call.name == "none" for call in decision.tool_calls):
                if decision.status == "ignored":
                    return {
                        "handled": False,
                        "reason": decision.answer or "llm_ignored_event",
                        "event": event_type,
                        "task_id": task_id,
                        "llm_steps": step,
                    }
                return {
                    "handled": False,
                    "reason": "llm_returned_no_tool_call",
                    "event": event_type,
                    "task_id": task_id,
                    "llm_steps": step,
                }

            for tool_call in decision.tool_calls:
                tool_result = await _execute_quality_control_tool(
                    bitrix,
                    tool_call,
                    task_id=task_id,
                    event_type=event_type,
                    context=context,
                )
                tool_results.append(tool_result)
                if tool_result["tool"] == "bitrix_task_get" and tool_result["status"] == "ok":
                    context["task_detail"] = _dict_value(tool_result.get("data", {}).get("task"))
                if tool_result["tool"] == "bitrix_task_results_list" and tool_result["status"] == "ok":
                    context["raw_results"] = tool_result.get("data", {}).get("results")
                final_result = tool_result.get("data", {}).get("final_result")
                if isinstance(final_result, dict):
                    final_result["llm_steps"] = step
                    return final_result

        return {
            "handled": False,
            "reason": "quality_control_llm_max_steps",
            "event": event_type,
            "task_id": task_id,
        }


async def handle_quality_control_webhook_event(
    bitrix: BitrixClient,
    *,
    payload: dict[str, Any],
    quality_llm: QualityControlLLM | None = None,
    status: dict[str, Any] | None = None,
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
                quality_llm=quality_llm,
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
    quality_llm: QualityControlLLM | None,
) -> dict[str, Any]:
    return await LLMDrivenQualityControlAgent(quality_llm).handle(
        bitrix,
        task_id=task_id,
        event_type=event_type,
        payload=payload,
    )


async def _execute_quality_control_tool(
    bitrix: BitrixClient,
    tool_call: QualityControlToolCall,
    *,
    task_id: int,
    event_type: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    if tool_call.name == "none":
        return {"status": "ok", "tool": "none", "data": {}}

    if tool_call.name == "bitrix_task_get":
        requested_task_id = _to_int(tool_call.args.get("task_id")) or task_id
        try:
            raw_detail = await bitrix.get_task(
                requested_task_id,
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
        except Exception as exc:
            return _quality_tool_error("bitrix_task_get", exc, {"task_id": requested_task_id})
        return {
            "status": "ok",
            "tool": "bitrix_task_get",
            "data": {"task_id": requested_task_id, "task": _extract_task_detail(raw_detail)},
        }

    if tool_call.name == "bitrix_task_results_list":
        requested_task_id = _to_int(tool_call.args.get("task_id")) or task_id
        try:
            raw_results = await bitrix.list_task_results(requested_task_id)
        except Exception as exc:
            return _quality_tool_error("bitrix_task_results_list", exc, {"task_id": requested_task_id})
        return {
            "status": "ok",
            "tool": "bitrix_task_results_list",
            "data": {
                "task_id": requested_task_id,
                "results": _extract_results(raw_results),
            },
        }

    if tool_call.name == "quality_control_action":
        return await _execute_quality_control_action_tool(
            bitrix,
            tool_call.args,
            task_id=task_id,
            event_type=event_type,
            context=context,
        )

    return {
        "status": "invalid_tool_call",
        "tool": tool_call.name,
        "error": f"unknown quality-control tool: {tool_call.name}",
    }


async def _execute_quality_control_action_tool(
    bitrix: BitrixClient,
    args: dict[str, Any],
    *,
    task_id: int,
    event_type: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    action = str(args.get("action") or "").strip().lower()
    reason = str(args.get("reason") or "").strip()
    if action in {"ignore", "skip", "noop"}:
        return {
            "status": "ok",
            "tool": "quality_control_action",
            "data": {
                "final_result": {
                    "handled": False,
                    "reason": reason or "llm_ignored_event",
                    "event": event_type,
                    "task_id": task_id,
                }
            },
        }

    task_detail = _dict_value(args.get("task")) or _dict_value(context.get("task_detail"))
    raw_results = args.get("results")
    if raw_results is None:
        raw_results = context.get("raw_results")
    if not task_detail:
        return {
            "status": "contract_violation",
            "tool": "quality_control_action",
            "error": "quality_control_action requires task data read through bitrix_task_get",
            "data": {"task_id": task_id},
        }

    validation = _validation_from_quality_action(args)
    task = await _build_quality_task_from_snapshot(
        task_detail,
        raw_results,
        reviewer=FixedQualityReviewer(validation),
    )
    if task is None:
        return {
            "status": "not_found",
            "tool": "quality_control_action",
            "data": {
                "final_result": {
                    "handled": False,
                    "reason": "task_detail_not_found",
                    "event": event_type,
                    "task_id": task_id,
                }
            },
        }

    policy_error = _quality_policy_error(task)
    if policy_error:
        return {
            "status": "denied",
            "tool": "quality_control_action",
            "data": {"final_result": policy_error},
        }

    result = _latest_task_result(task.id, raw_results)
    result_text = result.text if result else ""
    process_key = _webhook_quality_process_key(task.id, task_detail, result, result_text)
    duplicate = _quality_duplicate_result(task.id, event_type, process_key)
    if duplicate:
        return {
            "status": "duplicate",
            "tool": "quality_control_action",
            "data": {"final_result": duplicate},
        }

    _mark_quality_processing(task.id, event_type, process_key)
    try:
        report = QualityReport(
            tasks=[task],
            checked_at=datetime.now(MOSCOW_TZ),
            dry_run=get_settings().quality_control_dry_run,
        )
        actions = await apply_quality_actions(bitrix, report)
    except Exception:
        _mark_quality_failed(task.id, event_type, process_key)
        raise

    final_result = _mark_quality_done(
        task,
        event_type=event_type,
        process_key=process_key,
        actions=actions,
    )
    return {
        "status": "ok",
        "tool": "quality_control_action",
        "data": {"final_result": final_result},
    }


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
    task: QualityTask,
    *,
    event_type: str,
    process_key: str,
    actions: list[QualityAction],
) -> dict[str, Any]:
    settings = get_settings()
    state = _load_state(settings.quality_control_state_path)
    ledger = _webhook_quality_ledger(state)
    ledger[process_key] = {
        "status": "done",
        "task_id": task.id,
        "event": event_type,
        "processed_at": datetime.now(MOSCOW_TZ).isoformat(),
        "valid": not task.is_invalid,
        "actions": [action.action for action in actions],
    }
    _prune_webhook_quality_ledger(ledger)
    _save_state(settings.quality_control_state_path, state)
    return {
        "handled": True,
        "reason": "processed",
        "event": event_type,
        "task_id": task.id,
        "valid": not task.is_invalid,
        "actions": [action.action for action in actions],
        "process_key": process_key,
    }


def _quality_policy_error(task: QualityTask) -> dict[str, Any] | None:
    settings = get_settings()
    if (
        settings.quality_control_webhook_auto_managed_only
        and settings.quality_control_auto_manage_project_id is not None
        and task.group_id != settings.quality_control_auto_manage_project_id
    ):
        return {
            "handled": False,
            "reason": "outside_auto_managed_project",
            "event": "ONTASKUPDATE",
            "task_id": task.id,
            "group_id": task.group_id,
            "auto_manage_project_id": settings.quality_control_auto_manage_project_id,
        }
    return None


def _validation_from_quality_action(args: dict[str, Any]) -> TemplateValidation:
    validation = _dict_value(args.get("validation"))
    action = str(args.get("action") or "").strip().lower()
    valid_value = validation.get("valid") if "valid" in validation else args.get("valid")
    if valid_value is None:
        valid = action in {"approve", "accept", "accept_external_close"}
    else:
        valid = _to_bool(valid_value)
    issues = _string_list(validation.get("issues") or args.get("issues"))
    fixes = _string_list(validation.get("fixes") or args.get("fixes"))
    if not valid and not issues:
        issues.append("LLM-субагент считает результат недостаточным.")
    if not valid and not fixes:
        fixes.append("Уточните результат выполнения задачи.")
    return TemplateValidation(
        template_id=str(validation.get("template_id") or "llm_quality_control_agent_v1"),
        valid=valid,
        outcome=str(validation.get("outcome") or args.get("outcome") or action or "unknown"),
        issues=_unique_non_empty(issues),
        fixes=_unique_non_empty(fixes),
    )


async def apply_quality_actions(bitrix: BitrixClient, report: QualityReport) -> list[QualityAction]:
    settings = get_settings()
    state = _load_state(settings.quality_control_state_path)
    actions: list[QualityAction] = []
    director_user_ids = (
        settings.resolved_quality_control_director_user_ids if settings.quality_control_notify_director else []
    )

    for task in report.tasks:
        existing_entry = _quality_state_entry(state.get(str(task.id)))
        same_signature = existing_entry.get("signature") == task.state_signature
        auto_manage = _is_auto_managed_task(task)
        missing_director_user_ids = (
            _missing_user_ids(director_user_ids, existing_entry.get("director_user_ids", []))
            if same_signature
            else director_user_ids
        )

        if task.is_invalid:
            if (
                same_signature
                and not missing_director_user_ids
                and (not auto_manage or bool(existing_entry.get("returned_to_work")))
            ):
                continue

            should_change_task = auto_manage or not settings.quality_control_notify_only
            action = QualityAction(
                task_id=task.id,
                dry_run=report.dry_run,
                action=(
                    "notify_missing"
                    if same_signature and missing_director_user_ids
                    else "notify_only"
                    if not should_change_task
                    else "disapprove"
                    if str(task.status) == "4"
                    else "renew"
                ),
                reason="dry_run" if report.dry_run else "",
            )
            actions.append(action)

            if report.dry_run:
                continue

            message = format_quality_failure_message(task)
            if not should_change_task:
                message = "Тестовый режим контроля результата: задачу пока не возвращаю на доработку.\n\n" + message
                notified_director_user_ids = await _notify_users(
                    bitrix,
                    missing_director_user_ids,
                    message=message,
                    tag="bitrix_ai_agent_quality_control_test",
                    sub_tag=f"task:{task.id}",
                )
                action.sent_to_director = bool(notified_director_user_ids)
                action.notified_director_user_ids = notified_director_user_ids
                state[str(task.id)] = _quality_state_payload(
                    signature=task.state_signature,
                    director_user_ids=_merge_user_ids(
                        existing_entry.get("director_user_ids", []),
                        notified_director_user_ids,
                    ),
                    sent_to_responsible=bool(existing_entry.get("sent_to_responsible")),
                    returned_to_work=bool(existing_entry.get("returned_to_work")),
                    approved=bool(existing_entry.get("approved")),
                )
                continue

            if same_signature and missing_director_user_ids:
                notified_director_user_ids = await _notify_users(
                    bitrix,
                    missing_director_user_ids,
                    message=message,
                    tag="bitrix_ai_agent_quality_control",
                    sub_tag=f"task:{task.id}",
                )
                action.sent_to_director = bool(notified_director_user_ids)
                action.notified_director_user_ids = notified_director_user_ids
                state[str(task.id)] = _quality_state_payload(
                    signature=task.state_signature,
                    director_user_ids=_merge_user_ids(
                        existing_entry.get("director_user_ids", []),
                        notified_director_user_ids,
                    ),
                    sent_to_responsible=bool(existing_entry.get("sent_to_responsible")),
                    returned_to_work=bool(existing_entry.get("returned_to_work")),
                    approved=bool(existing_entry.get("approved")),
                )
                continue

            if str(task.status) == "4":
                await bitrix.disapprove_task(task.id)
            else:
                await bitrix.renew_task(task.id)
            action.returned_to_work = True
            await bitrix.add_task_comment(task_id=task.id, message=message)

            if settings.quality_control_notify_responsible and task.responsible_id:
                await bitrix.notify_user(
                    user_id=task.responsible_id,
                    message=message,
                    tag="bitrix_ai_agent_quality_control",
                    sub_tag=f"task:{task.id}",
                )
                action.sent_to_responsible = True

            notified_director_user_ids = await _notify_users(
                bitrix,
                missing_director_user_ids,
                message=message,
                tag="bitrix_ai_agent_quality_control",
                sub_tag=f"task:{task.id}",
            )
            if notified_director_user_ids:
                action.sent_to_director = True
                action.notified_director_user_ids = notified_director_user_ids

            state[str(task.id)] = _quality_state_payload(
                signature=task.state_signature,
                director_user_ids=_merge_user_ids(
                    existing_entry.get("director_user_ids", []),
                    notified_director_user_ids,
                ),
                sent_to_responsible=action.sent_to_responsible,
                returned_to_work=True,
                approved=False,
            )
            continue

        if not auto_manage:
            continue
        if same_signature and bool(existing_entry.get("approved")):
            continue
        if str(task.status) == "5":
            action = QualityAction(
                task_id=task.id,
                dry_run=report.dry_run,
                action="external_close_accepted",
                reason="closed_by_trusted_or_result_valid",
                approved=True,
            )
            actions.append(action)
            if not report.dry_run:
                state[str(task.id)] = _quality_state_payload(
                    signature=task.state_signature,
                    director_user_ids=_merge_user_ids(existing_entry.get("director_user_ids", [])),
                    sent_to_responsible=bool(existing_entry.get("sent_to_responsible")),
                    returned_to_work=False,
                    approved=True,
                )
            continue

        action = QualityAction(
            task_id=task.id,
            dry_run=report.dry_run,
            action="approve",
            reason="dry_run" if report.dry_run else "",
        )
        actions.append(action)
        if report.dry_run:
            continue

        if str(task.status) == "4":
            await bitrix.approve_task(task.id)
            action.approved = True
        else:
            action.reason = f"status_{task.status}_not_approved"

        message = format_quality_success_message(task)
        await bitrix.add_task_comment(task_id=task.id, message=message)
        if settings.quality_control_notify_responsible and task.responsible_id:
            await bitrix.notify_user(
                user_id=task.responsible_id,
                message=message,
                tag="bitrix_ai_agent_quality_control",
                sub_tag=f"task:{task.id}",
            )
            action.sent_to_responsible = True
        notified_director_user_ids = await _notify_users(
            bitrix,
            missing_director_user_ids,
            message=message,
            tag="bitrix_ai_agent_quality_control",
            sub_tag=f"task:{task.id}",
        )
        if notified_director_user_ids:
            action.sent_to_director = True
            action.notified_director_user_ids = notified_director_user_ids
        state[str(task.id)] = _quality_state_payload(
            signature=task.state_signature,
            director_user_ids=_merge_user_ids(
                existing_entry.get("director_user_ids", []),
                notified_director_user_ids,
            ),
            sent_to_responsible=action.sent_to_responsible,
            returned_to_work=False,
            approved=action.approved,
        )

    if not report.dry_run:
        _save_state(settings.quality_control_state_path, state)
    return actions


async def _notify_users(
    bitrix: BitrixClient,
    user_ids: list[int],
    *,
    message: str,
    tag: str,
    sub_tag: str,
) -> list[int]:
    notified: list[int] = []
    for user_id in user_ids:
        await bitrix.notify_user(user_id=user_id, message=message, tag=tag, sub_tag=sub_tag)
        notified.append(user_id)
    return notified


async def _build_quality_task_from_snapshot(
    task_detail: dict[str, Any],
    raw_results: object,
    *,
    reviewer: QualityReviewer,
) -> QualityTask | None:
    task_id = _to_int(_first(task_detail, "id", "ID"))
    if task_id is None:
        return None

    result = _latest_task_result(task_id, raw_results)
    result_text = result.text if result else ""
    title = str(_first(task_detail, "title", "TITLE") or "Без названия")
    description = _clean_result_text(str(_first(task_detail, "description", "DESCRIPTION") or ""))
    validation = await reviewer.review(title=title, description=description, result_text=result_text)
    return QualityTask(
        id=task_id,
        title=title,
        description=description,
        group_id=_to_int(_first(task_detail, "groupId", "GROUP_ID")),
        responsible_id=_to_int(_first(task_detail, "responsibleId", "RESPONSIBLE_ID")),
        creator_id=_to_int(_first(task_detail, "createdBy", "CREATED_BY")),
        status=_to_str(_first(task_detail, "status", "STATUS")),
        task_control=_is_yes(_first(task_detail, "taskControl", "TASK_CONTROL")),
        result=result,
        validation=validation,
    )


async def build_quality_task_for_result_text(
    bitrix: BitrixClient,
    *,
    task_id: int,
    result_text: str,
    result_created_by: int | None = None,
    reviewer: QualityReviewer | None = None,
) -> QualityTask | None:
    task_detail = await _fetch_task_detail(bitrix, task_id)
    if not task_detail:
        return None
    synthetic_result = {
        "id": None,
        "taskId": task_id,
        "text": result_text,
        "createdBy": result_created_by,
        "createdAt": datetime.now(MOSCOW_TZ).isoformat(),
        "status": "pending_chat_result",
    }
    return await _build_quality_task_from_snapshot(
        task_detail,
        [synthetic_result],
        reviewer=reviewer or LLMQualityReviewer(),
    )


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


def format_quality_failure_message(task: QualityTask) -> str:
    issues = "\n".join(f"- {issue}" for issue in task.validation.issues)
    fixes = "\n".join(f"- {fix}" for fix in _quality_fixes(task.validation))
    return (
        "Контроль результата: задачу нужно доработать.\n\n"
        f"Задача: {_format_task_link(task.id, task.title)}\n"
        f"Проверка: {task.validation.template_id}\n\n"
        f"Что неправильно в результате:\n{issues}\n\n"
        f"Что нужно добавить/исправить:\n{fixes}"
    )


def format_quality_success_message(task: QualityTask) -> str:
    return (
        "Контроль результата: результат принят.\n\n"
        f"Задача: {_format_task_link(task.id, task.title)}\n"
        f"Проверка: {task.validation.template_id}\n\n"
        "Результат соответствует описанию задачи. Задача закрыта агентом после контроля."
    )


def is_quality_exempt_responsible(responsible_id: int | None) -> bool:
    settings = get_settings()
    return (
        responsible_id is not None and responsible_id in settings.resolved_quality_control_exempt_responsible_user_ids
    )


def _presence_validation(text: str) -> TemplateValidation:
    if _clean_result_text(text):
        return TemplateValidation(template_id="result_presence_v1", valid=True, outcome="has_result", issues=[])
    return TemplateValidation(
        template_id="result_presence_v1",
        valid=False,
        outcome="empty",
        issues=["Результат пустой."],
        fixes=["Добавьте результат выполнения задачи."],
    )


def _quality_control_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "bitrix_task_get",
            "description": "Read one Bitrix task by id. Use this before judging a task update event.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
        {
            "name": "bitrix_task_results_list",
            "description": "Read task completion results for one Bitrix task. Use this before judging quality.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "integer"}},
                "required": ["task_id"],
            },
        },
        {
            "name": "quality_control_action",
            "description": (
                "Apply the LLM quality-control decision through backend guardrails. "
                "Call after reading task and results. Use action=ignore for events that should not be processed; "
                "action=approve for valid result; action=return_to_work for insufficient result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["ignore", "approve", "return_to_work", "accept_external_close"],
                    },
                    "reason": {"type": "string"},
                    "validation": {
                        "type": "object",
                        "properties": {
                            "valid": {"type": "boolean"},
                            "outcome": {"type": "string"},
                            "issues": {"type": "array", "items": {"type": "string"}},
                            "fixes": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "required": ["action"],
            },
        },
    ]


def _parse_quality_control_decision(data: dict[str, Any]) -> QualityControlDecision:
    raw_tool_calls = data.get("tool_calls")
    tool_calls: list[QualityControlToolCall] = []
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if name not in {"bitrix_task_get", "bitrix_task_results_list", "quality_control_action", "none"}:
                continue
            args = raw_call.get("args") if isinstance(raw_call.get("args"), dict) else {}
            tool_calls.append(
                QualityControlToolCall(
                    name=name,
                    args=args,
                    summary=str(raw_call.get("summary") or "").strip(),
                )
            )
    if not tool_calls:
        tool_calls = [QualityControlToolCall(name="none")]
    status = str(data.get("status") or "completed").strip()
    if status not in {"continue", "completed", "ignored", "failed"}:
        status = "completed"
    return QualityControlDecision(
        status=status,
        answer=str(data.get("answer") or "").strip(),
        tool_calls=tool_calls,
        confidence=confidence(data.get("confidence")),
        raw=data,
    )


def _quality_policy_context(settings: Any) -> dict[str, Any]:
    return {
        "dry_run": settings.quality_control_dry_run,
        "auto_managed_only": settings.quality_control_webhook_auto_managed_only,
        "auto_manage_project_id": settings.quality_control_auto_manage_project_id,
        "notify_only": settings.quality_control_notify_only,
        "notify_responsible": settings.quality_control_notify_responsible,
        "notify_director": settings.quality_control_notify_director,
        "director_user_ids": settings.resolved_quality_control_director_user_ids,
        "actor_user_id": settings.quality_control_actor_user_id,
        "exempt_responsible_user_ids": settings.resolved_quality_control_exempt_responsible_user_ids,
        "trusted_quality_closer_ids": sorted(_trusted_quality_closer_ids()),
    }


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    def redact(value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key).lower()
                if key_text in {"auth", "token", "application_token", "access_token", "refresh_token"}:
                    result[str(key)] = "<redacted>"
                else:
                    result[str(key)] = redact(item)
            return result
        if isinstance(value, list):
            return [redact(item) for item in value]
        return value

    redacted = redact(payload)
    return redacted if isinstance(redacted, dict) else {}


def _quality_tool_error(tool: str, exc: Exception, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "error",
        "tool": tool,
        "error": f"{type(exc).__name__}: {exc}",
        "data": data,
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


def _quality_fixes(validation: TemplateValidation) -> list[str]:
    if validation.fixes:
        return _unique_non_empty(validation.fixes)
    fixes: list[str] = []
    for issue in validation.issues:
        normalized = _normalize(issue)
        if "результат пустой" in normalized:
            fixes.append("Добавьте результат выполнения задачи.")
        elif "причина" in normalized:
            fixes.append("Добавьте причину, почему задача выполнена не полностью.")
        elif "что нужно" in normalized:
            fixes.append("Добавьте, что нужно получить/сделать, чтобы полностью закрыть задачу.")
        elif "не видно" in normalized:
            fixes.append("Опишите, как результат закрывает конкретные пункты из описания задачи.")
        else:
            fixes.append("Уточните результат по указанному пункту.")
    return _unique_non_empty(fixes) or ["Уточните результат выполнения задачи."]


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


def _quality_state_entry(value: object) -> dict[str, Any]:
    settings = get_settings()
    if isinstance(value, str):
        return {
            "signature": value,
            "director_user_ids": settings.resolved_quality_control_director_user_ids,
            "sent_to_responsible": False,
            "returned_to_work": False,
            "approved": False,
        }
    if isinstance(value, dict):
        return {
            "signature": value.get("signature"),
            "director_user_ids": _int_list(value.get("director_user_ids") or value.get("notified_user_ids") or []),
            "sent_to_responsible": bool(value.get("sent_to_responsible")),
            "returned_to_work": bool(value.get("returned_to_work")),
            "approved": bool(value.get("approved")),
        }
    return {
        "signature": None,
        "director_user_ids": [],
        "sent_to_responsible": False,
        "returned_to_work": False,
        "approved": False,
    }


def _quality_state_payload(
    *,
    signature: str,
    director_user_ids: list[int],
    sent_to_responsible: bool,
    returned_to_work: bool,
    approved: bool = False,
) -> dict[str, Any]:
    return {
        "signature": signature,
        "director_user_ids": _merge_user_ids(director_user_ids),
        "sent_to_responsible": sent_to_responsible,
        "returned_to_work": returned_to_work,
        "approved": approved,
    }


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


def _trusted_quality_closer_ids() -> set[int]:
    settings = get_settings()
    trusted = set(settings.resolved_quality_control_director_user_ids)
    if settings.quality_control_actor_user_id is not None:
        trusted.add(settings.quality_control_actor_user_id)
    return trusted


def _is_auto_managed_task(task: QualityTask) -> bool:
    auto_project_id = get_settings().quality_control_auto_manage_project_id
    if auto_project_id is None:
        return True
    return task.group_id == auto_project_id


def _missing_user_ids(required_user_ids: list[int], notified_user_ids: object) -> list[int]:
    notified = set(_int_list(notified_user_ids))
    return [user_id for user_id in required_user_ids if user_id not in notified]


def _merge_user_ids(*groups: object) -> list[int]:
    result: list[int] = []
    for user_id in _int_list([item for group in groups for item in (group if isinstance(group, list) else [group])]):
        if user_id not in result:
            result.append(user_id)
    return result


def _int_list(value: object) -> list[int]:
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for raw_value in value:
        try:
            result.append(int(raw_value))
        except (TypeError, ValueError):
            continue
    return result


def _format_task_link(task_id: object, title: str) -> str:
    return f"[#{task_id}]({_task_url(task_id)}) {title}".strip()


def _task_url(task_id: object) -> str:
    domain = get_settings().bitrix_domain.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    if not domain:
        return f"/company/personal/user/0/tasks/task/view/{task_id}/"
    return f"https://{domain}/company/personal/user/0/tasks/task/view/{task_id}/"


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def _is_yes(value: object) -> bool:
    return str(value or "").upper() in {"Y", "YES", "TRUE", "1"}


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "да"}


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _unique_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().replace("ё", "е"))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 80].rstrip() + "\n...[текст обрезан для проверки]..."


QUALITY_CONTROL_AGENT_SYSTEM_PROMPT = """
Ты LLM-субагент контроля качества результата задач Bitrix24.

Ты получаешь только webhook-событие и доступные tools. Backend не читает задачу
за тебя и не решает, что проверять. Ты сам выбираешь tools.

Рабочий порядок:
1. Если в tool_results ещё нет карточки задачи, вызови `bitrix_task_get`.
2. Если в tool_results ещё нет результатов задачи, вызови `bitrix_task_results_list`.
3. После чтения задачи и результатов оцени, нужно ли обрабатывать событие.
4. Если статус задачи не `4` (ждёт контроля) и не внешний статус `5`
   (закрыта не доверенным quality closer), вызови `quality_control_action`
   с `action="ignore"` и причиной.
5. Если policy.auto_managed_only=true и задача вне policy.auto_manage_project_id,
   вызови `quality_control_action` с `action="ignore"`.
6. Если ответственный входит в policy.exempt_responsible_user_ids, считай
   результат достаточным и вызови `quality_control_action` с `action="approve"`.
7. Иначе сравни описание задачи и последний результат исполнителя по смыслу.
   Учитывай `result_templates`, если они переданы в user JSON. Не придумывай
   требований, которых нет в описании или применимом шаблоне.
8. Если результат достаточный, вызови `quality_control_action` с
   `action="approve"` и `validation.valid=true`.
9. Если результат пустой, слишком общий, не связан с задачей или не закрывает
   существенные пункты описания, вызови `quality_control_action` с
   `action="return_to_work"`, `validation.valid=false`, списком `issues` и `fixes`.

Верни только JSON без markdown:
{
  "status": "continue|completed|ignored|failed",
  "answer": "краткое внутреннее пояснение",
  "confidence": 0.0,
  "tool_calls": [
    {
      "name": "bitrix_task_get|bitrix_task_results_list|quality_control_action|none",
      "args": {},
      "summary": ""
    }
  ]
}
""".strip()


SMART_QUALITY_SYSTEM_PROMPT = """
Ты LLM-проверяющий качества результата задачи Bitrix24.

Тебе дают JSON с названием задачи, описанием задачи и текстом результата исполнителя.
Нужно сравнить описание задачи и результат по смыслу. Если в JSON есть
`result_templates`, учитывай активный шаблон результата.

Правила оценки:
1. Не придумывай требования, которых нет в описании задачи.
2. Если описание задачи содержит несколько действий, проверь, видно ли из результата выполнение каждого существенного действия.
3. Если из результата видно, что существенные пункты описания выполнены, верни valid=true.
4. Если из результата видно, что задача выполнена не полностью, результат допустим только когда он ясно содержит:
   - что сделано;
   - что не сделано;
   - причины, почему сделано не всё;
   - что нужно для полного выполнения.
5. Если результат пустой, слишком общий или не связан с описанием задачи, верни valid=false.
6. Если по описанию невозможно надёжно понять критерии выполнения, не штрафуй за неясность самой задачи; укажи низкую уверенность.

Верни только JSON без markdown:
{
  "valid": true,
  "outcome": "all_done|not_all_done|unknown",
  "issues": ["что не нравится в результате"],
  "fixes": ["что исполнителю нужно добавить или исправить"],
  "completed_items": ["что видно выполненным"],
  "missing_items": ["что из описания задачи не видно выполненным"],
  "confidence": "high|medium|low"
}
""".strip()
