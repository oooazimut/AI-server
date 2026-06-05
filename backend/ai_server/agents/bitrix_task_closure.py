from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.models import AgentTask
from ai_server.settings import get_settings
from ai_server.workers.bitrix.quality_control import (
    LLMQualityReviewer,
    QualityReviewer,
    QualityTask,
    TemplateValidation,
    build_quality_task_for_result_text,
    format_quality_failure_message,
    is_quality_exempt_responsible,
)


TASK_CLOSURE_PENDING_METHOD = "ai_server.task_closure"


@dataclass(frozen=True)
class BitrixTaskClosureDraft:
    method: str = TASK_CLOSURE_PENDING_METHOD
    params: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    contract_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return not self.contract_errors and bool(self.params.get("result_text"))

    def as_action_details(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "summary": self.summary,
            "contract_errors": self.contract_errors,
            "notes": self.notes,
        }


class StaticQualityReviewer:
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


class TaskClosureService:
    def __init__(
        self,
        bitrix: BitrixClient,
        portal_search: PortalSearchIndex,
        *,
        actor_bitrix: BitrixClient | None = None,
        reviewer: QualityReviewer | None = None,
    ) -> None:
        self.bitrix = bitrix
        self.actor_bitrix = actor_bitrix or bitrix
        self.portal_search = portal_search
        self.reviewer = reviewer or LLMQualityReviewer()

    async def execute(
        self,
        params: dict[str, Any],
        *,
        current_user_id: int | None,
    ) -> dict[str, Any]:
        draft = build_task_closure_draft_from_args(
            AgentTask(task_id="pending_task_closure", request="", user={"id": str(current_user_id) if current_user_id else None}),
            params,
        )
        if not draft.is_ready:
            return {
                "status": "contract_violation",
                "error": "LLM called task_closure with arguments outside the tool contract.",
                **draft.as_action_details(),
            }

        result_text = str(draft.params["result_text"]).strip()
        task_id = _optional_int(draft.params.get("task_id"))
        if task_id is None:
            resolved = self._resolve_task_id(draft.params, current_user_id=current_user_id)
            if resolved.get("status") != "ok":
                return resolved
            task_id = int(resolved["task_id"])

        reviewer = self.reviewer
        responsible_id = await self._task_responsible_id(task_id)
        if is_quality_exempt_responsible(responsible_id):
            reviewer = StaticQualityReviewer(
                TemplateValidation(
                    template_id="exempt_responsible_policy",
                    valid=True,
                    outcome="exempt_responsible",
                    issues=[],
                )
            )

        task = await build_quality_task_for_result_text(
            self.bitrix,
            task_id=task_id,
            result_text=result_text,
            result_created_by=current_user_id,
            reviewer=reviewer,
        )
        if task is None:
            return {"status": "not_found", "message": f"Не нашёл задачу #{task_id}."}

        access_error = self._access_error(task, current_user_id)
        if access_error:
            return access_error

        if str(task.status) == "5":
            return {
                "status": "executed",
                "outcome": "already_closed",
                "valid": True,
                "task": _task_payload(task),
            }

        if task.is_invalid:
            return await self._reject_closure(task, result_text=result_text)
        return await self._close_task(task, result_text=result_text, current_user_id=current_user_id)

    def _resolve_task_id(
        self,
        params: dict[str, Any],
        *,
        current_user_id: int | None,
    ) -> dict[str, Any]:
        query = str(params.get("task_query") or params.get("query") or "").strip()
        if not query:
            return {
                "status": "contract_violation",
                "error": "task_closure requires task_id or task_query.",
            }

        candidates: list[dict[str, Any]] = []
        for item in self.portal_search.search(query, entity_types={"task"}, limit=20):
            task_id = _optional_int(item.entity_id)
            if task_id is None:
                continue
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            status = str(metadata.get("status") or "")
            if status == "5":
                continue
            responsible_id = _optional_int(metadata.get("responsible_id"))
            if current_user_id and responsible_id and responsible_id != current_user_id:
                continue
            group_id = _optional_int(metadata.get("group_id"))
            if not _is_allowed_project(group_id):
                continue
            candidates.append(
                {
                    "id": task_id,
                    "title": item.title,
                    "url": item.url,
                    "status": status,
                    "responsible_id": responsible_id,
                    "group_id": group_id,
                }
            )

        if not candidates:
            return {
                "status": "not_found",
                "message": (
                    "Не нашёл подходящую открытую задачу по описанию. "
                    "Напишите номер задачи, например: закрой задачу #7971, результат: ..."
                ),
            }
        if len(candidates) > 1:
            return {
                "status": "ambiguous",
                "message": "Нашёл несколько похожих задач. Укажите номер нужной задачи.",
                "candidates": candidates[:10],
            }
        return {"status": "ok", "task_id": candidates[0]["id"]}

    async def _task_responsible_id(self, task_id: int) -> int | None:
        try:
            raw = await self.bitrix.get_task(task_id, select=["ID", "RESPONSIBLE_ID"])
        except Exception:
            return None
        task = _extract_task_detail(raw)
        return _optional_int(_first(task, "responsibleId", "RESPONSIBLE_ID"))

    def _access_error(self, task: QualityTask, current_user_id: int | None) -> dict[str, Any] | None:
        if current_user_id is None:
            return {
                "status": "denied",
                "reason": "Не вижу ID текущего пользователя Bitrix, поэтому не могу закрывать задачу.",
            }
        if not _is_allowed_project(task.group_id):
            return {
                "status": "denied",
                "reason": (
                    "Закрытие задач через AI-server пока разрешено только в проекте "
                    f"#{get_settings().quality_control_auto_manage_project_id}."
                ),
                "task": _task_payload(task),
            }
        if current_user_id != task.responsible_id and current_user_id not in get_settings().resolved_agent_write_allowed_user_ids:
            return {
                "status": "denied",
                "reason": "Закрывать задачу через AI-server может её исполнитель или администратор агента.",
                "task": _task_payload(task),
            }
        return None

    async def _reject_closure(self, task: QualityTask, *, result_text: str) -> dict[str, Any]:
        message = format_quality_failure_message(task)
        if result_text:
            message += "\n\nРезультат, который исполнитель отправил через чат:\n" + _truncate(result_text, 2000)

        await self.actor_bitrix.add_task_comment(task_id=task.id, message=message)

        notified_user_ids: list[int] = []
        notify_user_ids: list[int | None] = []
        settings = get_settings()
        if settings.quality_control_notify_responsible and task.responsible_id:
            notify_user_ids.append(task.responsible_id)
        if settings.quality_control_notify_director:
            notify_user_ids.extend(settings.resolved_quality_control_director_user_ids)

        for user_id in _unique_ints(notify_user_ids):
            await self.actor_bitrix.notify_user(
                user_id=user_id,
                message=message,
                tag="ai_server_task_closure",
                sub_tag=f"task:{task.id}:rejected",
            )
            notified_user_ids.append(user_id)

        task_changed = False
        if str(task.status) == "4":
            await self.actor_bitrix.disapprove_task(task.id)
            task_changed = True
        elif str(task.status) in {"5", "7"}:
            await self.actor_bitrix.renew_task(task.id)
            task_changed = True

        return {
            "status": "executed",
            "outcome": "needs_revision",
            "valid": False,
            "task": _task_payload(task),
            "issues": task.validation.issues,
            "fixes": task.validation.fixes,
            "notified_user_ids": notified_user_ids,
            "task_reopened": task_changed,
        }

    async def _close_task(
        self,
        task: QualityTask,
        *,
        result_text: str,
        current_user_id: int | None,
    ) -> dict[str, Any]:
        stored_result_text = result_text
        if current_user_id:
            stored_result_text = (
                result_text.rstrip()
                + f"\n\n[Отправлено через AI-server пользователем Bitrix #{current_user_id}]"
            )

        added_result = await self.bitrix.add_task_result(task.id, stored_result_text)
        completed = await self.bitrix.complete_task(task.id)

        approved = False
        final_status = await self._task_status(task.id)
        if str(final_status) == "4":
            await self.actor_bitrix.approve_task(task.id)
            approved = True
            final_status = "5"

        return {
            "status": "executed",
            "outcome": "closed",
            "valid": True,
            "task": _task_payload(task),
            "result": added_result,
            "complete_result": completed,
            "approved": approved,
            "final_status": final_status,
        }

    async def _task_status(self, task_id: int) -> str | None:
        try:
            raw = await self.bitrix.get_task(task_id, select=["ID", "STATUS"])
        except Exception:
            return None
        task = _extract_task_detail(raw)
        status = _first(task, "status", "STATUS")
        return str(status) if status not in (None, "") else None


def build_task_closure_draft_from_args(task: AgentTask, args: dict[str, Any]) -> BitrixTaskClosureDraft:
    args = args or {}
    task_id = _optional_int(args.get("task_id") or args.get("taskId") or args.get("id"))
    task_query = _compact(str(args.get("task_query") or args.get("query") or ""))
    result_text = _compact(
        str(args.get("result_text") or args.get("result") or args.get("completion_result") or "")
    )

    contract_errors: list[str] = []
    params: dict[str, Any] = {}
    if task_id is not None:
        params["task_id"] = task_id
    elif task_query:
        params["task_query"] = task_query
    else:
        contract_errors.append("task_closure requires task_id or task_query")

    if result_text:
        params["result_text"] = result_text
    else:
        contract_errors.append("task_closure.result_text is required")

    summary = _summary(task_id=task_id, task_query=task_query, result_text=result_text)
    return BitrixTaskClosureDraft(
        params=params,
        summary=summary,
        contract_errors=_unique(contract_errors),
        notes=[f"Запрос пользователя #{task.user.id} подготовил LLM-субагент Bitrix24."] if task.user.id else [],
    )


def _summary(*, task_id: int | None, task_query: str, result_text: str) -> str:
    target = f"задачу #{task_id}" if task_id is not None else f"задачу по запросу `{task_query}`"
    result_part = _truncate(result_text, 120) if result_text else "без текста результата"
    return f"закрыть {target} с результатом: {result_part}"


def _is_allowed_project(group_id: int | None) -> bool:
    auto_project_id = get_settings().quality_control_auto_manage_project_id
    if auto_project_id is None:
        return True
    return group_id == auto_project_id


def _task_payload(task: QualityTask) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "responsible_id": task.responsible_id,
        "group_id": task.group_id,
    }


def _extract_task_detail(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        task = result.get("task")
        if isinstance(task, dict):
            return task
        return result
    return {}


def _first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _optional_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(str(value).strip().lstrip("#"))
    except (TypeError, ValueError):
        return None


def _unique_ints(values: list[int | None]) -> list[int]:
    result: list[int] = []
    for value in values:
        if value is None or value in result:
            continue
        result.append(value)
    return result


def _compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 40)].rstrip() + "\n...[обрезано]..."


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
