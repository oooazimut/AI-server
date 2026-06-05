from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentTask, ModelUsageRecord
from ai_server.settings import get_settings


TASK_CLOSURE_PENDING_METHOD = "ai_server.task_closure"
MOSCOW_TZ = timezone(timedelta(hours=3))


class TaskClosureError(RuntimeError):
    pass


class TaskClosureLLM(Protocol):
    async def decide(
        self,
        *,
        params: dict[str, Any],
        current_user_id: int | None,
        tool_results: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
        policy: dict[str, Any],
    ) -> "TaskClosureDecision":
        pass


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


@dataclass(frozen=True)
class TaskClosureToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class TaskClosureDecision:
    status: str
    answer: str
    tool_calls: list[TaskClosureToolCall]
    confidence: float = 0.5
    raw: dict[str, Any] = field(default_factory=dict)
    model_usage: ModelUsageRecord | None = None


class LLMTaskClosureService:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def decide(
        self,
        *,
        params: dict[str, Any],
        current_user_id: int | None,
        tool_results: list[dict[str, Any]],
        tool_definitions: list[dict[str, Any]],
        policy: dict[str, Any],
    ) -> TaskClosureDecision:
        settings = get_settings()
        if not settings.llm_configured:
            raise TaskClosureError("task_closure_llm_not_configured")

        completion = await self.client.complete(
            agent_id="bitrix24",
            messages=[
                {"role": "system", "content": TASK_CLOSURE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "pending_action": {
                                "params": params,
                                "current_user_id": current_user_id,
                                "current_datetime": datetime.now(MOSCOW_TZ).isoformat(),
                            },
                            "policy": policy,
                            "tools": tool_definitions,
                            "tool_results": tool_results,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        decision = _parse_task_closure_decision(completion.json_content())
        usage = completion.model_usage.model_copy(update={"role": "task_closure"})
        return TaskClosureDecision(
            status=decision.status,
            answer=decision.answer,
            tool_calls=decision.tool_calls,
            confidence=decision.confidence,
            raw=decision.raw,
            model_usage=usage,
        )


class TaskClosureService:
    def __init__(
        self,
        bitrix: BitrixClient,
        portal_search: PortalSearchIndex,
        *,
        actor_bitrix: BitrixClient | None = None,
        llm: TaskClosureLLM | None = None,
    ) -> None:
        self.bitrix = bitrix
        self.actor_bitrix = actor_bitrix or bitrix
        self.portal_search = portal_search
        self.llm = llm or LLMTaskClosureService()

    async def execute(
        self,
        params: dict[str, Any],
        *,
        current_user_id: int | None,
    ) -> dict[str, Any]:
        draft = build_task_closure_draft_from_args(
            AgentTask(
                task_id="pending_task_closure",
                request="",
                user={"id": str(current_user_id) if current_user_id else None},
            ),
            params,
        )
        if not draft.is_ready:
            return {
                "status": "contract_violation",
                "error": "LLM called task_closure with arguments outside the tool contract.",
                **draft.as_action_details(),
            }

        settings = get_settings()
        tool_results: list[dict[str, Any]] = []
        context: dict[str, Any] = {"tasks": {}, "writes": []}
        model_usage: list[ModelUsageRecord] = []

        for step in range(1, 8):
            decision = await self.llm.decide(
                params=draft.params,
                current_user_id=current_user_id,
                tool_results=tool_results,
                tool_definitions=_task_closure_tool_definitions(),
                policy=_task_closure_policy_context(settings),
            )
            if decision.model_usage:
                model_usage.append(decision.model_usage)

            tool_calls = decision.tool_calls or [TaskClosureToolCall(name="none")]
            if all(call.name == "none" for call in tool_calls):
                return _final_from_decision(
                    decision,
                    context=context,
                    tool_results=tool_results,
                    model_usage=model_usage,
                    step=step,
                )

            for tool_call in tool_calls:
                tool_result = await self._execute_tool(
                    tool_call,
                    params=draft.params,
                    current_user_id=current_user_id,
                    context=context,
                )
                tool_results.append(tool_result)
                _update_task_closure_context(context, tool_result)
                if tool_result["status"] in {"contract_violation", "denied", "error", "invalid_tool_call"}:
                    return _final_from_tool_error(
                        tool_result,
                        decision=decision,
                        context=context,
                        tool_results=tool_results,
                        model_usage=model_usage,
                        step=step,
                    )

            if _has_write_call(tool_calls) and decision.status in {"completed", "executed"}:
                return _final_from_writes(
                    decision,
                    context=context,
                    tool_results=tool_results,
                    model_usage=model_usage,
                    step=step,
                )

        return {
            "status": "error",
            "message": "LLM-режим Битрикс24-специалиста для закрытия задачи превысил лимит шагов.",
            "tool_results": tool_results,
            "model_usage": _usage_payload(model_usage),
        }

    async def _execute_tool(
        self,
        tool_call: TaskClosureToolCall,
        *,
        params: dict[str, Any],
        current_user_id: int | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        name = tool_call.name
        if name == "none":
            return {"status": "ok", "tool": "none", "data": {}}

        if name == "bitrix_task_get":
            return await self._tool_task_get(tool_call.args, params=params)
        if name == "bitrix_task_search":
            return self._tool_task_search(tool_call.args, params=params)
        if name == "bitrix_task_result_add":
            return await self._tool_task_result_add(
                tool_call.args,
                params=params,
                current_user_id=current_user_id,
                context=context,
            )
        if name == "bitrix_task_complete":
            return await self._tool_task_complete(
                tool_call.args,
                current_user_id=current_user_id,
                context=context,
            )
        if name == "bitrix_task_approve":
            return await self._tool_task_state_change(
                tool_call.args,
                tool="bitrix_task_approve",
                current_user_id=current_user_id,
                context=context,
            )
        if name == "bitrix_task_disapprove":
            return await self._tool_task_state_change(
                tool_call.args,
                tool="bitrix_task_disapprove",
                current_user_id=current_user_id,
                context=context,
            )
        if name == "bitrix_task_renew":
            return await self._tool_task_state_change(
                tool_call.args,
                tool="bitrix_task_renew",
                current_user_id=current_user_id,
                context=context,
            )
        if name == "bitrix_task_comment_add":
            return await self._tool_task_comment_add(
                tool_call.args,
                current_user_id=current_user_id,
                context=context,
            )
        if name == "bitrix_notify_user":
            return await self._tool_notify_user(tool_call.args, context=context)

        return {
            "status": "invalid_tool_call",
            "tool": name,
            "error": f"unknown task-closure tool: {name}",
        }

    async def _tool_task_get(self, args: dict[str, Any], *, params: dict[str, Any]) -> dict[str, Any]:
        task_id = _optional_int(args.get("task_id")) or _optional_int(params.get("task_id"))
        if task_id is None:
            return {
                "status": "contract_violation",
                "tool": "bitrix_task_get",
                "error": "bitrix_task_get requires task_id.",
            }
        try:
            raw = await self.bitrix.get_task(
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
        except Exception as exc:
            return _tool_exception("bitrix_task_get", exc, {"task_id": task_id})

        task = _extract_task_detail(raw)
        if not task:
            return {
                "status": "not_found",
                "tool": "bitrix_task_get",
                "data": {"task_id": task_id},
            }
        return {
            "status": "ok",
            "tool": "bitrix_task_get",
            "data": {"task_id": task_id, "task": task},
        }

    def _tool_task_search(self, args: dict[str, Any], *, params: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or params.get("task_query") or params.get("query") or "").strip()
        if not query:
            return {
                "status": "contract_violation",
                "tool": "bitrix_task_search",
                "error": "bitrix_task_search requires query.",
            }

        candidates: list[dict[str, Any]] = []
        for item in self.portal_search.search(query, entity_types={"task"}, limit=20):
            task_id = _optional_int(item.entity_id)
            if task_id is None:
                continue
            metadata = item.metadata if isinstance(item.metadata, dict) else {}
            candidates.append(
                {
                    "id": task_id,
                    "title": item.title,
                    "url": item.url,
                    "status": str(metadata.get("status") or ""),
                    "responsible_id": _optional_int(metadata.get("responsible_id")),
                    "group_id": _optional_int(metadata.get("group_id")),
                }
            )
        return {
            "status": "ok",
            "tool": "bitrix_task_search",
            "data": {"query": query, "candidates": candidates},
        }

    async def _tool_task_result_add(
        self,
        args: dict[str, Any],
        *,
        params: dict[str, Any],
        current_user_id: int | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        task_id, policy_error = _write_target(args, context=context, current_user_id=current_user_id)
        if policy_error:
            return {"tool": "bitrix_task_result_add", **policy_error}

        text = str(args.get("text") or args.get("result_text") or "").strip()
        if not text and _truthy(args.get("use_pending_result_text")):
            text = str(params.get("result_text") or "").strip()
        if not text:
            return {
                "status": "contract_violation",
                "tool": "bitrix_task_result_add",
                "error": "bitrix_task_result_add requires text or use_pending_result_text=true.",
                "data": {"task_id": task_id},
            }
        stored_text = text
        if current_user_id:
            stored_text = (
                text.rstrip()
                + f"\n\n[Отправлено через AI-server пользователем Bitrix #{current_user_id}]"
            )

        try:
            result = await self.bitrix.add_task_result(task_id, stored_text)
        except Exception as exc:
            return _tool_exception("bitrix_task_result_add", exc, {"task_id": task_id})
        return {
            "status": "ok",
            "tool": "bitrix_task_result_add",
            "data": {"task_id": task_id, "action": "add_result", "result": result},
        }

    async def _tool_task_complete(
        self,
        args: dict[str, Any],
        *,
        current_user_id: int | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        task_id, policy_error = _write_target(args, context=context, current_user_id=current_user_id)
        if policy_error:
            return {"tool": "bitrix_task_complete", **policy_error}
        try:
            result = await self.bitrix.complete_task(task_id)
        except Exception as exc:
            return _tool_exception("bitrix_task_complete", exc, {"task_id": task_id})
        return {
            "status": "ok",
            "tool": "bitrix_task_complete",
            "data": {"task_id": task_id, "action": "complete", "result": result},
        }

    async def _tool_task_state_change(
        self,
        args: dict[str, Any],
        *,
        tool: str,
        current_user_id: int | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        task_id, policy_error = _write_target(args, context=context, current_user_id=current_user_id)
        if policy_error:
            return {"tool": tool, **policy_error}
        try:
            if tool == "bitrix_task_approve":
                result = await self.actor_bitrix.approve_task(task_id)
                action = "approve"
            elif tool == "bitrix_task_disapprove":
                result = await self.actor_bitrix.disapprove_task(task_id)
                action = "disapprove"
            else:
                result = await self.actor_bitrix.renew_task(task_id)
                action = "renew"
        except Exception as exc:
            return _tool_exception(tool, exc, {"task_id": task_id})
        return {
            "status": "ok",
            "tool": tool,
            "data": {"task_id": task_id, "action": action, "result": result},
        }

    async def _tool_task_comment_add(
        self,
        args: dict[str, Any],
        *,
        current_user_id: int | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        task_id, policy_error = _write_target(args, context=context, current_user_id=current_user_id)
        if policy_error:
            return {"tool": "bitrix_task_comment_add", **policy_error}
        message = str(args.get("message") or "").strip()
        if not message:
            return {
                "status": "contract_violation",
                "tool": "bitrix_task_comment_add",
                "error": "bitrix_task_comment_add requires message.",
                "data": {"task_id": task_id},
            }
        try:
            result = await self.actor_bitrix.add_task_comment(task_id=task_id, message=message)
        except Exception as exc:
            return _tool_exception("bitrix_task_comment_add", exc, {"task_id": task_id})
        return {
            "status": "ok",
            "tool": "bitrix_task_comment_add",
            "data": {"task_id": task_id, "action": "comment", "result": result},
        }

    async def _tool_notify_user(self, args: dict[str, Any], *, context: dict[str, Any]) -> dict[str, Any]:
        user_id = _optional_int(args.get("user_id"))
        message = str(args.get("message") or "").strip()
        if user_id is None or not message:
            return {
                "status": "contract_violation",
                "tool": "bitrix_notify_user",
                "error": "bitrix_notify_user requires user_id and message.",
            }

        if not _can_notify_user(user_id, context=context):
            return {
                "status": "denied",
                "tool": "bitrix_notify_user",
                "reason": "notify target is outside task responsible/director policy.",
                "data": {"user_id": user_id},
            }

        try:
            result = await self.actor_bitrix.notify_user(
                user_id=user_id,
                message=message,
                tag="ai_server_task_closure",
                sub_tag=f"task:{_first_known_task_id(context) or 'unknown'}",
            )
        except Exception as exc:
            return _tool_exception("bitrix_notify_user", exc, {"user_id": user_id})
        return {
            "status": "ok",
            "tool": "bitrix_notify_user",
            "data": {"user_id": user_id, "action": "notify", "result": result},
        }


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


def _task_closure_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "bitrix_task_search",
            "type": "read",
            "description": "Найти кандидатов задач в локальном поисковом индексе, если pending_action содержит task_query, а не task_id.",
            "parameters": {"query": "string"},
        },
        {
            "name": "bitrix_task_get",
            "type": "read",
            "description": "Прочитать карточку задачи Bitrix24. Перед любой записью обязательно прочитай задачу.",
            "parameters": {"task_id": "integer"},
        },
        {
            "name": "bitrix_task_result_add",
            "type": "write",
            "description": "Добавить результат выполнения в задачу. Можно передать use_pending_result_text=true, чтобы использовать текст из pending_action.",
            "parameters": {"task_id": "integer", "text": "string", "use_pending_result_text": "boolean"},
        },
        {
            "name": "bitrix_task_complete",
            "type": "write",
            "description": "Завершить задачу.",
            "parameters": {"task_id": "integer"},
        },
        {
            "name": "bitrix_task_approve",
            "type": "write",
            "description": "Согласовать задачу, если после завершения она находится на контроле.",
            "parameters": {"task_id": "integer"},
        },
        {
            "name": "bitrix_task_disapprove",
            "type": "write",
            "description": "Отклонить результат задачи, если она находится на контроле и результат недостаточен.",
            "parameters": {"task_id": "integer"},
        },
        {
            "name": "bitrix_task_renew",
            "type": "write",
            "description": "Вернуть задачу в работу, если она ошибочно закрыта или результат недостаточен.",
            "parameters": {"task_id": "integer"},
        },
        {
            "name": "bitrix_task_comment_add",
            "type": "write",
            "description": "Добавить комментарий в задачу.",
            "parameters": {"task_id": "integer", "message": "string"},
        },
        {
            "name": "bitrix_notify_user",
            "type": "write",
            "description": "Отправить уведомление ответственному или директору по политике контроля качества.",
            "parameters": {"user_id": "integer", "message": "string"},
        },
        {
            "name": "none",
            "type": "control",
            "description": "Не вызывать tools и вернуть финальное решение.",
            "parameters": {},
        },
    ]


def _parse_task_closure_decision(data: dict[str, Any]) -> TaskClosureDecision:
    allowed_tools = {
        "bitrix_task_search",
        "bitrix_task_get",
        "bitrix_task_result_add",
        "bitrix_task_complete",
        "bitrix_task_approve",
        "bitrix_task_disapprove",
        "bitrix_task_renew",
        "bitrix_task_comment_add",
        "bitrix_notify_user",
        "none",
    }
    raw_tool_calls = data.get("tool_calls")
    tool_calls: list[TaskClosureToolCall] = []
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if name not in allowed_tools:
                continue
            args = raw_call.get("args") if isinstance(raw_call.get("args"), dict) else {}
            tool_calls.append(
                TaskClosureToolCall(
                    name=name,
                    args=args,
                    summary=str(raw_call.get("summary") or "").strip(),
                )
            )
    if not tool_calls:
        tool_calls = [TaskClosureToolCall(name="none")]

    status = str(data.get("status") or "completed").strip()
    if status not in {
        "continue",
        "completed",
        "executed",
        "needs_clarification",
        "ambiguous",
        "not_found",
        "denied",
        "failed",
    }:
        status = "completed"
    return TaskClosureDecision(
        status=status,
        answer=str(data.get("answer") or "").strip(),
        tool_calls=tool_calls,
        confidence=_confidence(data.get("confidence")),
        raw=data,
    )


def _task_closure_policy_context(settings: Any) -> dict[str, Any]:
    return {
        "auto_manage_project_id": settings.quality_control_auto_manage_project_id,
        "write_allowed_user_ids": settings.resolved_agent_write_allowed_user_ids,
        "quality_actor_user_id": settings.quality_control_actor_user_id,
        "notify_responsible": settings.quality_control_notify_responsible,
        "notify_director": settings.quality_control_notify_director,
        "director_user_ids": settings.resolved_quality_control_director_user_ids,
        "exempt_responsible_user_ids": settings.resolved_quality_control_exempt_responsible_user_ids,
    }


def _update_task_closure_context(context: dict[str, Any], tool_result: dict[str, Any]) -> None:
    data = tool_result.get("data") if isinstance(tool_result.get("data"), dict) else {}
    if tool_result.get("tool") == "bitrix_task_get" and tool_result.get("status") == "ok":
        task = data.get("task") if isinstance(data.get("task"), dict) else {}
        task_id = _optional_int(data.get("task_id")) or _task_id_from_detail(task)
        if task_id is not None:
            context.setdefault("tasks", {})[str(task_id)] = task
    action = data.get("action")
    if isinstance(action, str) and action:
        context.setdefault("writes", []).append(
            {
                "tool": tool_result.get("tool"),
                "task_id": data.get("task_id"),
                "action": action,
                "status": tool_result.get("status"),
            }
        )


def _final_from_tool_error(
    tool_result: dict[str, Any],
    *,
    decision: TaskClosureDecision,
    context: dict[str, Any],
    tool_results: list[dict[str, Any]],
    model_usage: list[ModelUsageRecord],
    step: int,
) -> dict[str, Any]:
    status = str(tool_result.get("status") or "error")
    return {
        "status": status,
        "message": decision.answer or str(tool_result.get("reason") or tool_result.get("error") or "Tool failed."),
        "reason": tool_result.get("reason"),
        "error": tool_result.get("error"),
        "task": _first_task_payload(context),
        "tool_results": tool_results,
        "llm_steps": step,
        "model_usage": _usage_payload(model_usage),
    }


def _final_from_decision(
    decision: TaskClosureDecision,
    *,
    context: dict[str, Any],
    tool_results: list[dict[str, Any]],
    model_usage: list[ModelUsageRecord],
    step: int,
) -> dict[str, Any]:
    outcome = str(decision.raw.get("outcome") or "").strip()
    status = decision.status
    if status in {"completed", "executed"} and outcome == "already_closed":
        status = "executed"
    return {
        "status": status,
        "outcome": outcome,
        "message": decision.answer,
        "task": _first_task_payload(context),
        "tool_results": tool_results,
        "llm_steps": step,
        "model_usage": _usage_payload(model_usage),
    }


def _final_from_writes(
    decision: TaskClosureDecision,
    *,
    context: dict[str, Any],
    tool_results: list[dict[str, Any]],
    model_usage: list[ModelUsageRecord],
    step: int,
) -> dict[str, Any]:
    writes = context.get("writes") if isinstance(context.get("writes"), list) else []
    write_actions = {str(write.get("action")) for write in writes if isinstance(write, dict)}
    if "complete" in write_actions or "approve" in write_actions:
        outcome = "closed"
    elif "disapprove" in write_actions or "renew" in write_actions:
        outcome = "needs_revision"
    elif "comment" in write_actions:
        outcome = "commented"
    else:
        outcome = "updated"
    return {
        "status": "executed",
        "outcome": str(decision.raw.get("outcome") or outcome),
        "message": decision.answer,
        "task": _first_task_payload(context),
        "actions": writes,
        "approved": "approve" in write_actions,
        "tool_results": tool_results,
        "llm_steps": step,
        "model_usage": _usage_payload(model_usage),
    }


def _has_write_call(tool_calls: list[TaskClosureToolCall]) -> bool:
    write_tools = {
        "bitrix_task_result_add",
        "bitrix_task_complete",
        "bitrix_task_approve",
        "bitrix_task_disapprove",
        "bitrix_task_renew",
        "bitrix_task_comment_add",
        "bitrix_notify_user",
    }
    return any(call.name in write_tools for call in tool_calls)


def _write_target(
    args: dict[str, Any],
    *,
    context: dict[str, Any],
    current_user_id: int | None,
) -> tuple[int, dict[str, Any] | None]:
    task_id = _optional_int(args.get("task_id"))
    if task_id is None:
        return 0, {
            "status": "contract_violation",
            "error": "write tool requires task_id.",
        }

    task = _known_task(context, task_id)
    if not task:
        return task_id, {
            "status": "contract_violation",
            "error": "write tool requires task data read through bitrix_task_get first.",
            "data": {"task_id": task_id},
        }

    policy_error = _write_policy_error(task, current_user_id)
    if policy_error:
        return task_id, policy_error
    return task_id, None


def _write_policy_error(task: dict[str, Any], current_user_id: int | None) -> dict[str, Any] | None:
    if current_user_id is None:
        return {
            "status": "denied",
            "reason": "Не вижу ID текущего пользователя Bitrix, поэтому не могу закрывать задачу.",
            "data": {"task": _task_payload_from_detail(task)},
        }
    group_id = _task_group_id(task)
    if not _is_allowed_project(group_id):
        return {
            "status": "denied",
            "reason": (
                "Закрытие задач через AI-server пока разрешено только в проекте "
                f"#{get_settings().quality_control_auto_manage_project_id}."
            ),
            "data": {"task": _task_payload_from_detail(task)},
        }
    responsible_id = _task_responsible_id(task)
    if current_user_id != responsible_id and current_user_id not in get_settings().resolved_agent_write_allowed_user_ids:
        return {
            "status": "denied",
            "reason": "Закрывать задачу через AI-server может её исполнитель или администратор агента.",
            "data": {"task": _task_payload_from_detail(task)},
        }
    return None


def _can_notify_user(user_id: int, *, context: dict[str, Any]) -> bool:
    settings = get_settings()
    if user_id in settings.resolved_quality_control_director_user_ids:
        return True
    for task in _known_tasks(context):
        if user_id == _task_responsible_id(task):
            return True
    return False


def _known_task(context: dict[str, Any], task_id: int) -> dict[str, Any]:
    tasks = context.get("tasks") if isinstance(context.get("tasks"), dict) else {}
    task = tasks.get(str(task_id)) if isinstance(tasks, dict) else None
    return task if isinstance(task, dict) else {}


def _known_tasks(context: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = context.get("tasks") if isinstance(context.get("tasks"), dict) else {}
    return [task for task in tasks.values() if isinstance(task, dict)]


def _first_known_task_id(context: dict[str, Any]) -> int | None:
    for task in _known_tasks(context):
        task_id = _task_id_from_detail(task)
        if task_id is not None:
            return task_id
    return None


def _first_task_payload(context: dict[str, Any]) -> dict[str, Any]:
    for task in _known_tasks(context):
        return _task_payload_from_detail(task)
    return {}


def _task_payload_from_detail(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _task_id_from_detail(task),
        "title": str(_first_ci(task, "title", "TITLE") or ""),
        "status": str(_first_ci(task, "status", "STATUS") or ""),
        "responsible_id": _task_responsible_id(task),
        "group_id": _task_group_id(task),
    }


def _task_id_from_detail(task: dict[str, Any]) -> int | None:
    return _optional_int(_first_ci(task, "id", "ID", "taskId", "TASK_ID"))


def _task_group_id(task: dict[str, Any]) -> int | None:
    return _optional_int(_first_ci(task, "groupId", "GROUP_ID", "group_id"))


def _task_responsible_id(task: dict[str, Any]) -> int | None:
    return _optional_int(_first_ci(task, "responsibleId", "RESPONSIBLE_ID", "responsible_id"))


def _summary(*, task_id: int | None, task_query: str, result_text: str) -> str:
    target = f"задачу #{task_id}" if task_id is not None else f"задачу по запросу `{task_query}`"
    result_part = _truncate(result_text, 120) if result_text else "без текста результата"
    return f"закрыть {target} с результатом: {result_part}"


def _is_allowed_project(group_id: int | None) -> bool:
    auto_project_id = get_settings().quality_control_auto_manage_project_id
    if auto_project_id is None:
        return True
    return group_id == auto_project_id


def _tool_exception(tool: str, exc: Exception, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "error",
        "tool": tool,
        "error": f"{type(exc).__name__}: {exc}",
        "data": data,
    }


def _extract_task_detail(result: object) -> dict[str, Any]:
    if isinstance(result, dict):
        task = result.get("task")
        if isinstance(task, dict):
            return task
        return result
    return {}


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


def _optional_int(value: object) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(str(value).strip().lstrip("#"))
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "да"}
    return bool(value)


def _confidence(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(number, 0.0), 1.0)


def _usage_payload(model_usage: list[ModelUsageRecord]) -> list[dict[str, Any]]:
    return [usage.model_dump() for usage in model_usage]


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


TASK_CLOSURE_SYSTEM_PROMPT = """
Ты Битрикс24-специалист в режиме подтверждённого закрытия задачи.

Пользователь уже подтвердил pending-action. Backend не решает за тебя, какую
задачу закрывать, достаточно ли результата и какие Bitrix tools вызывать.
Backend только исполняет выбранные тобой tools и применяет security guardrails.

Рабочий порядок:
1. Если pending_action содержит task_query, сначала вызови `bitrix_task_search`,
   затем выбери одного кандидата и вызови `bitrix_task_get`.
2. Если pending_action содержит task_id, сначала вызови `bitrix_task_get`.
3. Перед любой записью обязательно прочитай задачу через `bitrix_task_get`.
4. Сам проверь права по policy и карточке задачи: обычный пользователь закрывает
   свою задачу; администраторы из policy.write_allowed_user_ids могут шире.
5. Сравни pending_action.params.result_text с названием/описанием задачи по смыслу.
   Не придумывай требований, которых нет в задаче.
6. Если результат достаточный, добавь результат через `bitrix_task_result_add`
   с `use_pending_result_text=true`, затем заверши через `bitrix_task_complete`.
7. Если после завершения нужно узнать финальный статус, снова вызови
   `bitrix_task_get`. Если задача ждёт контроля и policy/права позволяют,
   вызови `bitrix_task_approve`.
8. Если результат недостаточный, не закрывай задачу. Верни
   `needs_clarification` с понятным ответом или, если нужно зафиксировать это в
   Bitrix, вызови `bitrix_task_comment_add`/`bitrix_task_disapprove`/`bitrix_task_renew`
   и при необходимости `bitrix_notify_user`.
9. Если задача уже закрыта, верни `status="completed"`, `outcome="already_closed"`
   и tool `none`.

Верни только JSON без markdown:
{
  "status": "continue|completed|needs_clarification|ambiguous|not_found|denied|failed",
  "outcome": "closed|already_closed|needs_revision|commented|updated",
  "answer": "короткий ответ человеку",
  "confidence": 0.0,
  "tool_calls": [
    {
      "name": "bitrix_task_search|bitrix_task_get|bitrix_task_result_add|bitrix_task_complete|bitrix_task_approve|bitrix_task_disapprove|bitrix_task_renew|bitrix_task_comment_add|bitrix_notify_user|none",
      "args": {},
      "summary": ""
    }
  ]
}
""".strip()
