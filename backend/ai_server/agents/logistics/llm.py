from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from ai_server.agents.specialist_llm_shared import (
    DIALOG_HISTORY_PROMPT_FRAGMENT,
    SKILLS_PROMPT_FRAGMENT,
    allowed_tool_definitions,
    compact_tool_result,
    decision_status,
    load_instructions,
    result_status,
    retrieval_context,
    safe_context,
    skills_context,
)
from ai_server.llm import LLMClient, OpenAICompatibleLLMClient
from ai_server.models import AgentManifest, AgentTask, ModelUsageRecord, ToolResult
from ai_server.retrieval import RetrievalHit
from ai_server.utils import confidence

logger = logging.getLogger(__name__)

ALLOWED_TOOL_NAMES = {
    "vehicle_usage_context",
    "vehicle_usage_get_operators",
    "vehicle_usage_set_operators",
    "vehicle_usage_start_day",
    "vehicle_usage_get_report",
    "vehicle_usage_get_employee_period_report",
    "vehicle_usage_get_vehicle_period_report",
    "vehicle_usage_save_draft",
    "vehicle_usage_save_report",
    "vehicle_usage_update_report",
    "vehicle_usage_cancel_day",
    "none",
}


class LogisticsAgentLLM(Protocol):
    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        dialog_history: list[dict[str, str]] | None = None,
        available_skills: list | None = None,
    ) -> LogisticsLLMDecisionResult:
        pass

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: LogisticsLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> LogisticsLLMFinalResult:
        pass


@dataclass(frozen=True)
class LogisticsLLMToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass(frozen=True)
class LogisticsLLMDecision:
    status: str
    answer: str
    tool_calls: list[LogisticsLLMToolCall]
    confidence: float = 0.5


@dataclass(frozen=True)
class LogisticsLLMDecisionResult:
    decision: LogisticsLLMDecision
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LogisticsLLMFinalResult:
    status: str
    answer: str
    model_usage: ModelUsageRecord
    raw: dict[str, Any] = field(default_factory=dict)


class LogisticsLLMService:
    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or OpenAICompatibleLLMClient()

    async def decide(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        retrieval_hits: list[RetrievalHit],
        tool_definitions: list[dict[str, Any]],
        tool_results: list[ToolResult] | None = None,
        dialog_history: list[dict[str, str]] | None = None,
        available_skills: list | None = None,
    ) -> LogisticsLLMDecisionResult:
        instructions = load_instructions(manifest)
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _decision_system_prompt(instructions)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent": {
                                "id": manifest.id,
                                "name": manifest.name,
                                "handoff_description": manifest.handoff_description,
                            },
                            "request": task.request,
                            "user": task.user.model_dump(),
                            "context": safe_context(task.context),
                            "current_datetime": datetime.now(UTC).astimezone().isoformat(),
                            "available_skills": skills_context(available_skills or []),
                            "dialog_history": dialog_history or [],
                            "retrieval_context": retrieval_context(retrieval_hits),
                            "tools": allowed_tool_definitions(tool_definitions, ALLOWED_TOOL_NAMES),
                            "tool_results": [compact_tool_result(result) for result in (tool_results or [])],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        return LogisticsLLMDecisionResult(
            decision=_parse_decision(completion.json_content()),
            model_usage=completion.model_usage,
            raw=completion.raw,
        )

    async def compose(
        self,
        *,
        manifest: AgentManifest,
        task: AgentTask,
        decision: LogisticsLLMDecision,
        tool_results: list[ToolResult],
        approval_actions: list[dict[str, Any]] | None = None,
    ) -> LogisticsLLMFinalResult:
        operators_answer = _compose_operators_answer(tool_results)
        if operators_answer is not None:
            return LogisticsLLMFinalResult(
                status="completed",
                answer=operators_answer,
                model_usage=ModelUsageRecord(
                    agent_id=manifest.id,
                    provider="",
                    model="",
                    status="skipped",
                    notes=["deterministic_vehicle_usage_get_operators"],
                ),
            )
        report_answer = _compose_get_report_answer(tool_results)
        if report_answer is not None:
            return LogisticsLLMFinalResult(
                status="completed",
                answer=report_answer,
                model_usage=ModelUsageRecord(
                    agent_id=manifest.id,
                    provider="",
                    model="",
                    status="skipped",
                    notes=["deterministic_vehicle_usage_get_report"],
                ),
            )
        period_answer = _compose_period_report_answer(tool_results)
        if period_answer is not None:
            return LogisticsLLMFinalResult(
                status="completed",
                answer=period_answer,
                model_usage=ModelUsageRecord(
                    agent_id=manifest.id,
                    provider="",
                    model="",
                    status="skipped",
                    notes=["deterministic_vehicle_usage_period_report"],
                ),
            )
        save_answer = _compose_save_report_answer(tool_results)
        if save_answer is not None:
            return LogisticsLLMFinalResult(
                status="completed",
                answer=save_answer,
                model_usage=ModelUsageRecord(
                    agent_id=manifest.id,
                    provider="",
                    model="",
                    status="skipped",
                    notes=["deterministic_vehicle_usage_save_report"],
                ),
            )
        update_answer = _compose_update_report_answer(tool_results)
        if update_answer is not None:
            return LogisticsLLMFinalResult(
                status="completed",
                answer=update_answer,
                model_usage=ModelUsageRecord(
                    agent_id=manifest.id,
                    provider="",
                    model="",
                    status="skipped",
                    notes=["deterministic_vehicle_usage_update_report"],
                ),
            )
        completion = await self.client.complete(
            agent_id=manifest.id,
            messages=[
                {"role": "system", "content": _compose_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "initial_decision": _decision_dict(decision),
                            "tool_results": [compact_tool_result(result) for result in tool_results],
                            "approval_actions": approval_actions or [],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            json_mode=True,
        )
        parsed = completion.json_content()
        return LogisticsLLMFinalResult(
            status=result_status(parsed.get("status")),
            answer=str(parsed.get("answer") or "").strip() or "Готово.",
            model_usage=completion.model_usage,
            raw=completion.raw,
        )


def logistics_llm_failure_result(message: str, agent_id: str = "logistics") -> LogisticsLLMFinalResult:
    return LogisticsLLMFinalResult(
        status="failed",
        answer=f"Не смог обработать запрос Логиста через LLM: {message}",
        model_usage=ModelUsageRecord(
            agent_id=agent_id,
            provider="",
            model="",
            status="error",
            notes=[message],
        ),
    )


def _decision_system_prompt(instructions: str = "") -> str:
    extra = f"\n\nДополнительные инструкции:\n{instructions}" if instructions else ""
    return (
        "Ты LLM-специалист Логист внутри корпоративного AI-server. "
        "Твоя зона: ежедневный учет служебных автомобилей, статусы сотрудников, смены, выезды, "
        "утренние отчеты и уточнения к ним. "
        "Ты не вызываешь Bitrix напрямую, не пишешь в чат сам и не пишешь в базу напрямую: выбирай vehicle_usage tools; state хранится в PostgreSQL schema logistics. "
        "Backend-tools только читают/пишут структурированные данные; "
        "они не решают, что имел в виду человек. "
        f"{SKILLS_PROMPT_FRAGMENT}"
        f"{DIALOG_HISTORY_PROMPT_FRAGMENT}"
        "Если человек просит запустить, начать или стартовать отчет по машинам, вызови vehicle_usage_start_day. "
        "Не используй vehicle_usage_get_operators как подготовительный шаг для запуска отчета. "
        "Если человек явно спрашивает, кто сейчас операторы отчета по машинам или просит список операторов, вызови vehicle_usage_get_operators. "
        "Сначала получи vehicle_usage_context, если в tool_results еще нет roster/vehicles/latest_request. "
        "Если человек просит показать отчет за дату или исправить прошлый отчет — сначала вызови vehicle_usage_get_report. "
        "Сам распознавай естественный язык: кто работает, кто в отпуске/болеет/на объекте, какая машина за кем, "
        "является ли ответ подтверждением, исправлением или просьбой начать заново. "
        "При разборе нового ответа оператора не подставляй сотрудников, водителей, машины и статусы из прошлых отчетов, latest_request, day_report или примеров. "
        "В parsed JSON включай назначение машины или статус сотрудника только если это явно сказано в текущем сообщении оператора; недостающие позиции оставляй unknown и уточняй. "
        "Если человек пишет, что отчет сегодня не требуется, день выходной или процедуру надо отменить — вызови vehicle_usage_cancel_day. "
        "Если данных не хватает, не сохраняй финальный отчет: сохрани черновик при необходимости и задай уточнение. "
        "Если задача пришла от scheduler и пора отправить утренний запрос или повторное напоминание, "
        "сформулируй точный текст сообщения в answer и не вызывай send tools. "
        "Если scheduler сообщает, что ответа нет к времени эскалации, сформулируй точный текст уведомления "
        "в answer и не вызывай notify tools. "
        "vehicle_usage_save_report вызывай только когда отчет явно подтвержден человеком или задача от scheduler "
        "содержит уже подтвержденный структурированный отчет. "
        "Для сохранения используй единый JSON: people=[{staff_order, full_name, status, notes}], vehicles=[{vehicle_id или vehicle_name, status, drivers:[full_name], notes}]. "
        "Машина может иметь 0, 1 или несколько сотрудников; сотрудник может быть без машины. Все известные машины должны попасть в vehicles со статусом in_use/idle/repair/not_required/unknown. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human",'
        '"answer":"короткий предварительный ответ",'
        '"confidence":0.0,'
        '"tool_calls":[{"name":"vehicle_usage_context|vehicle_usage_get_operators|vehicle_usage_set_operators|vehicle_usage_start_day|vehicle_usage_get_report|vehicle_usage_get_employee_period_report|vehicle_usage_get_vehicle_period_report|vehicle_usage_save_draft|vehicle_usage_save_report|vehicle_usage_update_report|vehicle_usage_cancel_day|none","args":{},"summary":""}]}.'
        f"{extra}"
    )


def _compose_system_prompt() -> str:
    return (
        "Ты тот же Логист. Сформируй итоговый текст для Переговорщика по результатам vehicle_usage tools. "
        "Не выдумывай сохраненные записи. Если сохранен черновик, попроси проверить/подтвердить. "
        "Если сохранен финальный отчет, скажи кратко что сохранено. "
        "Если задача от scheduler про напоминание или эскалацию, answer должен быть точным текстом сообщения, "
        "которое Переговорщик отправит людям. "
        "ФОРМАТИРОВАНИЕ ОБЯЗАТЕЛЬНО: используй переносы строк (\\n) для структуры. "
        "Для утреннего запроса/напоминания — вежливое обращение + суть в одной строке. "
        "Для подтверждения отчёта — сначала краткий итог, затем список сотрудников построчно "
        "(имя — статус/авто), затем список машин построчно (авто — водитель). "
        "Если сотруднику назначена машина, в подтверждении пиши статус как 'работал / Авто N', а не 'в офисе, Авто N' или 'на авто / Авто N'. "
        "Для эскалации — чёткий сигнал + дата + что именно отсутствует. "
        "Никогда не сваливай всё в одну строку через запятую или пробел. "
        "Верни только JSON-объект без markdown: "
        '{"status":"completed|needs_clarification|needs_human|failed","answer":"ответ человеку"}.'
    )


def _parse_decision(data: dict[str, Any]) -> LogisticsLLMDecision:
    raw_tool_calls = data.get("tool_calls")
    tool_calls: list[LogisticsLLMToolCall] = []
    if isinstance(raw_tool_calls, list):
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                continue
            name = str(raw_call.get("name") or "").strip()
            if name not in ALLOWED_TOOL_NAMES:
                continue
            args = raw_call.get("args") if isinstance(raw_call.get("args"), dict) else {}
            tool_calls.append(
                LogisticsLLMToolCall(
                    name=name,
                    args=args,
                    summary=str(raw_call.get("summary") or "").strip(),
                )
            )
    if not tool_calls:
        tool_calls = [LogisticsLLMToolCall(name="none")]
    return LogisticsLLMDecision(
        status=decision_status(data.get("status")),
        answer=str(data.get("answer") or "").strip(),
        confidence=confidence(data.get("confidence")),
        tool_calls=tool_calls,
    )


def _decision_dict(decision: LogisticsLLMDecision) -> dict[str, Any]:
    return {
        "status": decision.status,
        "answer": decision.answer,
        "confidence": decision.confidence,
        "tool_calls": [{"name": call.name, "args": call.args, "summary": call.summary} for call in decision.tool_calls],
    }


def _compose_operators_answer(tool_results: list[ToolResult]) -> str | None:
    for result in reversed(tool_results):
        if result.tool != "vehicle_usage_get_operators" or str(result.status) != "ok":
            continue
        data = result.data if isinstance(result.data, dict) else {}
        operators = data.get("operators") if isinstance(data.get("operators"), list) else []
        if not operators:
            return "Операторы отчета по машинам не назначены."
        lines = ["Операторы отчета по машинам:"]
        for item in operators:
            if not isinstance(item, dict):
                continue
            user_id = str(item.get("user_id") or "").strip()
            full_name = str(item.get("full_name") or "").strip()
            if full_name and user_id:
                lines.append(f"- {full_name} (Bitrix ID {user_id})")
            elif user_id:
                lines.append(f"- Bitrix ID {user_id}")
        return "\n".join(lines) if len(lines) > 1 else "Операторы отчета по машинам не назначены."
    return None


def _compose_get_report_answer(tool_results: list[ToolResult]) -> str | None:
    for result in reversed(tool_results):
        if result.tool != "vehicle_usage_get_report" or str(result.status) != "ok":
            continue
        data = result.data if isinstance(result.data, dict) else {}
        report_date = str(data.get("report_date") or "").strip()
        employees = data.get("employee_statuses") if isinstance(data.get("employee_statuses"), list) else []
        vehicles = data.get("vehicle_assignments") if isinstance(data.get("vehicle_assignments"), list) else []
        drivers = data.get("vehicle_drivers") if isinstance(data.get("vehicle_drivers"), list) else []
        if not employees and not vehicles and not drivers:
            return f"Отчет по машинам за {report_date or 'указанную дату'} отсутствует."

        lines = [f"Отчет по машинам за {report_date}:" if report_date else "Отчет по машинам:"]
        employee_lines = [_format_employee_line(item) for item in employees]
        employee_lines = [line for line in employee_lines if line]
        if employee_lines:
            lines.extend(["", "Сотрудники:", *employee_lines])

        vehicle_lines = [_format_vehicle_line(item, drivers) for item in vehicles]
        vehicle_lines = [line for line in vehicle_lines if line]
        if vehicle_lines:
            lines.extend(["", "Машины:", *vehicle_lines])

        request = data.get("request") if isinstance(data.get("request"), dict) else {}
        response_text = str(request.get("response_text") or "").strip()
        if response_text and not vehicle_lines:
            lines.extend(["", "Исходный ответ:", response_text])
        return "\n".join(lines).strip()
    return None


def _compose_period_report_answer(tool_results: list[ToolResult]) -> str | None:
    for result in reversed(tool_results):
        if result.tool not in {
            "vehicle_usage_get_employee_period_report",
            "vehicle_usage_get_vehicle_period_report",
        } or str(result.status) != "ok":
            continue
        data = result.data if isinstance(result.data, dict) else {}
        days = data.get("days") if isinstance(data.get("days"), list) else []
        date_from = str(data.get("date_from") or "").strip()
        date_to = str(data.get("date_to") or "").strip()
        if result.tool == "vehicle_usage_get_employee_period_report":
            subject = str(data.get("employee_name") or "").strip() or "сотруднику"
            title = f"Отчет по {subject} за период {date_from} - {date_to}:"
            lines = [title]
            if not days:
                return f"{title}\n\nДанных за период нет."
            for item in days:
                report_date = item.get("status_date") or item.get("date") or ""
                status = _vehicle_usage_status_label(item.get("status") or "unknown")
                vehicle = item.get("vehicle_name") or item.get("vehicle") or ""
                notes = item.get("notes") or ""
                details = " / ".join(str(value).strip() for value in (status, vehicle, notes) if str(value or "").strip())
                lines.append(f"- {report_date} — {details}")
        else:
            subject = str(data.get("vehicle_name") or "").strip() or "машине"
            title = f"Отчет по {subject} за период {date_from} - {date_to}:"
            lines = [title]
            if not days:
                return f"{title}\n\nДанных за период нет."
            for item in days:
                report_date = item.get("assignment_date") or item.get("date") or ""
                status = _vehicle_usage_status_label(item.get("status") or "unknown")
                drivers = item.get("drivers") if isinstance(item.get("drivers"), list) else []
                driver_text = ", ".join(str(value).strip() for value in drivers if str(value or "").strip())
                notes = item.get("notes") or ""
                details = " / ".join(value for value in (driver_text, str(status).strip(), str(notes).strip()) if value)
                lines.append(f"- {report_date} — {details}")
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        if summary:
            lines.extend(["", "Итог:"])
            labeled_summary: dict[str, int] = {}
            for status, count in summary.items():
                label = _vehicle_usage_status_label(status)
                labeled_summary[label] = labeled_summary.get(label, 0) + int(count or 0)
            for label, count in labeled_summary.items():
                lines.append(f"- {label}: {count}")
        return "\n".join(lines).strip()
    return None


def _compose_update_report_answer(tool_results: list[ToolResult]) -> str | None:
    for result in reversed(tool_results):
        if result.tool != "vehicle_usage_update_report" or str(result.status) != "ok":
            continue
        data = result.data if isinstance(result.data, dict) else {}
        report_date = str(data.get("report_date") or "").strip() or "указанную дату"
        employee_updates = int(data.get("employee_updates") or 0)
        vehicle_updates = int(data.get("vehicle_updates") or 0)
        change_summary = str(data.get("change_summary") or "").strip()
        lines = [f"Отчет по машинам за {report_date} обновлен."]
        details = []
        if employee_updates:
            details.append(f"сотрудники: {employee_updates}")
        if vehicle_updates:
            details.append(f"машины: {vehicle_updates}")
        if details:
            lines.append("Изменено: " + ", ".join(details) + ".")
        if change_summary:
            lines.append(change_summary)
        return "\n".join(lines)
    return None


def _compose_save_report_answer(tool_results: list[ToolResult]) -> str | None:
    for result in reversed(tool_results):
        if result.tool != "vehicle_usage_save_report" or str(result.status) != "ok":
            continue
        data = result.data if isinstance(result.data, dict) else {}
        report_date = str(data.get("request_date") or data.get("report_date") or "").strip() or "указанную дату"
        staff_count = int(data.get("staff_entries_saved") or 0)
        vehicle_count = int(data.get("vehicle_assignments_saved") or 0)
        lines = [f"Финальный отчет по машинам за {report_date} сохранен."]
        details = []
        if staff_count:
            details.append(f"сотрудники: {staff_count}")
        if vehicle_count:
            details.append(f"машины: {vehicle_count}")
        if details:
            lines.append("Сохранено: " + ", ".join(details) + ".")
        return "\n".join(lines)
    return None


def _format_employee_line(item: Any) -> str:
    if isinstance(item, dict):
        name = item.get("full_name") or item.get("employee_name") or item.get("name") or item.get("employee_id")
        status = item.get("status") or ""
        vehicle = item.get("vehicle") or item.get("vehicle_name") or ""
        notes = item.get("notes") or ""
    elif isinstance(item, (list, tuple)):
        name = item[2] if len(item) > 2 else item[0] if item else ""
        status = item[3] if len(item) > 3 else item[1] if len(item) > 1 else ""
        vehicle = ""
        notes = item[4] if len(item) > 4 else item[2] if len(item) > 2 else ""
    else:
        return ""
    status = _vehicle_usage_status_label(status)
    details = " / ".join(str(value).strip() for value in (status, vehicle, notes) if str(value or "").strip())
    return f"- {str(name).strip()} — {details}" if details else f"- {str(name).strip()}"


def _format_vehicle_line(item: Any, drivers: list[Any]) -> str:
    if isinstance(item, dict):
        vehicle = item.get("vehicle_name") or item.get("brand_model") or item.get("vehicle") or item.get("name")
        status = item.get("status") or item.get("assignment_status") or ""
        driver_names = item.get("drivers") if isinstance(item.get("drivers"), list) else []
        if not driver_names:
            driver_names = _driver_names_for_vehicle(vehicle, item.get("vehicle_id"), drivers)
        notes = item.get("notes") or ""
    elif isinstance(item, (list, tuple)):
        vehicle = item[2] if len(item) > 2 else item[1] if len(item) > 1 else item[0] if item else ""
        status = item[6] if len(item) > 6 else ""
        driver_names = [item[5]] if len(item) > 5 and item[5] else []
        notes = item[7] if len(item) > 7 else ""
    else:
        return ""
    pieces = []
    if driver_names:
        pieces.append(", ".join(str(name).strip() for name in driver_names if str(name or "").strip()))
    if status:
        pieces.append(_vehicle_usage_status_label(status))
    if notes:
        note = _vehicle_usage_status_label(notes)
        if note not in pieces and str(notes).strip() not in pieces:
            pieces.append(note)
    details = " / ".join(piece for piece in pieces if piece)
    return f"- {str(vehicle).strip()} — {details}" if details else f"- {str(vehicle).strip()}"


def _vehicle_usage_status_label(value: Any) -> str:
    raw = str(value or "").strip()
    labels = {
        "worked": "работал",
        "on_car": "работал",
        "выезд": "работал",
        "на выезде": "работал",
        "in_use": "в работе",
        "idle": "простой",
        "office": "работал",
        "in_office": "работал",
        "at_office": "работал",
        "work": "работал",
        "working": "работал",
        "shift": "работал",
        "object": "работал",
        "on_object": "работал",
        "site": "работал",
        "on_site": "работал",
        "field": "работал",
        "trip": "работал",
        "работа": "работал",
        "работает": "работал",
        "на работе": "работал",
        "в офисе": "работал",
        "офис": "работал",
        "объект": "работал",
        "на объекте": "работал",
        "на авто": "работал",
        "на машине": "работал",
        "на смене": "работал",
        "vacation": "отпуск",
        "on_leave": "отпуск",
        "leave": "отпуск",
        "holiday": "отпуск",
        "sick": "болеет",
        "day_off": "выходной",
        "not_required": "не требуется",
        "repair": "ремонт",
        "not_working": "не работает",
        "unknown": "неизвестно",
    }
    return labels.get(raw.casefold(), raw)


def _driver_names_for_vehicle(vehicle_name: Any, vehicle_id: Any, drivers: list[Any]) -> list[str]:
    result: list[str] = []
    for item in drivers:
        if not isinstance(item, dict):
            continue
        same_name = vehicle_name and item.get("vehicle_name") == vehicle_name
        same_id = vehicle_id is not None and item.get("vehicle_id") == vehicle_id
        if same_name or same_id:
            name = item.get("full_name") or item.get("employee_name")
            if name:
                result.append(str(name))
    return result
