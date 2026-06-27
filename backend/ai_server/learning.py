from __future__ import annotations

import contextlib
import json
import logging
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from ai_server.models import AgentResult, AgentTask
from ai_server.settings import Settings, get_settings
from ai_server.tracing import trace_id_from_task

logger = logging.getLogger(__name__)

_Subscriber = Callable[[dict[str, Any]], None]


class EventStream:
    """Append-only event journal with in-process subscriber support."""

    schema_version = "learning_event.v1"

    def __init__(
        self,
        settings: Settings | None = None,
        path: Path | str | None = None,
        *,
        enabled: bool | None = None,
        capture_text: bool | None = None,
        max_text_chars: int | None = None,
    ) -> None:
        _settings = settings or get_settings()
        self.path = Path(path) if path is not None else _settings.learning_events_path
        self.enabled = _settings.learning_events_enabled if enabled is None else enabled
        self.capture_text = _settings.learning_events_capture_text if capture_text is None else capture_text
        self.max_text_chars = max_text_chars or _settings.learning_events_max_text_chars
        self._subscribers: list[_Subscriber] = []

    def subscribe(self, callback: _Subscriber) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: _Subscriber) -> None:
        with contextlib.suppress(ValueError):
            self._subscribers.remove(callback)

    def _notify_subscribers(self, event: dict[str, Any]) -> None:
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                logger.exception("EventStream subscriber error")

    def record_event(
        self,
        *,
        event_type: str,
        source: str,
        agent_id: str = "",
        task_id: str = "",
        user_id: str | int | None = None,
        channel: str = "",
        request: str = "",
        response: str = "",
        status: str = "",
        handoff_to: list[str] | None = None,
        actions: list[Any] | None = None,
        model_usage: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"recorded": False, "reason": "disabled"}

        text_captured = bool(self.capture_text)
        event = {
            "schema_version": self.schema_version,
            "id": str(uuid4()),
            "created_at": _now_iso(),
            "event_type": event_type,
            "source": source,
            "agent_id": agent_id,
            "task_id": task_id,
            "user": {
                "id": str(user_id) if user_id is not None else None,
                "channel": channel,
            },
            "request": _truncate_text(request, self.max_text_chars) if text_captured else "",
            "response": _truncate_text(response, self.max_text_chars) if text_captured else "",
            "status": status,
            "handoff_to": list(handoff_to or []),
            "actions": _compact_value(actions or [], max_chars=self.max_text_chars),
            "model_usage": _compact_value(model_usage or [], max_chars=self.max_text_chars),
            "metadata": _compact_value(metadata or {}, max_chars=self.max_text_chars),
            "privacy": {
                "text_captured": text_captured,
                "max_text_chars": self.max_text_chars,
            },
        }

        try:
            self._append(event)
        except Exception as exc:
            logger.exception("Failed to write learning event")
            return {
                "recorded": False,
                "reason": "write_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        self._notify_subscribers(event)
        return {"recorded": True, "event_id": event["id"], "path": str(self.path)}

    def record_agent_result(
        self,
        task: AgentTask,
        result: AgentResult,
        *,
        event_type: str = "agent_result",
        metadata: dict[str, Any] | None = None,
        elapsed_ms: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        actions = [{"kind": "taken", **action.model_dump()} for action in result.actions_taken] + [
            {"kind": "approval_required", **action.model_dump()} for action in result.actions_requiring_approval
        ]
        user_id = task.user.id if task.user else None
        channel = task.user.channel if task.user and task.user.channel else task.source
        event_metadata: dict[str, Any] = {
            "confidence": result.confidence,
            "trace_id": trace_id_from_task(task),
            "task_context": task.context,
            "files": task.files,
            "logs": result.logs,
            "diagnostic_trace": _diagnostic_trace_summary(actions, handoff_to=result.handoff_to),
            **(metadata or {}),
        }
        if elapsed_ms:
            event_metadata["elapsed_ms"] = elapsed_ms
        return self.record_event(
            event_type=event_type,
            source=task.source,
            agent_id=result.agent_id,
            task_id=task.task_id,
            user_id=user_id,
            channel=channel,
            request=task.request,
            response=result.answer,
            status=result.status,
            handoff_to=result.handoff_to,
            actions=actions,
            model_usage=[usage.model_dump() for usage in result.model_usage],
            metadata=event_metadata,
        )

    def record_feedback(
        self,
        *,
        event_id: str,
        rating: int | None = None,
        corrected_answer: str = "",
        comment: str = "",
        tags: list[str] | None = None,
        user_id: str | int | None = None,
        channel: str = "manual",
    ) -> dict[str, Any]:
        target_event = self.get_event(event_id)
        target_metadata = target_event.get("metadata") if isinstance(target_event, dict) else {}
        trace_id = target_metadata.get("trace_id") if isinstance(target_metadata, dict) else ""
        return self.record_event(
            event_type="human_feedback",
            source="manual_feedback",
            agent_id="human_reviewer",
            task_id=event_id,
            user_id=user_id,
            channel=channel,
            request=comment,
            response=corrected_answer,
            status="reviewed",
            metadata={
                "target_event_id": event_id,
                "trace_id": trace_id or "",
                "rating": rating,
                "tags": tags or [],
            },
        )

    def latest(self, *, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0 or not self.path.exists():
            return []
        lines: deque[str] = deque(maxlen=limit)
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    lines.append(line)
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
        return events

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        if not event_id or not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("id") == event_id:
                    return event
        return None

    def feedback_for(self, event_id: str, *, limit: int = 10) -> list[dict[str, Any]]:
        if not event_id or limit <= 0 or not self.path.exists():
            return []
        matches: deque[dict[str, Any]] = deque(maxlen=limit)
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(event, dict)
                    and event.get("event_type") == "human_feedback"
                    and (event.get("metadata") or {}).get("target_event_id") == event_id
                ):
                    matches.append(event)
        return list(matches)

    def diagnostic_groups(self, *, limit: int = 100, detailed: bool = False) -> dict[str, Any]:
        events = self._latest_by_type("diagnostic_report", limit=limit)
        groups: dict[str, dict[str, Any]] = {}
        for event in events:
            for key in _diagnostic_group_keys(event):
                group = groups.setdefault(key, {"key": key, "count": 0, "event_ids": [], "examples": []})
                group["count"] += 1
                group["event_ids"].append(event.get("id"))
                if len(group["examples"]) < 3:
                    group["examples"].append(
                        {
                            "id": event.get("id"),
                            "target_event_id": (event.get("metadata") or {}).get("target_event_id"),
                            "status": event.get("status"),
                            "response": event.get("response"),
                        }
                    )

        ordered = sorted(groups.values(), key=lambda item: (-int(item["count"]), str(item["key"])))
        if detailed:
            for group in ordered:
                group["diagnosis"] = _diagnostic_group_diagnosis(str(group["key"]))
        return {
            "total_reports": len(events),
            "mode": "detailed" if detailed else "brief",
            "groups": ordered,
        }

    def _latest_by_type(self, event_type: str, *, limit: int) -> list[dict[str, Any]]:
        if limit <= 0 or not self.path.exists():
            return []
        lines: deque[str] = deque(maxlen=max(limit * 5, limit))
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    lines.append(line)
        matches: deque[dict[str, Any]] = deque(maxlen=limit)
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("event_type") == event_type:
                matches.append(event)
        return list(matches)

    def stats(self) -> dict[str, Any]:
        by_event_type: Counter[str] = Counter()
        by_source: Counter[str] = Counter()
        by_status: Counter[str] = Counter()
        total = 0
        invalid = 0
        last_event_at = None

        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        invalid += 1
                        continue
                    if not isinstance(event, dict):
                        invalid += 1
                        continue
                    total += 1
                    by_event_type[str(event.get("event_type") or "unknown")] += 1
                    by_source[str(event.get("source") or "unknown")] += 1
                    status = str(event.get("status") or "")
                    if status:
                        by_status[status] += 1
                    last_event_at = event.get("created_at") or last_event_at

        return {
            "enabled": self.enabled,
            "capture_text": self.capture_text,
            "path": str(self.path),
            "exists": self.path.exists(),
            "total_events": total,
            "invalid_lines": invalid,
            "by_event_type": dict(by_event_type),
            "by_source": dict(by_source),
            "by_status": dict(by_status),
            "last_event_at": last_event_at,
        }

    def _append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _compact_value(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, BaseModel):
        return _compact_value(value.model_dump(), max_chars=max_chars)
    if is_dataclass(value) and not isinstance(value, type):
        return _compact_value(asdict(value), max_chars=max_chars)
    if isinstance(value, dict):
        return {str(key): _compact_value(item, max_chars=max(1000, max_chars // 2)) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_compact_value(item, max_chars=max(1000, max_chars // 2)) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _truncate_text(value, max_chars) if isinstance(value, str) else value
    return _truncate_text(str(value), max_chars)


def _diagnostic_trace_summary(actions: list[dict[str, Any]], *, handoff_to: list[str]) -> dict[str, Any]:
    loaded_rules: list[dict[str, Any]] = []
    loaded_skills: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for action in actions:
        details = action.get("details") if isinstance(action.get("details"), dict) else {}
        loaded_rules.extend(_list_of_dicts(details.get("loaded_rules")))
        loaded_skills.extend(_list_of_dicts(details.get("loaded_skills")))
        for call in _list_of_dicts(details.get("tool_calls")):
            tool_calls.append({"action": action.get("name"), **call})
        status = str(action.get("status") or "")
        error = details.get("error")
        if status in {"error", "failed"} or error:
            errors.append(
                {
                    "action": action.get("name"),
                    "status": status,
                    "error": error,
                }
            )

    return {
        "called_agents": list(handoff_to),
        "loaded_rules": _dedupe_by_id_file(loaded_rules),
        "loaded_skills": _dedupe_by_id_file(loaded_skills),
        "tool_calls": tool_calls,
        "errors": errors,
    }


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _dedupe_by_id_file(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        key = (str(item.get("id") or ""), str(item.get("file") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _diagnostic_group_keys(event: dict[str, Any]) -> list[str]:
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    trace = metadata.get("diagnostic_trace") if isinstance(metadata.get("diagnostic_trace"), dict) else {}
    task_context = metadata.get("task_context") if isinstance(metadata.get("task_context"), dict) else {}
    target_event = task_context.get("target_event") if isinstance(task_context.get("target_event"), dict) else {}
    target_trace = (target_event.get("metadata") or {}).get("diagnostic_trace")
    target_trace = target_trace if isinstance(target_trace, dict) else {}
    feedback_events = task_context.get("feedback_events") if isinstance(task_context.get("feedback_events"), list) else []

    keys: list[str] = []
    for agent_id in _strings(target_trace.get("called_agents")):
        keys.append(f"target_agent:{agent_id}")
    for rule in _list_of_dicts(target_trace.get("loaded_rules")):
        if rule.get("id"):
            keys.append(f"loaded_rule:{rule['id']}")
    for skill in _list_of_dicts(target_trace.get("loaded_skills")):
        if skill.get("id"):
            keys.append(f"loaded_skill:{skill['id']}")
    for tool_call in _list_of_dicts(target_trace.get("tool_calls")):
        if tool_call.get("name"):
            keys.append(f"tool_call:{tool_call['name']}")
    for error in _list_of_dicts(target_trace.get("errors")):
        if error.get("action"):
            keys.append(f"error_action:{error['action']}")
    for feedback in feedback_events:
        if not isinstance(feedback, dict):
            continue
        for tag in _strings((feedback.get("metadata") or {}).get("tags")):
            keys.append(f"tag:{tag}")
    for agent_id in _strings(trace.get("called_agents")):
        keys.append(f"diagnostic_agent:{agent_id}")
    return sorted(set(keys)) or ["ungrouped"]


def _diagnostic_group_diagnosis(key: str) -> dict[str, str]:
    prefix, _, value = key.partition(":")
    if prefix == "loaded_skill":
        return {
            "problem": f"Повторяющиеся ошибки связаны со skill `{value}`.",
            "likely_reason": "Агент открывает этот skill, но правило может быть неполным, неточным или не покрывать реальный сценарий.",
            "fix_proposal": f"Проверить `skill_index.yaml` и текст skill `{value}`; добавить недостающие условия, алгоритм и regression test.",
        }
    if prefix == "loaded_rule":
        return {
            "problem": f"Повторяющиеся ошибки связаны с правилом `{value}`.",
            "likely_reason": "Правило подгружается, но может давать недостаточно точные указания для текущего типа запроса.",
            "fix_proposal": f"Уточнить главу правила `{value}` и добавить тест на сценарий, где оно должно сработать.",
        }
    if prefix == "target_agent":
        return {
            "problem": f"Ошибки часто происходят в контуре агента `{value}`.",
            "likely_reason": "Проблема может быть в routing, skill/rule агента, tool-вызове или финальной интерпретации результата.",
            "fix_proposal": f"Разобрать последние отчеты по `{value}` и проверить: выбранный skill, tool_calls, tool_results и финальный ответ.",
        }
    if prefix == "tool_call":
        return {
            "problem": f"Повторяющиеся ошибки связаны с tool `{value}`.",
            "likely_reason": "Tool может вызываться с неполными аргументами, неверным методом или без нужной предварительной проверки.",
            "fix_proposal": f"Проверить правила подготовки аргументов для `{value}` и добавить regression test на неправильный вызов.",
        }
    if prefix == "error_action":
        return {
            "problem": f"В отчетах повторяется ошибка на действии `{value}`.",
            "likely_reason": "На этом этапе возникает технический сбой, ошибка tool или некорректная обработка результата.",
            "fix_proposal": f"Проверить обработку action `{value}`, лог ошибки и добавить тест на этот failure-path.",
        }
    if prefix == "tag":
        return {
            "problem": f"Пользовательские feedback часто помечены тегом `{value}`.",
            "likely_reason": "Люди вручную указывают на повторяющуюся область проблемы.",
            "fix_proposal": f"Посмотреть примеры с тегом `{value}` и сделать общий патч для повторяющегося сценария.",
        }
    return {
        "problem": "Группа ошибок пока не имеет точной категории.",
        "likely_reason": "Недостаточно структурированных признаков в diagnostic_trace или feedback.",
        "fix_proposal": "Улучшить трассировку и добавить более точные tags/loaded_rules/loaded_skills/tool_calls.",
    }


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 24)] + "\n...[truncated]"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


LearningEventRecorder = EventStream
