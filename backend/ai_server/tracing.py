from __future__ import annotations

import json
import logging
from collections import Counter, deque
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from ai_server.models import AgentTask
from ai_server.settings import Settings, get_settings

logger = logging.getLogger(__name__)

TRACE_ID_CONTEXT_KEY = "trace_id"
SPAN_ID_CONTEXT_KEY = "span_id"
PARENT_SPAN_ID_CONTEXT_KEY = "parent_span_id"


class TraceRecorder:
    """Append-only trace journal for reconstructing agent execution chains."""

    schema_version = "trace_event.v1"

    def __init__(
        self,
        settings: Settings | None = None,
        path: Path | str | None = None,
        *,
        enabled: bool | None = None,
        capture_payload: bool | None = None,
        max_payload_chars: int | None = None,
    ) -> None:
        _settings = settings or get_settings()
        self.path = Path(path) if path is not None else _settings.trace_events_path
        self.enabled = _settings.trace_events_enabled if enabled is None else enabled
        self.capture_payload = _settings.trace_events_capture_payload if capture_payload is None else capture_payload
        self.max_payload_chars = max_payload_chars or _settings.trace_events_max_payload_chars

    def ensure_task_context(
        self,
        task: AgentTask,
        *,
        parent_span_id: str = "",
    ) -> tuple[str, str]:
        trace_id = str(task.context.get(TRACE_ID_CONTEXT_KEY) or new_trace_id())
        span_id = str(task.context.get(SPAN_ID_CONTEXT_KEY) or new_span_id())
        task.context[TRACE_ID_CONTEXT_KEY] = trace_id
        task.context[SPAN_ID_CONTEXT_KEY] = span_id
        if parent_span_id and not task.context.get(PARENT_SPAN_ID_CONTEXT_KEY):
            task.context[PARENT_SPAN_ID_CONTEXT_KEY] = parent_span_id
        return trace_id, span_id

    def child_context(
        self,
        task: AgentTask,
        *,
        span_id: str | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        trace_id, parent_span_id = self.ensure_task_context(task)
        child_span_id = span_id or new_span_id()
        return trace_id, child_span_id, {
            **task.context,
            TRACE_ID_CONTEXT_KEY: trace_id,
            SPAN_ID_CONTEXT_KEY: child_span_id,
            PARENT_SPAN_ID_CONTEXT_KEY: parent_span_id,
        }

    def record(
        self,
        *,
        event_name: str,
        trace_id: str,
        span_id: str = "",
        parent_span_id: str = "",
        agent_id: str = "",
        task_id: str = "",
        status: str = "",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"recorded": False, "reason": "disabled"}

        event = {
            "schema_version": self.schema_version,
            "id": str(uuid4()),
            "created_at": _now_iso(),
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "event_name": event_name,
            "agent_id": agent_id,
            "task_id": task_id,
            "status": status,
            "payload": _compact_value(payload or {}, max_chars=self.max_payload_chars)
            if self.capture_payload
            else {},
            "privacy": {
                "payload_captured": bool(self.capture_payload),
                "max_payload_chars": self.max_payload_chars,
            },
        }
        try:
            self._append(event)
        except Exception as exc:
            logger.exception("Failed to write trace event")
            return {
                "recorded": False,
                "reason": "write_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"recorded": True, "event_id": event["id"], "trace_id": trace_id, "path": str(self.path)}

    def latest(self, *, limit: int = 50) -> list[dict[str, Any]]:
        if limit <= 0 or not self.path.exists():
            return []
        lines: deque[str] = deque(maxlen=limit)
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    lines.append(line)
        return _parse_json_lines(lines)

    def for_trace(self, trace_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        if not trace_id or limit <= 0 or not self.path.exists():
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
                if isinstance(event, dict) and event.get("trace_id") == trace_id:
                    matches.append(event)
        return list(matches)

    def stats(self) -> dict[str, Any]:
        by_event_name: Counter[str] = Counter()
        by_agent_id: Counter[str] = Counter()
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
                    by_event_name[str(event.get("event_name") or "unknown")] += 1
                    by_agent_id[str(event.get("agent_id") or "unknown")] += 1
                    last_event_at = event.get("created_at") or last_event_at

        return {
            "enabled": self.enabled,
            "capture_payload": self.capture_payload,
            "path": str(self.path),
            "exists": self.path.exists(),
            "total_events": total,
            "invalid_lines": invalid,
            "by_event_name": dict(by_event_name),
            "by_agent_id": dict(by_agent_id),
            "last_event_at": last_event_at,
        }

    def _append(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def new_trace_id() -> str:
    return str(uuid4())


def new_span_id() -> str:
    return str(uuid4())


def trace_id_from_task(task: AgentTask) -> str:
    return str(task.context.get(TRACE_ID_CONTEXT_KEY) or "")


def span_id_from_task(task: AgentTask) -> str:
    return str(task.context.get(SPAN_ID_CONTEXT_KEY) or "")


def parent_span_id_from_task(task: AgentTask) -> str:
    return str(task.context.get(PARENT_SPAN_ID_CONTEXT_KEY) or "")


def _parse_json_lines(lines: deque[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


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
    if isinstance(value, str):
        return _truncate_text(value, max_chars)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _truncate_text(str(value), max_chars)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 24)] + "\n...[truncated]"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
