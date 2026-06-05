from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from ai_server.models import AgentResult, AgentTask
from ai_server.settings import get_settings


logger = logging.getLogger(__name__)


class LearningEventRecorder:
    """Append-only local journal for future evals, reviews, and fine-tuning datasets."""

    schema_version = "learning_event.v1"

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        enabled: bool | None = None,
        capture_text: bool | None = None,
        max_text_chars: int | None = None,
    ) -> None:
        settings = get_settings()
        self.path = Path(path) if path is not None else settings.learning_events_path
        self.enabled = settings.learning_events_enabled if enabled is None else enabled
        self.capture_text = settings.learning_events_capture_text if capture_text is None else capture_text
        self.max_text_chars = max_text_chars or settings.learning_events_max_text_chars

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

        return {"recorded": True, "event_id": event["id"], "path": str(self.path)}

    def record_agent_result(
        self,
        task: AgentTask,
        result: AgentResult,
        *,
        event_type: str = "agent_result",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        actions = [
            {"kind": "taken", **action.model_dump()}
            for action in result.actions_taken
        ] + [
            {"kind": "approval_required", **action.model_dump()}
            for action in result.actions_requiring_approval
        ]
        user_id = task.user.id if task.user else None
        channel = task.user.channel if task.user and task.user.channel else task.source
        event_metadata = {
            "confidence": result.confidence,
            "task_context": task.context,
            "files": task.files,
            "logs": result.logs,
            **(metadata or {}),
        }
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

    def record_pending_result(
        self,
        *,
        dialog_key: str,
        user_id: str | int | None,
        request_text: str,
        result: Any,
        source: str = "bitrix24_chat",
        channel: str = "bitrix24_chat",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        action = getattr(result, "action", None)
        data = getattr(result, "data", {}) if result is not None else {}
        action_record = {
            "kind": "pending_action",
            "name": "bitrix_pending_action",
            "status": getattr(result, "status", ""),
            "details": {
                "message": getattr(result, "message", ""),
                "data": data,
                "pending": _pending_action_payload(action),
            },
        }
        return self.record_event(
            event_type="pending_action_result",
            source=source,
            agent_id="bitrix24_pending_control",
            task_id=dialog_key,
            user_id=user_id,
            channel=channel,
            request=request_text,
            response=getattr(result, "message", ""),
            status=getattr(result, "status", ""),
            handoff_to=["bitrix24"],
            actions=[action_record],
            model_usage=data.get("model_usage", []) if isinstance(data, dict) else [],
            metadata={"dialog_key": dialog_key, **(metadata or {})},
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


def _pending_action_payload(action: Any) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "method": getattr(action, "method", ""),
        "params": getattr(action, "params", {}),
        "summary": getattr(action, "summary", ""),
        "created_by": getattr(action, "created_by", None),
        "created_at": getattr(action, "created_at", None),
    }


def _compact_value(value: Any, *, max_chars: int) -> Any:
    if isinstance(value, BaseModel):
        return _compact_value(value.model_dump(), max_chars=max_chars)
    if is_dataclass(value) and not isinstance(value, type):
        return _compact_value(asdict(value), max_chars=max_chars)
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, max_chars=max(1000, max_chars // 2))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_compact_value(item, max_chars=max(1000, max_chars // 2)) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _truncate_text(value, max_chars) if isinstance(value, str) else value
    return _truncate_text(str(value), max_chars)


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 24)] + "\n...[truncated]"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
