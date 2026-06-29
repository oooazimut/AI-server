from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from ai_server.agent_store import SqliteStore
from ai_server.agents.ports import ChannelPort
from ai_server.diagnostics import run_diagnostic_via_orchestrator
from ai_server.learning import LearningEventRecorder
from ai_server.models import AgentResult, AgentTask, UserContext
from ai_server.settings import Settings, get_settings
from ai_server.tracing import TraceRecorder, new_span_id

FEEDBACK_PROMPT = "Оцените ответ по 10-балльной шкале. Если что-то не так, напишите пояснение."

_RATING_RE = re.compile(
    r"^\s*(?:оценка\s*)?(?P<rating>10|[1-9])(?:\s*(?:/|из)\s*10)?(?:[\s,.:;-]+(?P<comment>.*))?\s*$",
    re.IGNORECASE,
)
_BAD_COMMENT_MARKERS = (
    "не помог",
    "не наш",
    "не то",
    "невер",
    "ошиб",
    "плох",
    "нет ",
    "нету",
    "неправ",
    "не выполн",
)


@dataclass(frozen=True)
class ParsedFeedback:
    rating: int
    comment: str
    outcome: str


class FeedbackPromptStore(SqliteStore):
    def __init__(self, path: Path | str | None = None, *, settings: Settings | None = None) -> None:
        _settings = settings or get_settings()
        self.path = Path(path) if path is not None else _settings.dialog_state_path
        self.ensure_schema()

    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_feedback (
                    dialog_key TEXT PRIMARY KEY,
                    learning_event_id TEXT NOT NULL,
                    trace_id TEXT,
                    recipient_id TEXT,
                    user_id TEXT,
                    answer TEXT,
                    status TEXT NOT NULL DEFAULT 'awaiting_rating',
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                )
                """
            )
            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pending_feedback_expires_at
                    ON pending_feedback(expires_at)
                """
            )

    def save_pending(
        self,
        *,
        dialog_key: str,
        learning_event_id: str,
        trace_id: str = "",
        recipient_id: str = "",
        user_id: str = "",
        answer: str = "",
        ttl_hours: int = 24,
    ) -> None:
        if not dialog_key or not learning_event_id:
            return
        now = _now()
        expires_at = now + timedelta(hours=max(1, ttl_hours))
        with self._connection() as db:
            db.execute(
                """
                INSERT INTO pending_feedback (
                    dialog_key, learning_event_id, trace_id, recipient_id, user_id,
                    answer, status, created_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'awaiting_rating', ?, ?)
                ON CONFLICT(dialog_key) DO UPDATE SET
                    learning_event_id = excluded.learning_event_id,
                    trace_id = excluded.trace_id,
                    recipient_id = excluded.recipient_id,
                    user_id = excluded.user_id,
                    answer = excluded.answer,
                    status = 'awaiting_rating',
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    dialog_key,
                    learning_event_id,
                    trace_id,
                    recipient_id,
                    user_id,
                    answer,
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )

    def get_pending(self, dialog_key: str) -> dict[str, Any] | None:
        if not dialog_key:
            return None
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM pending_feedback WHERE dialog_key = ? AND status = 'awaiting_rating'",
                (dialog_key,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        expires_at = str(item.get("expires_at") or "")
        if expires_at:
            try:
                if datetime.fromisoformat(expires_at) < _now():
                    self.clear_pending(dialog_key)
                    return None
            except ValueError:
                return item
        return item

    def clear_pending(self, dialog_key: str) -> None:
        if not dialog_key:
            return
        with self._connection() as db:
            db.execute("DELETE FROM pending_feedback WHERE dialog_key = ?", (dialog_key,))


class FeedbackLoopService:
    def __init__(
        self,
        *,
        store: FeedbackPromptStore | None = None,
        learning_recorder: LearningEventRecorder,
        trace_recorder: TraceRecorder,
        manifests: list[Any],
        channels: dict[str, ChannelPort] | None = None,
        diagnostic_llm: Any = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store or FeedbackPromptStore(settings=self._settings)
        self._learning_recorder = learning_recorder
        self._trace_recorder = trace_recorder
        self._manifests = manifests
        self._channels = channels or {}
        self._diagnostic_llm = diagnostic_llm

    def append_prompt(self, message: str) -> str:
        if not message or FEEDBACK_PROMPT.casefold() in message.casefold():
            return message
        return f"{message}\n\n{FEEDBACK_PROMPT}"

    def remember_answer(self, task: AgentTask, result: AgentResult, learning_record: dict[str, Any] | None) -> None:
        if not result.answer or not learning_record or learning_record.get("recorded") is not True:
            return
        if task.context.get("channel_id") != "bitrix24":
            return
        self._store.save_pending(
            dialog_key=str(task.context.get("dialog_key") or ""),
            learning_event_id=str(learning_record.get("event_id") or ""),
            trace_id=str(task.context.get("trace_id") or ""),
            recipient_id=str(task.context.get("recipient_id") or ""),
            user_id=str(task.user.id or "") if task.user else "",
            answer=result.answer,
        )

    async def try_handle_feedback(self, task: AgentTask) -> dict[str, Any] | None:
        dialog_key = str(task.context.get("dialog_key") or "")
        pending = self._store.get_pending(dialog_key)
        if not pending:
            return None

        parsed = parse_feedback_text(task.request)
        if parsed is None:
            return None

        learning_event_id = str(pending.get("learning_event_id") or "")
        trace_id = str(pending.get("trace_id") or "")
        trace_events = self._trace_recorder.for_trace(trace_id, limit=500) if trace_id else []
        user_id = task.user.id if task.user else None
        feedback = self._learning_recorder.record_feedback(
            event_id=learning_event_id,
            rating=parsed.rating,
            rating_scale=10,
            outcome=parsed.outcome,
            comment=parsed.comment,
            tags=["bitrix_chat_feedback"],
            user_id=user_id,
            channel="bitrix24_chat",
            trace_events=trace_events,
        )
        incident = feedback.get("incident") if isinstance(feedback.get("incident"), dict) else {}
        diagnostic = None
        if incident.get("recorded") is True:
            diagnostic = await self._diagnose(
                target_event_id=learning_event_id,
                feedback_event_id=str(feedback.get("event_id") or ""),
                incident_event_id=str(incident.get("event_id") or ""),
                comment=parsed.comment,
            )

        self._trace_recorder.record(
            event_name="human_feedback_received",
            trace_id=trace_id,
            span_id=new_span_id(),
            agent_id="feedback_loop",
            task_id=task.task_id,
            status="recorded",
            payload={
                "learning_event_id": learning_event_id,
                "feedback_event_id": feedback.get("event_id"),
                "incident_event_id": incident.get("event_id"),
                "diagnostic_report_id": (diagnostic or {}).get("diagnostic_report_id"),
                "rating": parsed.rating,
                "rating_scale": 10,
                "outcome": parsed.outcome,
                "comment": parsed.comment,
            },
        )
        self._store.clear_pending(dialog_key)
        await self._send_feedback_ack(task, parsed=parsed, feedback=feedback, diagnostic=diagnostic)
        return {
            "handled": True,
            "routed_to": "feedback_loop",
            "learning_event_id": learning_event_id,
            "feedback_event_id": feedback.get("event_id"),
            "incident_event_id": incident.get("event_id"),
            "diagnostic_report_id": (diagnostic or {}).get("diagnostic_report_id"),
        }

    async def _diagnose(
        self,
        *,
        target_event_id: str,
        feedback_event_id: str,
        incident_event_id: str,
        comment: str,
    ) -> dict[str, Any] | None:
        target_event = self._learning_recorder.get_event(target_event_id)
        if target_event is None:
            return None
        feedback_events = [
            event
            for event in self._learning_recorder.feedback_for(target_event_id, limit=10)
            if event.get("id") == feedback_event_id
        ]
        incident_events = [
            event
            for event in self._learning_recorder.incidents_for(target_event_id, limit=10)
            if event.get("id") == incident_event_id
        ]
        target_metadata = target_event.get("metadata") if isinstance(target_event.get("metadata"), dict) else {}
        trace_id = str(target_metadata.get("trace_id") or "")
        trace_events = self._trace_recorder.for_trace(trace_id, limit=500) if trace_id else []
        task = AgentTask(
            task_id=str(uuid4()),
            source="learning_diagnose",
            user=UserContext(channel="diagnostics"),
            request=comment or f"Разбери learning event {target_event_id}",
            context={
                "event_id": target_event_id,
                "target_event": target_event,
                "trace_id": trace_id,
                "trace_events": trace_events,
                "feedback_events": feedback_events,
                "incident_events": incident_events,
                "feedback": feedback_events[-1] if feedback_events else {},
                "rating": (feedback_events[-1].get("metadata") or {}).get("rating") if feedback_events else None,
                "comment": comment,
            },
        )
        result = await run_diagnostic_via_orchestrator(
            manifests=self._manifests,
            task=task,
            diagnostic_llm=self._diagnostic_llm,
            trace_recorder=self._trace_recorder,
        )
        diagnostic_record = self._learning_recorder.record_agent_result(
            task,
            result,
            event_type="diagnostic_report",
            metadata={
                "target_event_id": target_event_id,
                "feedback_event_ids": [event.get("id") for event in feedback_events],
                "feedback_event_id": feedback_event_id,
                "incident_event_ids": [event.get("id") for event in incident_events],
            },
        )
        return {
            "status": result.status,
            "answer": result.answer,
            "diagnostic_report_id": diagnostic_record.get("event_id"),
            "diagnostic_record": diagnostic_record,
        }

    async def _send_feedback_ack(
        self,
        task: AgentTask,
        *,
        parsed: ParsedFeedback,
        feedback: dict[str, Any],
        diagnostic: dict[str, Any] | None,
    ) -> None:
        channel = self._channels.get(str(task.context.get("channel_id") or ""))
        recipient_id = str(task.context.get("recipient_id") or "")
        if channel is None or not recipient_id:
            return
        lines = [f"Спасибо, оценка {parsed.rating}/10 сохранена."]
        incident = feedback.get("incident") if isinstance(feedback.get("incident"), dict) else {}
        if incident.get("recorded") is True:
            lines.append("Я создал incident для разбора.")
        if diagnostic and diagnostic.get("diagnostic_report_id"):
            lines.append("Diagnostic Agent подготовил внутренний diagnostic_report для команды.")
        await channel.send(recipient_id, "\n\n".join(lines))


def parse_feedback_text(text: str) -> ParsedFeedback | None:
    match = _RATING_RE.match(text or "")
    if not match:
        return None
    rating = int(match.group("rating"))
    comment = str(match.group("comment") or "").strip()
    outcome = "not_completed" if rating <= 5 or _has_bad_comment(comment) else "completed"
    return ParsedFeedback(rating=rating, comment=comment, outcome=outcome)


def _has_bad_comment(comment: str) -> bool:
    folded = f" {comment.casefold()} "
    return any(marker in folded for marker in _BAD_COMMENT_MARKERS)


def _now() -> datetime:
    return datetime.now(UTC)
