from __future__ import annotations

import logging
import re

from ai_server.integrations.postgres.diagnost_agent import PostgresDiagnostStore
from ai_server.models import AgentTask

logger = logging.getLogger(__name__)

# Words/symbols that count as a positive or negative rating.
# Mapped to rating score 1-10.
_RATING_MAP: dict[str, int] = {
    "👍": 10,
    "10": 10,
    "10/10": 10,
    "отлично": 10,
    "супер": 10,
    "хорошо": 8,
    "неплохо": 7,
    "нормально": 5,
    "средне": 5,
    "плохо": 3,
    "слабо": 2,
    "👎": 1,
    "ужасно": 1,
    "нет": 1,
}


def _parse_numeric_rating(value: str) -> int | None:
    if value.isdigit():
        rating = int(value)
        return rating if 1 <= rating <= 10 else None
    if value.endswith("/10"):
        prefix = value[:-3].strip()
        if prefix.isdigit():
            rating = int(prefix)
            return rating if 1 <= rating <= 10 else None
    return None


def _detect_rating(text: str) -> tuple[int | None, str]:
    """Return (rating, raw_text) or (None, raw_text) if the text is not a rating."""
    raw_text = text.strip()
    normalized = raw_text.lower()
    rating = _RATING_MAP.get(normalized)
    if rating is None:
        rating = _parse_numeric_rating(normalized)
    if rating is None:
        match = re.match(r"^(10|[1-9])(?:\s*/\s*10)?(?:\b|\s|[.,:;!?)\]-])", normalized)
        if match:
            rating = int(match.group(1))
    return rating, raw_text


class FeedbackReceiverAdapter:
    """Implements FeedbackReceiverPort: classify feedback messages and save to diagnost store.

    Called from webhook_event_queue BEFORE routing to orchestrator.
    Returns True  → intercept; False → pass through normally.
    """

    def __init__(self, store: PostgresDiagnostStore) -> None:
        self._store = store

    async def handle(self, task: AgentTask) -> bool:
        user_id = str(task.user.id) if task.user and task.user.id is not None else ""
        if not user_id:
            return False

        rating, raw_text = _detect_rating(task.request or "")
        if rating is None:
            return False

        pending = await self._store.get_pending_feedback_for_user(user_id)
        if pending is None:
            return False

        try:
            await self._store.save_feedback(
                pending["event_id"],
                user_id,
                rating=rating,
                raw_text=raw_text,
                dialog_key=str(pending.get("dialog_key") or task.context.get("dialog_key") or ""),
            )
            await self._store.mark_pending_received(int(pending["id"]))
        except Exception:
            logger.exception("FeedbackReceiverAdapter: failed to save feedback for user %s", user_id)
            return False

        return True
