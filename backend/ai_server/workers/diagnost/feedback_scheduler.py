from __future__ import annotations

import asyncio
import logging

from ai_server.integrations.postgres.diagnost_agent import PostgresDiagnostStore

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = "Оцените ответ: 👍 (хорошо) или 👎 (плохо), либо введите оценку от 1 до 5."
_DEFAULT_DELAY_SECONDS = 10
_DEFAULT_POLL_INTERVAL = 30


async def run_feedback_scheduler_worker(
    store: PostgresDiagnostStore,
    bitrix: object,
    *,
    prompt_text: str = _DEFAULT_PROMPT,
    delay_seconds: int = _DEFAULT_DELAY_SECONDS,
    poll_interval: int = _DEFAULT_POLL_INTERVAL,
) -> None:
    """Read unsent pending_feedback rows and send a rating prompt via Bitrix.

    bitrix must expose: async send_bot_message(dialog_id, message) — BitrixClient satisfies this.
    """
    logger.info("FeedbackScheduler: started (delay=%ds, poll=%ds)", delay_seconds, poll_interval)
    while True:
        try:
            items = await store.get_unsent_feedback_requests(delay_seconds=delay_seconds)
            for item in items:
                dialog_key = str(item.get("dialog_key") or "")
                if not dialog_key:
                    continue
                try:
                    await bitrix.send_bot_message(dialog_key, prompt_text)  # type: ignore[attr-defined]
                    await store.mark_pending_sent(int(item["id"]))
                    logger.debug("FeedbackScheduler: prompt sent for event %s", item.get("event_id"))
                except Exception:
                    logger.exception("FeedbackScheduler: failed to send prompt for event %s", item.get("event_id"))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("FeedbackScheduler: unexpected error in poll loop")

        await asyncio.sleep(poll_interval)
