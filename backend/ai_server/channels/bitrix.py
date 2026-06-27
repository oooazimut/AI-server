from __future__ import annotations

import re
from typing import Any

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.settings import Settings, get_settings

_MARKETPLACE_PATH_RE = re.compile(r"(/marketplace/view/[A-Za-z0-9._-]+/?)")


class BitrixChatChannel:
    """Bitrix24 chat channel — pure outbound transport (ChannelPort).

    Receives messages from the orchestrator via send() and delivers them to
    the Bitrix24 bot API. All webhook routing is handled by WebhookEventWorker.
    """

    channel_id = "bitrix24"

    def __init__(
        self,
        *,
        bitrix: BitrixClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.bitrix = bitrix or BitrixClient(settings=self._settings)

    async def send(self, recipient_id: str, body: str) -> None:
        """ChannelPort: deliver an outbound message to a Bitrix24 dialog."""
        if not body or not recipient_id or self._settings.agent_dry_run:
            return
        await self.bitrix.send_bot_message(
            recipient_id,
            body,
            bot_id=self._settings.bitrix_bot_id,
            keyboard=_keyboard_for_message(body),
        )


# Backward-compatibility alias (used by routes that reference BitrixWebhookProcessor)
BitrixWebhookProcessor = BitrixChatChannel


def _keyboard_for_message(message: str) -> dict[str, Any] | None:
    match = _MARKETPLACE_PATH_RE.search(message or "")
    if not match:
        return None
    return {"BUTTONS": [{"TEXT": "Открыть AI-помощник", "LINK": match.group(1)}]}
