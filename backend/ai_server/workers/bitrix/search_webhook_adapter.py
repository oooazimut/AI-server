from __future__ import annotations

from typing import Any

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.settings import Settings
from ai_server.workers.bitrix.search_webhook_indexer import prepare_search_webhook_job, process_search_webhook_job


class SearchWebhookHandlerAdapter:
    """Adapter: implements SearchWebhookHandlerPort using worker functions."""

    def __init__(self, *, bitrix: BitrixClient, index: PortalSearchIndex, settings: Settings) -> None:
        self._bitrix = bitrix
        self._index = index
        self._settings = settings

    async def handle(self, payload: dict[str, Any], *, status: dict[str, Any]) -> dict[str, Any]:
        job, result = prepare_search_webhook_job(payload, settings=self._settings)
        if job:
            result = await process_search_webhook_job(
                self._bitrix, self._index, job, status=status, settings=self._settings
            )
        return result
