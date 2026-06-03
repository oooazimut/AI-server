from __future__ import annotations

from typing import Any
from uuid import uuid4

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.events import MESSAGE_EVENTS, parse_incoming_message, payload_event_type
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.models import AgentTask, UserContext
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import load_agent_manifests
from ai_server.settings import get_settings
from ai_server.workers.bitrix.search_webhook_indexer import (
    prepare_search_webhook_job,
    process_search_webhook_job,
)


class BitrixWebhookProcessor:
    def __init__(
        self,
        *,
        bitrix: BitrixClient | None = None,
        portal_search: PortalSearchIndex | None = None,
        search_webhook_status: dict[str, Any] | None = None,
        orchestrator: InternalOrchestrator | None = None,
    ) -> None:
        self.bitrix = bitrix or BitrixClient()
        self.portal_search = portal_search or PortalSearchIndex()
        self.search_webhook_status = search_webhook_status if search_webhook_status is not None else {}
        self.orchestrator = orchestrator

    async def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_type = payload_event_type(payload)
        if event_type not in MESSAGE_EVENTS:
            search_job, search_result = prepare_search_webhook_job(payload)
            if search_job:
                search_result = await process_search_webhook_job(
                    self.bitrix,
                    self.portal_search,
                    search_job,
                    status=self.search_webhook_status,
                )
            return {
                "handled": bool(search_result.get("handled")),
                "event": event_type,
                "search_index": search_result,
            }

        incoming = parse_incoming_message(payload)
        orchestrator = self.orchestrator or InternalOrchestrator(load_agent_manifests())
        result = await orchestrator.handle(
            AgentTask(
                task_id=str(uuid4()),
                source="bitrix24_chat",
                user=UserContext(
                    id=str(incoming.user_id) if incoming.user_id is not None else None,
                    channel="bitrix24_chat",
                    raw={
                        "dialog_id": incoming.dialog_id,
                        "chat_id": incoming.chat_id,
                        "message_id": incoming.message_id,
                        "bot_id": incoming.bot_id,
                    },
                ),
                request=incoming.text,
                files=[file.model_dump() for file in incoming.files],
                context={"bitrix_event_type": incoming.event_type},
            )
        )

        reply_sent = False
        send_error = None
        settings = get_settings()
        if result.answer and incoming.dialog_id and not settings.agent_dry_run:
            try:
                await self.bitrix.send_bot_message(
                    incoming.dialog_id,
                    result.answer,
                    bot_id=incoming.bot_id or settings.bitrix_bot_id,
                )
                reply_sent = True
            except Exception as exc:
                send_error = f"{type(exc).__name__}: {exc}"

        return {
            "handled": True,
            "event": event_type,
            "agent_result_status": result.status,
            "reply_sent": reply_sent,
            "send_error": send_error,
            "handoff_to": result.handoff_to,
            "actions": [action.model_dump() for action in result.actions_taken],
            "approval_actions": [action.model_dump() for action in result.actions_requiring_approval],
        }
