from __future__ import annotations

from typing import Any
from uuid import uuid4

from ai_server.integrations.bitrix.dialog_state import (
    BitrixPendingActionService,
    DialogStateStore,
    PendingActionResult,
    PendingBitrixAction,
    is_cancel_text,
    is_confirm_text,
    make_dialog_key,
)
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.events import MESSAGE_EVENTS, parse_incoming_message, payload_event_type
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.models import ActionRecord, AgentTask, UserContext
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
        bitrix_oauth: BitrixOAuthService | None = None,
        search_webhook_status: dict[str, Any] | None = None,
        orchestrator: InternalOrchestrator | None = None,
        pending_actions: BitrixPendingActionService | None = None,
    ) -> None:
        settings = get_settings()
        self.bitrix = bitrix or BitrixClient()
        self.portal_search = portal_search or PortalSearchIndex()
        self.search_webhook_status = search_webhook_status if search_webhook_status is not None else {}
        self.orchestrator = orchestrator
        self.pending_actions = pending_actions or BitrixPendingActionService(
            store=DialogStateStore(settings.dialog_state_path),
            bitrix=self.bitrix,
            bitrix_oauth=bitrix_oauth,
            audit_log_path=settings.bitrix_write_audit_log_path,
        )

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
        settings = get_settings()
        dialog_key = make_dialog_key(
            chat_id=incoming.chat_id,
            dialog_id=incoming.dialog_id,
            user_id=incoming.user_id,
        )
        direct_result = await self._maybe_handle_pending_control(dialog_key, incoming.text, user_id=incoming.user_id)
        if direct_result:
            reply_sent, send_error = await self._send_reply(
                incoming.dialog_id,
                direct_result.message,
                bot_id=incoming.bot_id or settings.bitrix_bot_id,
            )
            return {
                "handled": True,
                "event": event_type,
                "agent_result_status": direct_result.status,
                "reply_sent": reply_sent,
                "send_error": send_error,
                "handoff_to": ["bitrix24"],
                "actions": [_pending_result_action(direct_result)],
                "approval_actions": [],
                "dialog_key": dialog_key,
            }

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

        pending_action = self._save_first_pending_action(
            dialog_key,
            result.actions_requiring_approval,
            user_id=incoming.user_id,
        )
        reply_text = _with_pending_confirmation_hint(result.answer, pending_action)
        reply_sent, send_error = await self._send_reply(
            incoming.dialog_id,
            reply_text,
            bot_id=incoming.bot_id or settings.bitrix_bot_id,
        )

        return {
            "handled": True,
            "event": event_type,
            "agent_result_status": result.status,
            "reply_sent": reply_sent,
            "send_error": send_error,
            "handoff_to": result.handoff_to,
            "actions": [action.model_dump() for action in result.actions_taken],
            "approval_actions": [action.model_dump() for action in result.actions_requiring_approval],
            "pending_action_saved": pending_action is not None,
            "dialog_key": dialog_key,
        }

    async def _maybe_handle_pending_control(
        self,
        dialog_key: str,
        text: str,
        *,
        user_id: int | None,
    ) -> PendingActionResult | None:
        pending = self.pending_actions.pending_for(dialog_key)
        if not pending:
            return None
        if is_cancel_text(text):
            return self.pending_actions.cancel(dialog_key)
        if not is_confirm_text(text):
            return None

        settings = get_settings()
        if settings.agent_dry_run:
            return PendingActionResult(
                status="dry_run",
                message=(
                    "AGENT_DRY_RUN включён: действие не выполнено. "
                    "Ожидающее действие оставлено без изменений."
                ),
                action=pending,
            )
        return await self.pending_actions.confirm(dialog_key, user_id=user_id)

    async def _send_reply(
        self,
        dialog_id: str,
        message: str,
        *,
        bot_id: int | None = None,
    ) -> tuple[bool, str | None]:
        settings = get_settings()
        if not message or not dialog_id or settings.agent_dry_run:
            return False, None
        try:
            await self.bitrix.send_bot_message(
                dialog_id,
                message,
                bot_id=bot_id or settings.bitrix_bot_id,
            )
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def _save_first_pending_action(
        self,
        dialog_key: str,
        approval_actions: list[ActionRecord],
        *,
        user_id: int | None,
    ) -> PendingBitrixAction | None:
        for action in approval_actions:
            pending = _pending_from_approval_action(action, user_id=user_id)
            if pending:
                self.pending_actions.save_pending(dialog_key, pending)
                return pending
        return None


def _pending_from_approval_action(
    action: ActionRecord,
    *,
    user_id: int | None,
) -> PendingBitrixAction | None:
    if action.name != "bitrix_api":
        return None
    details = action.details
    method = str(details.get("method") or "").strip()
    raw_params = details.get("params")
    if not method or not isinstance(raw_params, dict):
        return None
    return PendingBitrixAction(
        method=method,
        params=raw_params,
        summary=str(details.get("summary") or method),
        created_by=user_id,
    )


def _with_pending_confirmation_hint(answer: str, action: PendingBitrixAction | None) -> str:
    if not action:
        return answer
    hint = f"Подготовил действие: {action.summary}. Для выполнения напишите «да», для отмены - «отмена»."
    return f"{answer}\n\n{hint}" if answer else hint


def _pending_result_action(result: PendingActionResult) -> dict[str, Any]:
    details: dict[str, Any] = {"message": result.message, **result.data}
    if result.action:
        details.update(
            {
                "method": result.action.method,
                "params": result.action.params,
                "summary": result.action.summary,
            }
        )
    return ActionRecord(name="bitrix_pending_action", status=result.status, details=details).model_dump()
