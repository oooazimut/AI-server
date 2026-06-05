from __future__ import annotations

from typing import Any
from uuid import uuid4

from ai_server.integrations.bitrix.dialog_state import (
    BitrixPendingActionService,
    DialogStateStore,
    PendingActionResult,
    PendingBitrixAction,
    make_dialog_key,
)
from ai_server.agents.bitrix_llm import BitrixAgentLLM
from ai_server.agents.pending_control_llm import PendingControlLLM, PendingControlLLMService
from ai_server.orchestrators.internal_llm import InternalOrchestratorLLM
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.events import MESSAGE_EVENTS, parse_incoming_message, payload_event_type
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.models import ActionRecord, AgentTask, UserContext
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import load_agent_manifests
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
from ai_server.technical_footer import TechnicalFooterService, append_footer
from ai_server.tools.bitrix import BitrixToolset
from ai_server.workers.bitrix.search_webhook_indexer import (
    prepare_search_webhook_job,
    process_search_webhook_job,
)
from ai_server.workers.bitrix.quality_control import handle_quality_control_webhook_event


class BitrixWebhookProcessor:
    def __init__(
        self,
        *,
        bitrix: BitrixClient | None = None,
        portal_search: PortalSearchIndex | None = None,
        bitrix_oauth: BitrixOAuthService | None = None,
        search_webhook_status: dict[str, Any] | None = None,
        quality_control_status: dict[str, Any] | None = None,
        orchestrator: InternalOrchestrator | None = None,
        pending_actions: BitrixPendingActionService | None = None,
        bitrix_tools: BitrixToolset | None = None,
        bitrix_retriever: HybridKnowledgeRetriever | None = None,
        bitrix_llm: BitrixAgentLLM | None = None,
        pending_control_llm: PendingControlLLM | None = None,
        orchestrator_llm: InternalOrchestratorLLM | None = None,
        technical_footer: TechnicalFooterService | None = None,
    ) -> None:
        settings = get_settings()
        self.bitrix = bitrix or BitrixClient()
        self.portal_search = portal_search or PortalSearchIndex()
        self.bitrix_oauth = bitrix_oauth
        self.search_webhook_status = search_webhook_status if search_webhook_status is not None else {}
        self.quality_control_status = quality_control_status if quality_control_status is not None else {}
        self.orchestrator = orchestrator
        self.bitrix_tools = bitrix_tools
        self.bitrix_retriever = bitrix_retriever
        self.bitrix_llm = bitrix_llm
        self.pending_control_llm = pending_control_llm or PendingControlLLMService()
        self.orchestrator_llm = orchestrator_llm
        self.technical_footer = technical_footer or TechnicalFooterService()
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
            quality_bitrix, quality_actor_error = await self._quality_control_bitrix_client()
            if quality_actor_error:
                quality_result = quality_actor_error
            else:
                quality_result = await handle_quality_control_webhook_event(
                    quality_bitrix,
                    payload=payload,
                    status=self.quality_control_status,
                )
            return {
                "handled": bool(search_result.get("handled")) or bool(quality_result.get("handled")),
                "event": event_type,
                "search_index": search_result,
                "quality_control": quality_result,
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
            footer = await self.technical_footer.build_for_pending_action(
                user_id=incoming.user_id,
                channel="bitrix24_chat",
                status=direct_result.status,
            )
            reply_sent, send_error = await self._send_reply(
                incoming.dialog_id,
                append_footer(direct_result.message, footer),
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

        orchestrator = self.orchestrator or InternalOrchestrator(
            load_agent_manifests(),
            bitrix_retriever=self.bitrix_retriever,
            bitrix_llm=self.bitrix_llm,
            orchestrator_llm=self.orchestrator_llm,
            bitrix_tools=self.bitrix_tools
            or BitrixToolset(
                client=self.bitrix,
                portal_search=self.portal_search,
                pending_actions=self.pending_actions,
                dialog_key=dialog_key,
                user_id=incoming.user_id,
            ),
        )
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
        reply_text = result.answer
        footer = await self.technical_footer.build_for_agent_result(
            result,
            user_id=incoming.user_id,
            channel="bitrix24_chat",
        )
        reply_text = append_footer(reply_text, footer)
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

        try:
            control_result = await self.pending_control_llm.classify(
                dialog_key=dialog_key,
                user_id=user_id,
                user_text=text,
                pending_action=pending,
            )
        except Exception as exc:
            return PendingActionResult(
                status="needs_clarification",
                message=(
                    "Не смог уверенно понять, подтверждаете вы ожидающее действие или отменяете его. "
                    f"Ожидающее действие: {pending.summary}. Подтверждаем или отменяем?"
                ),
                action=pending,
                data={"classification_error": f"{type(exc).__name__}: {exc}"},
            )

        decision = control_result.decision.decision
        if decision == "new_request":
            return None
        if decision == "cancel":
            return self.pending_actions.cancel(dialog_key)
        if decision != "confirm":
            return PendingActionResult(
                status="needs_clarification",
                message=(
                    control_result.decision.answer
                    or f"У меня есть ожидающее действие: {pending.summary}. Подтверждаем выполнение или отменяем?"
                ),
                action=pending,
                data={
                    "pending_control_decision": decision,
                    "pending_control_confidence": control_result.decision.confidence,
                    "pending_control_reasoning": control_result.decision.reasoning,
                },
            )

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

    async def _quality_control_bitrix_client(self) -> tuple[BitrixClient, dict[str, Any] | None]:
        settings = get_settings()
        if not settings.quality_control_webhook_enabled:
            return self.bitrix, None
        actor_user_id = settings.quality_control_actor_user_id
        if not actor_user_id:
            if settings.quality_control_dry_run:
                return self.bitrix, None
            error = {
                "handled": False,
                "reason": "quality_actor_not_configured",
                "message": (
                    "Для боевого фонового контроля качества нужно задать "
                    "QUALITY_CONTROL_ACTOR_USER_ID и авторизовать этого пользователя через Bitrix OAuth."
                ),
            }
            _record_quality_actor_error(self.quality_control_status, error)
            return self.bitrix, error
        if settings.quality_control_dry_run:
            return self.bitrix, None
        if not settings.bitrix_oauth_enabled or self.bitrix_oauth is None:
            error = {
                "handled": False,
                "reason": "quality_actor_oauth_disabled",
                "actor_user_id": actor_user_id,
            }
            _record_quality_actor_error(self.quality_control_status, error)
            return self.bitrix, error
        try:
            return await self.bitrix_oauth.client_for_user(actor_user_id), None
        except BitrixOAuthTokenMissing:
            error = {
                "handled": False,
                "reason": "quality_actor_oauth_missing",
                "actor_user_id": actor_user_id,
                "message": (
                    "Для фонового контроля качества нужно один раз авторизовать локальное "
                    "приложение под служебным пользователем AI-помощника."
                ),
            }
            _record_quality_actor_error(self.quality_control_status, error)
            return self.bitrix, error


def _pending_from_approval_action(
    action: ActionRecord,
    *,
    user_id: int | None,
) -> PendingBitrixAction | None:
    if action.name not in {"bitrix_api", "bitrix_task_closure"}:
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


def _record_quality_actor_error(status: dict[str, Any], error: dict[str, Any]) -> None:
    status["last_error"] = error.get("message") or error.get("reason")
    status["last_reason"] = error.get("reason")
    status["errors"] = int(status.get("errors") or 0) + 1
