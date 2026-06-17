from __future__ import annotations

import logging
import re
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from ai_server.agent_scheduler import AgentScheduler
from ai_server.agents.bitrix24 import Bitrix24Specialist
from ai_server.agents.bitrix_llm import BitrixAgentLLM
from ai_server.agents.bitrix_store import BitrixAgentStore
from ai_server.attachments import AttachmentService, StoredAttachment
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.dialog_state import (
    BitrixPendingActionService,
    DialogStateStore,
    PendingBitrixAction,
    make_dialog_key,
)
from ai_server.integrations.bitrix.events import MESSAGE_EVENTS, parse_incoming_message, payload_event_type
from ai_server.integrations.bitrix.oauth import BitrixOAuthService, BitrixOAuthTokenMissing
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.learning import LearningEventRecorder
from ai_server.models import ActionRecord, AgentManifest, AgentTask, UserContext
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.orchestrators.internal_llm import InternalOrchestratorLLM
from ai_server.registry import load_agent_manifests
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import Settings, get_settings
from ai_server.specialists import build_specialist_registry
from ai_server.technical_footer import TechnicalFooterService, append_footer
from ai_server.tools.bitrix import BitrixToolset
from ai_server.tools.document_access import DocumentToolset
from ai_server.tools.vehicle_usage import VehicleUsageToolset
from ai_server.transcription import TranscriptionResult, build_transcriber
from ai_server.workers.bitrix.quality_control import handle_quality_control_webhook_event
from ai_server.workers.bitrix.search_webhook_indexer import (
    prepare_search_webhook_job,
    process_search_webhook_job,
)

logger = logging.getLogger(__name__)

_MARKETPLACE_PATH_RE = re.compile(r"(/marketplace/view/[A-Za-z0-9._-]+/?)")


class BitrixWebhookProcessor:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        manifests: list[AgentManifest] | None = None,
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
        orchestrator_llm: InternalOrchestratorLLM | None = None,
        technical_footer: TechnicalFooterService | None = None,
        attachment_service: AttachmentService | None = None,
        transcriber: Any | None = None,
        learning_recorder: LearningEventRecorder | None = None,
        scheduler: AgentScheduler | None = None,
        bitrix_store: BitrixAgentStore | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._manifests = manifests or load_agent_manifests()
        self.bitrix = bitrix or BitrixClient()
        self.portal_search = portal_search or PortalSearchIndex()
        self.bitrix_oauth = bitrix_oauth
        self.scheduler = scheduler
        self.search_webhook_status = search_webhook_status if search_webhook_status is not None else {}
        self.quality_control_status = quality_control_status if quality_control_status is not None else {}
        self.orchestrator = orchestrator
        self.bitrix_tools = bitrix_tools
        self.bitrix_retriever = bitrix_retriever
        self.bitrix_llm = bitrix_llm
        self.orchestrator_llm = orchestrator_llm
        self.technical_footer = technical_footer or TechnicalFooterService()
        self.attachment_service = attachment_service or AttachmentService(self.bitrix)
        self.transcriber = transcriber or build_transcriber()
        self.learning_recorder = learning_recorder
        self._bitrix_store = bitrix_store or BitrixAgentStore()
        self.pending_actions = pending_actions or BitrixPendingActionService(
            store=DialogStateStore(self._settings.dialog_state_path),
            bitrix=self.bitrix,
            bitrix_oauth=bitrix_oauth,
            audit_log_path=self._settings.bitrix_write_audit_log_path,
            dry_run=self._settings.agent_dry_run,
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
                    specialist=self._build_quality_control_specialist(quality_bitrix),
                )
            return {
                "handled": bool(search_result.get("handled")) or bool(quality_result.get("handled")),
                "event": event_type,
                "search_index": search_result,
                "quality_control": quality_result,
            }

        incoming = parse_incoming_message(payload)
        attachment_context = await self._prepare_attachments(incoming)
        if attachment_context["transcription_text"]:
            incoming = incoming.model_copy(
                update={"text": _merge_text_and_transcription(incoming.text, attachment_context["transcription_text"])}
            )
        settings = self._settings
        dialog_key = make_dialog_key(
            chat_id=incoming.chat_id,
            dialog_id=incoming.dialog_id,
            user_id=incoming.user_id,
        )
        dialog_state = self.pending_actions.store.load(dialog_key)
        pending_action = dialog_state.pending_action
        recent_turns = dialog_state.turns[-8:]

        manifests = self._manifests
        orchestrator = self.orchestrator or InternalOrchestrator(
            manifests,
            specialists=build_specialist_registry(
                manifests,
                audience="employee",
                scheduler=self.scheduler,
                bitrix_retriever=self.bitrix_retriever,
                bitrix_llm=self.bitrix_llm,
                bitrix_store=self._bitrix_store,
                bitrix_tools=self.bitrix_tools
                or BitrixToolset(
                    client=self.bitrix,
                    portal_search=self.portal_search,
                    pending_actions=self.pending_actions,
                    dialog_key=dialog_key,
                    user_id=incoming.user_id,
                ),
                document_tools=DocumentToolset(
                    client=self.bitrix,
                    portal_search=self.portal_search,
                    user_id=incoming.user_id,
                ),
                vehicle_usage_tools=VehicleUsageToolset(
                    client=self.bitrix,
                    user_id=incoming.user_id,
                    dialog_id=incoming.dialog_id,
                ),
            ),
            orchestrator_llm=self.orchestrator_llm,
            scheduler=self.scheduler,
        )
        task = AgentTask(
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
            files=[
                *[file.model_dump() for file in incoming.files],
                *attachment_context["stored_files"],
            ],
            context={
                "bitrix_event_type": incoming.event_type,
                "transcriptions": attachment_context["transcriptions"],
                "attachment_errors": attachment_context["errors"],
                "dialog_key": dialog_key,
                "pending_action": asdict(pending_action) if pending_action is not None else None,
                "dialog_history": recent_turns,
            },
        )
        result = await orchestrator.handle(task)

        if incoming.text and result.answer:
            self.pending_actions.append_turn(dialog_key, incoming.text, result.answer)

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
        learning_event = self._record_agent_result(
            task,
            result,
            metadata={
                "dialog_key": dialog_key,
                "pending_action_saved": pending_action is not None,
                "reply_sent": reply_sent,
                "send_error": send_error,
            },
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
            "transcriptions": attachment_context["transcriptions"],
            "attachment_errors": attachment_context["errors"],
            "learning_event": learning_event,
        }

    async def _prepare_attachments(self, incoming) -> dict[str, Any]:
        if not incoming.files:
            return {"stored_files": [], "transcriptions": [], "transcription_text": "", "errors": []}

        errors: list[str] = []
        stored_files: list[StoredAttachment] = []
        transcriptions: list[TranscriptionResult] = []
        try:
            stored_files = await self.attachment_service.download_message_files(incoming)
        except Exception as exc:
            errors.append(f"download:{type(exc).__name__}: {exc}")
            return {"stored_files": [], "transcriptions": [], "transcription_text": "", "errors": errors}

        for attachment in stored_files:
            if not attachment.is_audio:
                continue
            try:
                transcriptions.append(await self.transcriber.transcribe(attachment))
            except Exception as exc:
                errors.append(f"transcribe:{attachment.file_id}:{type(exc).__name__}: {exc}")

        transcription_text = "\n\n".join(item.text for item in transcriptions if item.text)
        return {
            "stored_files": [item.model_dump() for item in stored_files],
            "transcriptions": [item.model_dump() for item in transcriptions],
            "transcription_text": transcription_text,
            "errors": errors,
        }

    async def _send_reply(
        self,
        dialog_id: str,
        message: str,
        *,
        bot_id: int | None = None,
    ) -> tuple[bool, str | None]:
        settings = self._settings
        if not message or not dialog_id or settings.agent_dry_run:
            return False, None
        try:
            await self.bitrix.send_bot_message(
                dialog_id,
                message,
                bot_id=bot_id or settings.bitrix_bot_id,
                keyboard=_keyboard_for_message(message),
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

    def _record_agent_result(
        self,
        task: AgentTask,
        result: Any,
        *,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.learning_recorder is None:
            return None
        try:
            return self.learning_recorder.record_agent_result(task, result, metadata=metadata)
        except Exception:
            logger.exception("Failed to record Bitrix learning event")
            return {"recorded": False, "reason": "unexpected_error"}

    def _build_quality_control_specialist(self, actor_bitrix: BitrixClient) -> Bitrix24Specialist | None:
        bitrix24_manifest = next((m for m in self._manifests if m.id == "bitrix24"), None)
        if bitrix24_manifest is None:
            return None
        deliver_fn = _make_quality_deliver_fn(self.bitrix, self._settings)
        return Bitrix24Specialist(
            bitrix24_manifest,
            tools=BitrixToolset(
                client=self.bitrix,
                actor_client=actor_bitrix,
                auto_execute=True,
            ),
            retriever=self.bitrix_retriever,
            llm=self.bitrix_llm,
            scheduler=self.scheduler,
            bitrix_store=self._bitrix_store,
            deliver_fn=deliver_fn,
            settings=self._settings,
        )

    async def _quality_control_bitrix_client(self) -> tuple[BitrixClient, dict[str, Any] | None]:
        settings = self._settings
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
        specialist_id=str(details.get("specialist_id") or "bitrix24"),
    )


def _make_quality_deliver_fn(bitrix: BitrixClient, settings: Settings):
    async def _deliver(user_id_str: str, message: str) -> None:
        uid = int(user_id_str) if user_id_str.lstrip("-").isdigit() else None
        if uid is not None:
            await bitrix.notify_user(user_id=uid, message=message, tag="task_proposal")
        else:
            await bitrix.send_bot_message(user_id_str, message, bot_id=settings.bitrix_bot_id)

    return _deliver


def _record_quality_actor_error(status: dict[str, Any], error: dict[str, Any]) -> None:
    status["last_error"] = error.get("message") or error.get("reason")
    status["last_reason"] = error.get("reason")
    status["errors"] = int(status.get("errors") or 0) + 1


def _keyboard_for_message(message: str) -> dict[str, Any] | None:
    match = _MARKETPLACE_PATH_RE.search(message or "")
    if not match:
        return None
    return {"BUTTONS": [{"TEXT": "Открыть AI-помощник", "LINK": match.group(1)}]}


def _merge_text_and_transcription(text: str, transcription: str) -> str:
    cleaned_text = text.strip()
    cleaned_transcription = transcription.strip()
    if cleaned_text and cleaned_transcription:
        return f"{cleaned_text}\n\nРасшифровка голосового сообщения:\n{cleaned_transcription}"
    return cleaned_transcription or cleaned_text
