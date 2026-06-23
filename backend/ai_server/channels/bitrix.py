from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from ai_server.attachments import AttachmentService, StoredAttachment
from ai_server.channels.ports import QualityControlHandlerPort, SearchWebhookHandlerPort
from ai_server.integrations.bitrix.bitrix_store import BitrixAgentStore
from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.dialog_state import (
    BitrixPendingActionService,
    DialogStateStore,
    PendingBitrixAction,
    make_dialog_key,
)
from ai_server.integrations.bitrix.events import MESSAGE_EVENTS, parse_incoming_message, payload_event_type
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.integrations.bitrix.portal_search import PortalSearchIndex
from ai_server.integrations.bitrix.ports import BitrixAgentStorePort
from ai_server.learning import LearningEventRecorder
from ai_server.models import ActionRecord, AgentManifest, AgentTask, UserContext
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.registry import load_agent_manifests
from ai_server.settings import Settings, get_settings
from ai_server.technical_footer import TechnicalFooterService, append_footer
from ai_server.transcription import TranscriptionResult, build_transcriber

logger = logging.getLogger(__name__)

_MARKETPLACE_PATH_RE = re.compile(r"(/marketplace/view/[A-Za-z0-9._-]+/?)")


@dataclass
class BitrixTaskContext:
    bitrix_event_type: str
    dialog_key: str
    dialog_id: str
    pending_action: dict[str, Any] | None
    dialog_history: list[dict[str, Any]]
    transcriptions: list[dict[str, Any]]
    attachment_errors: list[str]

    def to_context(self) -> dict[str, Any]:
        return {
            "bitrix_event_type": self.bitrix_event_type,
            "dialog_key": self.dialog_key,
            "dialog_id": self.dialog_id,
            "pending_action": self.pending_action,
            "dialog_history": self.dialog_history,
            "transcriptions": self.transcriptions,
            "attachment_errors": self.attachment_errors,
        }


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
        technical_footer: TechnicalFooterService | None = None,
        attachment_service: AttachmentService | None = None,
        transcriber: Any | None = None,
        learning_recorder: LearningEventRecorder | None = None,
        bitrix_store: BitrixAgentStorePort | None = None,
        search_webhook_handler: SearchWebhookHandlerPort | None = None,
        quality_control_handler: QualityControlHandlerPort | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._search_webhook_handler = search_webhook_handler
        self._quality_control_handler = quality_control_handler
        self._manifests = manifests or load_agent_manifests()
        self.bitrix = bitrix or BitrixClient(settings=self._settings)
        self.portal_search = portal_search or PortalSearchIndex()
        self.bitrix_oauth = bitrix_oauth
        self.search_webhook_status = search_webhook_status if search_webhook_status is not None else {}
        self.quality_control_status = quality_control_status if quality_control_status is not None else {}
        self.technical_footer = technical_footer or TechnicalFooterService(settings=self._settings)
        self.attachment_service = attachment_service or AttachmentService(self.bitrix)
        self.transcriber = transcriber or build_transcriber()
        self.learning_recorder = learning_recorder
        self._bitrix_store: BitrixAgentStorePort = bitrix_store or BitrixAgentStore()
        self.pending_actions = pending_actions or BitrixPendingActionService(
            store=DialogStateStore(self._settings.dialog_state_path),
            bitrix=self.bitrix,
            bitrix_oauth=bitrix_oauth,
            audit_log_path=self._settings.bitrix_write_audit_log_path,
            dry_run=self._settings.agent_dry_run,
            settings=self._settings,
        )
        self._orchestrator: InternalOrchestrator = orchestrator or InternalOrchestrator(self._manifests)

    async def process(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_type = payload_event_type(payload)
        if event_type not in MESSAGE_EVENTS:
            return await self._handle_background_events(payload, event_type)
        return await self._handle_message_event(payload, event_type)

    # ------------------------------------------------------------------
    # Background event handlers (search index, quality control)
    # ------------------------------------------------------------------

    async def _handle_background_events(self, payload: dict[str, Any], event_type: str) -> dict[str, Any]:
        search_result = await self._handle_search_webhook(payload)
        quality_result = await self._handle_quality_control_webhook(payload)
        return {
            "handled": bool(search_result.get("handled")) or bool(quality_result.get("handled")),
            "event": event_type,
            "search_index": search_result,
            "quality_control": quality_result,
        }

    async def _handle_search_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._search_webhook_handler is not None:
            return await self._search_webhook_handler.handle(payload, status=self.search_webhook_status)
        return {"handled": False, "reason": "handler_not_configured"}

    async def _handle_quality_control_webhook(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._quality_control_handler is not None:
            return await self._quality_control_handler.handle(payload, status=self.quality_control_status)
        return {"handled": False, "reason": "handler_not_configured"}

    # ------------------------------------------------------------------
    # Chat message handler
    # ------------------------------------------------------------------

    async def _handle_message_event(self, payload: dict[str, Any], event_type: str) -> dict[str, Any]:
        incoming = parse_incoming_message(payload)
        attachment_context = await self._prepare_attachments(incoming)
        if attachment_context["transcription_text"]:
            incoming = incoming.model_copy(
                update={"text": _merge_text_and_transcription(incoming.text, attachment_context["transcription_text"])}
            )
        dialog_key = make_dialog_key(
            chat_id=incoming.chat_id,
            dialog_id=incoming.dialog_id,
            user_id=incoming.user_id,
        )
        dialog_state = self.pending_actions.store.load(dialog_key)
        task = self._build_task(incoming, attachment_context, dialog_state, dialog_key)
        result = await self._orchestrator.handle(task)
        return await self._finalize_message(incoming, task, result, dialog_key, event_type, attachment_context)

    def _build_task(
        self, incoming: Any, attachment_context: dict[str, Any], dialog_state: Any, dialog_key: str
    ) -> AgentTask:
        pending_action = dialog_state.pending_action
        # In PG mode each specialist loads its own dialog history from its schema.
        # In SQLite mode use the shared turns from DialogStateStore.
        shared_turns = [] if self._settings.database_url else dialog_state.turns[-8:]
        ctx = BitrixTaskContext(
            bitrix_event_type=incoming.event_type,
            dialog_key=dialog_key,
            dialog_id=incoming.dialog_id or "",
            pending_action=asdict(pending_action) if pending_action is not None else None,
            dialog_history=shared_turns,
            transcriptions=attachment_context["transcriptions"],
            attachment_errors=attachment_context["errors"],
        )
        return AgentTask(
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
            context=ctx.to_context(),
        )

    async def _finalize_message(
        self,
        incoming: Any,
        task: AgentTask,
        result: Any,
        dialog_key: str,
        event_type: str,
        attachment_context: dict[str, Any],
    ) -> dict[str, Any]:
        # In PG mode each specialist appends its own turn in BaseSpecialist.handle().
        if incoming.text and result.answer and not self._settings.database_url:
            self.pending_actions.store.append_turn(dialog_key, incoming.text, result.answer)
        pending_action = self._save_first_pending_action(
            dialog_key, result.actions_requiring_approval, user_id=incoming.user_id
        )
        reply_text = append_footer(
            result.answer,
            await self.technical_footer.build_for_agent_result(
                result, user_id=incoming.user_id, channel="bitrix24_chat"
            ),
        )
        reply_sent, send_error = await self._send_reply(
            incoming.dialog_id,
            reply_text,
            bot_id=incoming.bot_id or self._settings.bitrix_bot_id,
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
