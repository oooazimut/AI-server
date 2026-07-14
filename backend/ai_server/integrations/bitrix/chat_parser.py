"""Parse Bitrix webhook payloads into AgentTask objects.

Separates parsing/attachment concerns from routing (WebhookEventWorker)
and transport (BitrixChatChannel). All logic that was in
BitrixChatChannel._handle_message_event / _build_task / _prepare_attachments
lives here so the channel can be a pure transport adapter.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from ai_server.attachments import AttachmentService, StoredAttachment
from ai_server.integrations.bitrix.events import parse_incoming_message
from ai_server.models import AgentTask, UserContext
from ai_server.transcription import TranscriptionResult

_TASK_QUALITY_WEBHOOK_EVENTS = {"ONTASKUPDATE"}


def make_dialog_key(*, chat_id: int | None = None, dialog_id: str = "", user_id: int | None = None) -> str:
    resolved = user_id or 0
    if chat_id:
        return f"chat:{chat_id}:user:{resolved}"
    if dialog_id:
        return f"dialog:{dialog_id}:user:{resolved}"
    return f"user:{resolved}"


async def build_agent_task_from_bitrix_chat(
    payload: dict[str, Any],
    *,
    attachment_service: AttachmentService,
    transcriber: Any,
    settings: Any,
) -> AgentTask:
    """Build an AgentTask from a Bitrix chat webhook payload.

    Handles attachment downloads and voice transcription before packaging into
    an AgentTask so that downstream agents receive a fully-resolved task.
    """
    incoming = parse_incoming_message(payload)
    attachment_context = await _prepare_attachments(
        incoming, attachment_service=attachment_service, transcriber=transcriber
    )
    if attachment_context["transcription_text"]:
        incoming = incoming.model_copy(
            update={"text": _merge_text_and_transcription(incoming.text, attachment_context["transcription_text"])}
        )
    base_dialog_key = make_dialog_key(
        chat_id=incoming.chat_id,
        dialog_id=incoming.dialog_id,
        user_id=incoming.user_id,
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
        context={
            "bitrix_event_type": incoming.event_type,
            "dialog_key": base_dialog_key,
            "base_dialog_key": base_dialog_key,
            "dialog_id": incoming.dialog_id or "",
            "channel_id": "bitrix24",
            "recipient_id": incoming.dialog_id or "",
            "dialog_history": [],
            "transcriptions": attachment_context["transcriptions"],
            "attachment_errors": attachment_context["errors"],
        },
    )


def build_agent_task_from_task_event(payload: dict[str, Any]) -> AgentTask:
    """Build a quality-control AgentTask from a Bitrix task webhook payload."""
    event_type = str(payload.get("event") or payload.get("EVENT") or "").upper()
    task_id = _extract_task_id_from_event(payload)
    return AgentTask(
        task_id=uuid4().hex,
        source="bitrix_task_webhook",
        request="quality_control",
        context={
            "bitrix_event_type": event_type,
            "task_id": task_id,
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _prepare_attachments(
    incoming: Any,
    *,
    attachment_service: AttachmentService,
    transcriber: Any,
) -> dict[str, Any]:
    if not incoming.files:
        return {"stored_files": [], "transcriptions": [], "transcription_text": "", "errors": []}

    errors: list[str] = []
    stored_files: list[StoredAttachment] = []
    transcriptions: list[TranscriptionResult] = []
    try:
        stored_files = await attachment_service.download_message_files(incoming)
    except Exception as exc:
        errors.append(f"download:{type(exc).__name__}: {exc}")
        return {"stored_files": [], "transcriptions": [], "transcription_text": "", "errors": errors}

    for attachment in stored_files:
        if not attachment.is_audio:
            continue
        try:
            transcriptions.append(await transcriber.transcribe(attachment))
        except Exception as exc:
            errors.append(f"transcribe:{attachment.file_id}:{type(exc).__name__}: {exc}")

    transcription_text = "\n\n".join(item.text for item in transcriptions if item.text)
    return {
        "stored_files": [item.model_dump() for item in stored_files],
        "transcriptions": [item.model_dump() for item in transcriptions],
        "transcription_text": transcription_text,
        "errors": errors,
    }


def _extract_task_id_from_event(payload: dict[str, Any]) -> int | None:
    data = _dict_value(_first_ci(payload, "data", "DATA"))
    fields_after = _dict_value(_first_ci(data, "FIELDS_AFTER", "fieldsAfter"))
    fields_before = _dict_value(_first_ci(data, "FIELDS_BEFORE", "fieldsBefore"))
    for container in (fields_after, fields_before, data, payload):
        task_id = _to_int(_first_ci(container, "ID", "id", "TASK_ID", "taskId", "task_id"))
        if task_id is not None:
            return task_id
    return None


def _merge_text_and_transcription(text: str, transcription: str) -> str:
    cleaned_text = text.strip()
    cleaned_transcription = transcription.strip()
    if cleaned_text and cleaned_transcription:
        return f"{cleaned_text}\n\nРасшифровка голосового сообщения:\n{cleaned_transcription}"
    return cleaned_transcription or cleaned_text


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_ci(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    lowered = {str(k).lower(): v for k, v in data.items()}
    for key in keys:
        v = lowered.get(key.lower())
        if v is not None:
            return v
    return None


def _to_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
