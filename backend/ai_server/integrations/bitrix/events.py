from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


MESSAGE_EVENTS = {
    "ONIMBOTV2MESSAGEADD",
    "ONIMBOTMESSAGEADD",
}


class IncomingFile(BaseModel):
    id: int | None = None
    name: str | None = None
    type: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class IncomingMessage(BaseModel):
    event_type: str
    bot_id: int | None = None
    dialog_id: str = ""
    chat_id: int | None = None
    message_id: int | None = None
    user_id: int | None = None
    text: str = ""
    files: list[IncomingFile] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


def payload_event_type(payload: dict[str, Any]) -> str:
    return str(payload.get("event") or payload.get("EVENT") or payload.get("type") or "").upper()


def parse_incoming_message(payload: dict[str, Any]) -> IncomingMessage:
    event_type = payload_event_type(payload)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if event_type.startswith("ONIMBOTV2"):
        return _parse_v2_message(event_type, data, payload)
    return _parse_legacy_message(event_type, data, payload)


def _parse_v2_message(
    event_type: str,
    data: dict[str, Any],
    payload: dict[str, Any],
) -> IncomingMessage:
    bot = _dict(data.get("bot"))
    chat = _dict(data.get("chat"))
    message = _dict(data.get("message"))
    user = _dict(data.get("user"))
    return IncomingMessage(
        event_type=event_type,
        bot_id=_int(bot.get("id")),
        dialog_id=str(chat.get("dialogId") or ""),
        chat_id=_int(chat.get("id") or chat.get("chatId")),
        message_id=_int(message.get("id")),
        user_id=_int(user.get("id") or message.get("authorId")),
        text=str(message.get("text") or ""),
        files=_extract_files(message, data),
        raw=payload,
    )


def _parse_legacy_message(
    event_type: str,
    data: dict[str, Any],
    payload: dict[str, Any],
) -> IncomingMessage:
    params = _dict(data.get("PARAMS"))
    user = _dict(data.get("USER"))
    bot_payload = _dict(data.get("BOT"))
    bot = next(iter(bot_payload.values()), {}) if bot_payload else {}
    return IncomingMessage(
        event_type=event_type,
        bot_id=_int(bot.get("BOT_ID")),
        dialog_id=str(params.get("DIALOG_ID") or params.get("TO_USER_ID") or ""),
        chat_id=_int(params.get("CHAT_ID") or params.get("TO_CHAT_ID")),
        message_id=_int(params.get("MESSAGE_ID")),
        user_id=_int(user.get("ID") or params.get("FROM_USER_ID")),
        text=str(params.get("MESSAGE") or ""),
        files=_extract_files(params, data),
        raw=payload,
    )


def _extract_files(*containers: dict[str, Any]) -> list[IncomingFile]:
    files: list[IncomingFile] = []
    for container in containers:
        params = container.get("params") or container.get("PARAMS")
        if isinstance(params, dict):
            files.extend(_extract_file_ids_from_params(params))
        for key in (
            "files",
            "file",
            "FILES",
            "FILE",
            "attachments",
            "attachment",
            "ATTACHMENTS",
            "ATTACHMENT",
        ):
            value = container.get(key)
            if value:
                files.extend(_extract_files_from_value(value, source=key))
    return _dedupe_files(files)


def _extract_file_ids_from_params(params: dict[str, Any]) -> list[IncomingFile]:
    files: list[IncomingFile] = []
    for key in (
        "FILE_ID",
        "fileId",
        "fileIds",
        "FILES",
        "files",
        "FILE",
        "file",
        "ATTACHMENTS",
        "attachments",
    ):
        value = params.get(key)
        if value:
            files.extend(_extract_files_from_value(value, source=f"params.{key}"))
    return files


def _extract_files_from_value(value: Any, *, source: str) -> list[IncomingFile]:
    file_id = _int(value)
    if file_id is not None:
        return [IncomingFile(id=file_id, raw={"source": source})]
    if isinstance(value, list):
        files: list[IncomingFile] = []
        for index, item in enumerate(value):
            files.extend(_extract_files_from_value(item, source=f"{source}[{index}]"))
        return files
    if not isinstance(value, dict):
        return []

    direct_id = _int(
        value.get("id")
        or value.get("ID")
        or value.get("fileId")
        or value.get("FILE_ID")
        or value.get("file_id")
    )
    if direct_id is not None:
        return [
            IncomingFile(
                id=direct_id,
                name=value.get("name") or value.get("NAME") or value.get("fileName") or value.get("FILE_NAME"),
                type=(
                    value.get("type")
                    or value.get("TYPE")
                    or value.get("extension")
                    or value.get("EXTENSION")
                    or value.get("mediaType")
                    or value.get("contentType")
                ),
                raw={**value, "source": source},
            )
        ]

    files: list[IncomingFile] = []
    for key, item in value.items():
        next_source = f"{source}.{key}"
        if _int(key) is not None and not isinstance(item, dict):
            item_id = _int(item)
            if item_id is not None:
                files.append(IncomingFile(id=item_id, raw={"source": next_source}))
            continue
        files.extend(_extract_files_from_value(item, source=next_source))
    return files


def _dedupe_files(files: list[IncomingFile]) -> list[IncomingFile]:
    result: list[IncomingFile] = []
    seen: set[int] = set()
    for file in files:
        if file.id is not None:
            if file.id in seen:
                continue
            seen.add(file.id)
        result.append(file)
    return result


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None

