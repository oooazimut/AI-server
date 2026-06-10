from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import unquote

import httpx
from pydantic import BaseModel

from ai_server.integrations.bitrix.client import BitrixClient
from ai_server.integrations.bitrix.events import IncomingFile, IncomingMessage
from ai_server.settings import Settings, get_settings

logger = logging.getLogger(__name__)


AUDIO_EXTENSIONS = {".aac", ".amr", ".flac", ".m4a", ".mp3", ".oga", ".ogg", ".opus", ".wav", ".webm"}


class StoredAttachment(BaseModel):
    file_id: int
    name: str
    content_type: str | None = None
    size: int
    path: str
    is_audio: bool = False


class AttachmentDownloadError(RuntimeError):
    pass


class AttachmentService:
    def __init__(
        self,
        bitrix: BitrixClient,
        settings: Settings | None = None,
        *,
        storage_dir: Path | None = None,
        max_bytes: int | None = None,
    ) -> None:
        _settings = settings or get_settings()
        self.bitrix = bitrix
        self.storage_dir = storage_dir or _settings.attachment_storage_dir
        self.max_bytes = max_bytes or _settings.attachment_max_bytes

    async def download_message_files(self, message: IncomingMessage) -> list[StoredAttachment]:
        stored = []
        for file in message.files:
            if file.id is None:
                continue
            stored.append(await self.download_file(file, message))
        return stored

    async def download_file(self, file: IncomingFile, message: IncomingMessage) -> StoredAttachment:
        if file.id is None:
            raise AttachmentDownloadError("Cannot download a file without file id")

        download_url = await self._resolve_download_url(file, message)
        size = 0
        target: Path | None = None
        content_type: str | None = None
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(120.0), follow_redirects=True, trust_env=False) as client,
            client.stream("GET", download_url) as response,
        ):
            response.raise_for_status()
            content_type = response.headers.get("content-type")
            filename = self._build_filename(file, message, response.headers.get("content-disposition"))
            target = self.storage_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.max_bytes:
                        output.close()
                        target.unlink(missing_ok=True)
                        raise AttachmentDownloadError(f"Attachment {file.id} exceeds {self.max_bytes} bytes")
                    output.write(chunk)

        if target is None:
            raise AttachmentDownloadError(f"Attachment {file.id} was not downloaded")

        return StoredAttachment(
            file_id=file.id,
            name=file.name or target.name,
            content_type=content_type,
            size=size,
            path=str(target),
            is_audio=self._is_audio(file, content_type, target.name),
        )

    def _build_filename(
        self,
        file: IncomingFile,
        message: IncomingMessage,
        content_disposition: str | None = None,
    ) -> str:
        name = self._safe_filename(
            file.name or self._filename_from_content_disposition(content_disposition) or f"file-{file.id}"
        )
        message_part = message.message_id or "message"
        return f"{message_part}-{file.id}-{name}"

    def _safe_filename(self, value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9а-яА-Я._-]+", "_", value).strip("._")
        return normalized or "attachment"

    def _filename_from_content_disposition(self, value: str | None) -> str | None:
        if not value:
            return None
        utf_match = re.search(r"filename\*=UTF-8''([^;]+)", value, flags=re.IGNORECASE)
        if utf_match:
            return unquote(utf_match.group(1).strip().strip('"'))
        match = re.search(r'filename="?([^";]+)"?', value, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _is_audio(self, file: IncomingFile, content_type: str | None, downloaded_name: str | None = None) -> bool:
        if content_type and content_type.startswith("audio/"):
            return True
        if file.type and str(file.type).lower() in {"audio", "voice"}:
            return True
        return Path(file.name or downloaded_name or "").suffix.lower() in AUDIO_EXTENSIONS

    async def _resolve_download_url(self, file: IncomingFile, message: IncomingMessage) -> str:
        if file.id is None:
            raise AttachmentDownloadError("Cannot download a file without file id")
        try:
            return await self.bitrix.get_bot_file_download_url(file.id, bot_id=message.bot_id)
        except Exception as bot_exc:
            if not message.dialog_id:
                raise
            logger.info("Bot file download failed, trying chat file download: file_id=%s", file.id)
            try:
                return await self.bitrix.get_chat_file_download_url(file.id, dialog_id=message.dialog_id)
            except Exception:
                logger.exception("Chat file download failed: file_id=%s dialog_id=%s", file.id, message.dialog_id)
                raise bot_exc from None
