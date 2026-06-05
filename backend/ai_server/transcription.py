from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ai_server.attachments import StoredAttachment
from ai_server.integrations.yandex_auth import YandexAuthError, yandex_auth_header
from ai_server.settings import get_settings


class TranscriptionResult(BaseModel):
    text: str
    model: str
    attachment: StoredAttachment
    raw: dict[str, Any] = Field(default_factory=dict)


class TranscriptionNotConfigured(RuntimeError):
    pass


class TranscriptionError(RuntimeError):
    pass


class YandexSpeechKitTranscriber:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        language: str | None = None,
        max_bytes: int | None = None,
        ffmpeg_path: str | None = None,
    ) -> None:
        settings = get_settings()
        self.base_url = (base_url or settings.yandex_speechkit_base_url).rstrip("/")
        self.language = language or settings.yandex_speechkit_lang
        self.max_bytes = max_bytes or settings.yandex_speechkit_max_bytes
        self.ffmpeg_path = ffmpeg_path or settings.ffmpeg_path

    @property
    def configured(self) -> bool:
        return get_settings().transcription_configured

    async def transcribe(self, attachment: StoredAttachment) -> TranscriptionResult:
        if not self.configured:
            raise TranscriptionNotConfigured(
                "YANDEX_API_KEY or YANDEX_IAM_TOKEN/YANDEX_FOLDER_ID is not configured"
            )

        source = Path(attachment.path)
        if not source.exists():
            raise TranscriptionError(f"Attachment does not exist: {source}")

        prepared_path, content_type = await self._prepare_audio(source)
        if prepared_path.stat().st_size > self.max_bytes:
            raise TranscriptionError(f"Attachment exceeds SpeechKit sync limit of {self.max_bytes} bytes")

        text, payload = await self._recognize_file(prepared_path, content_type)
        return TranscriptionResult(text=text, model="yandex_speechkit", attachment=attachment, raw=payload)

    async def _recognize_file(self, path: Path, content_type: str) -> tuple[str, dict[str, Any]]:
        settings = get_settings()
        params = {"lang": self.language}
        if not settings.yandex_api_key:
            params["folderId"] = settings.yandex_folder_id

        try:
            headers = yandex_auth_header()
        except YandexAuthError as exc:
            raise TranscriptionNotConfigured(str(exc)) from exc

        headers["Content-Type"] = content_type
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0), trust_env=False) as client:
            response = await client.post(
                f"{self.base_url}/speech/v1/stt:recognize",
                headers=headers,
                params=params,
                content=path.read_bytes(),
            )

        if response.status_code >= 400:
            raise TranscriptionError(f"Yandex SpeechKit failed: {response.status_code} {response.text}")

        payload = response.json()
        text = str(payload.get("result") or "").strip()
        if not text:
            raise TranscriptionError(f"Yandex SpeechKit returned empty text: {payload}")
        return text, payload

    async def _prepare_audio(self, source: Path) -> tuple[Path, str]:
        suffix = source.suffix.lower()
        if suffix in {".ogg", ".opus"}:
            return source, "audio/ogg"

        settings = get_settings()
        if not settings.yandex_speechkit_convert_to_ogg:
            raise TranscriptionError(f"Yandex SpeechKit does not accept {suffix or 'this'} audio directly")

        target = source.with_suffix(".ogg")
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            return target, "audio/ogg"
        if not shutil.which(self.ffmpeg_path):
            raise TranscriptionError(
                "ffmpeg is required to convert Bitrix voice messages to OggOpus for Yandex SpeechKit. "
                "Set FFMPEG_PATH or install ffmpeg."
            )

        process = await asyncio.create_subprocess_exec(
            self.ffmpeg_path,
            "-y",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "libopus",
            "-b:a",
            "32k",
            str(target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            raise TranscriptionError(f"ffmpeg conversion failed: {stderr.decode(errors='ignore')}")
        return target, "audio/ogg"


def build_transcriber() -> YandexSpeechKitTranscriber:
    settings = get_settings()
    if settings.stt_provider == "yandex_speechkit":
        return YandexSpeechKitTranscriber()
    raise TranscriptionNotConfigured(f"Unknown STT_PROVIDER: {settings.stt_provider}")
