import asyncio
from pathlib import Path

import pytest

from ai_server.attachments import StoredAttachment
from ai_server.settings import get_settings
from ai_server.transcription import (
    OpenAITranscriber,
    TranscriptionError,
    UnconfiguredTranscriber,
    YandexSpeechKitTranscriber,
    build_transcriber,
)


class _FakeFfmpegProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode

    async def communicate(self):
        return b"", b""


def _isolate_env(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    for var in (
        "STT_PROVIDER",
        "OPENAI_API_KEY",
        "YANDEX_API_KEY",
        "YANDEX_IAM_TOKEN",
        "YANDEX_FOLDER_ID",
        "YANDEX_SPEECHKIT_CHUNK_SECONDS",
        "YANDEX_SPEECHKIT_MAX_CHUNKS",
    ):
        monkeypatch.delenv(var, raising=False)


def _attachment(path: Path) -> StoredAttachment:
    return StoredAttachment(
        file_id=1,
        name=path.name,
        content_type="audio/mp4",
        size=path.stat().st_size,
        path=str(path),
        is_audio=True,
    )


def test_build_transcriber_openai(monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("STT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    transcriber = build_transcriber()

    assert isinstance(transcriber, OpenAITranscriber)
    assert transcriber.configured
    assert transcriber.model == "gpt-4o-transcribe"
    assert get_settings().transcription_configured


def test_build_transcriber_openai_without_key_not_configured(monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("STT_PROVIDER", "openai")

    transcriber = build_transcriber()

    assert isinstance(transcriber, OpenAITranscriber)
    assert not transcriber.configured
    assert not get_settings().transcription_configured


def test_build_transcriber_defaults_to_yandex(monkeypatch):
    _isolate_env(monkeypatch)

    transcriber = build_transcriber()

    assert isinstance(transcriber, YandexSpeechKitTranscriber)


def test_build_transcriber_unknown_provider(monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("STT_PROVIDER", "bogus")

    transcriber = build_transcriber()

    assert isinstance(transcriber, UnconfiguredTranscriber)


def test_yandex_transcriber_splits_audio_and_combines_text(monkeypatch, tmp_path):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("YANDEX_SPEECHKIT_CHUNK_SECONDS", "25")
    monkeypatch.setenv("YANDEX_SPEECHKIT_MAX_CHUNKS", "12")

    source = tmp_path / "voice.m4a"
    source.write_bytes(b"audio")
    captured_args = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured_args["args"] = args
        pattern = str(args[-1])
        Path(pattern.replace("%03d", "000")).write_bytes(b"first")
        Path(pattern.replace("%03d", "001")).write_bytes(b"second")
        return _FakeFfmpegProcess()

    monkeypatch.setattr("ai_server.transcription.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("ai_server.transcription.asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    transcriber = YandexSpeechKitTranscriber()
    recognized = []

    async def fake_recognize_file(path, content_type):
        recognized.append((path.name, content_type))
        return f"text-{len(recognized)}", {"result": f"text-{len(recognized)}"}

    transcriber._recognize_file = fake_recognize_file

    result = asyncio.run(transcriber.transcribe(_attachment(source)))

    assert result.text == "text-1 text-2"
    assert [item[1] for item in recognized] == ["audio/ogg", "audio/ogg"]
    assert "-segment_time" in captured_args["args"]
    assert "25" in captured_args["args"]
    assert result.raw == {"chunks": [{"result": "text-1"}, {"result": "text-2"}]}


def test_yandex_transcriber_rejects_too_many_chunks(monkeypatch, tmp_path):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("YANDEX_API_KEY", "test-key")
    monkeypatch.setenv("YANDEX_SPEECHKIT_CHUNK_SECONDS", "25")
    monkeypatch.setenv("YANDEX_SPEECHKIT_MAX_CHUNKS", "2")

    source = tmp_path / "voice.m4a"
    source.write_bytes(b"audio")

    async def fake_create_subprocess_exec(*args, **kwargs):
        pattern = str(args[-1])
        for index in range(3):
            Path(pattern.replace("%03d", f"{index:03d}")).write_bytes(b"chunk")
        return _FakeFfmpegProcess()

    monkeypatch.setattr("ai_server.transcription.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr("ai_server.transcription.asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    transcriber = YandexSpeechKitTranscriber()

    with pytest.raises(TranscriptionError, match="maximum is about 50 seconds"):
        asyncio.run(transcriber.transcribe(_attachment(source)))
