from ai_server.settings import get_settings
from ai_server.transcription import (
    OpenAITranscriber,
    UnconfiguredTranscriber,
    YandexSpeechKitTranscriber,
    build_transcriber,
)


def _isolate_env(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    for var in ("STT_PROVIDER", "OPENAI_API_KEY", "YANDEX_API_KEY", "YANDEX_IAM_TOKEN", "YANDEX_FOLDER_ID"):
        monkeypatch.delenv(var, raising=False)


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
