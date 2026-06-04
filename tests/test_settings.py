from ai_server.settings import get_settings


def test_llm_defaults_to_deepseek_flash(monkeypatch):
    monkeypatch.delenv("AI_SERVER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AI_SERVER_LLM_MODEL", raising=False)
    monkeypatch.delenv("AI_SERVER_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    settings = get_settings()

    assert settings.llm_provider == "deepseek"
    assert settings.llm_model == "deepseek-v4-flash"
    assert settings.llm_configured is False


def test_llm_settings_can_be_overridden(monkeypatch):
    monkeypatch.setenv("AI_SERVER_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AI_SERVER_LLM_MODEL", "custom-model")
    monkeypatch.setenv("AI_SERVER_LLM_API_KEY", "secret")
    monkeypatch.setenv("AI_SERVER_LLM_BASE_URL", "https://example.local/v1")
    monkeypatch.setenv("AI_SERVER_LLM_MAX_TOKENS", "1234")

    settings = get_settings()

    assert settings.llm_provider == "openai_compatible"
    assert settings.llm_model == "custom-model"
    assert settings.llm_base_url == "https://example.local/v1"
    assert settings.llm_max_tokens == 1234
    assert settings.llm_configured is True
