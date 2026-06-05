from ai_server.settings import get_settings


def test_settings_loads_layered_env_files(monkeypatch, tmp_path):
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    first.write_text(
        "AI_SERVER_LLM_PROVIDER=from-file\n"
        "AI_SERVER_LLM_MODEL=first-model\n"
        "AI_SERVER_LLM_API_KEY=file-secret\n",
        encoding="utf-8",
    )
    second.write_text("AI_SERVER_LLM_MODEL=second-model\n", encoding="utf-8")
    monkeypatch.delenv("AI_SERVER_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("AI_SERVER_LLM_MODEL", raising=False)
    monkeypatch.delenv("AI_SERVER_LLM_API_KEY", raising=False)
    monkeypatch.setenv("AI_SERVER_ENV_FILE", f"{first},{second}")

    settings = get_settings()

    assert settings.llm_provider == "from-file"
    assert settings.llm_model == "second-model"
    assert settings.llm_configured is True


def test_llm_defaults_to_deepseek_flash(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
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
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
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


def test_bitrix_oauth_bot_settings_can_be_loaded(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    oauth_db = tmp_path / "bitrix_oauth.sqlite"
    monkeypatch.setenv("BITRIX_BOT_AUTH_MODE", "oauth")
    monkeypatch.setenv("BITRIX_BOT_OAUTH_USER_ID", "9")
    monkeypatch.setenv("BITRIX_OAUTH_DB_PATH", str(oauth_db))

    settings = get_settings()

    assert settings.bitrix_bot_uses_oauth is True
    assert settings.bitrix_bot_oauth_user_id == 9
    assert settings.bitrix_oauth_db_path == oauth_db


def test_bitrix_oauth_urls_are_resolved(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://ai.example.com")
    monkeypatch.setenv("BITRIX_DOMAIN", "portal.bitrix24.ru")
    monkeypatch.setenv("BITRIX_OAUTH_CLIENT_ID", "local.123")

    settings = get_settings()

    assert settings.resolved_bitrix_app_url == "https://ai.example.com/bitrix/app"
    assert settings.resolved_bitrix_oauth_callback_url == "https://ai.example.com/bitrix/oauth/callback"
    assert settings.resolved_bitrix_oauth_start_url.startswith(
        "https://portal.bitrix24.ru/oauth/authorize/?client_id=local.123"
    )


def test_technical_footer_allowed_user_ids(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ALLOWED_USER_IDS", "1,9")

    settings = get_settings()

    assert settings.resolved_tech_footer_allowed_user_ids == [1, 9]
