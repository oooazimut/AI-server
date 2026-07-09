from ai_server.settings import get_settings


def test_settings_loads_layered_env_files(monkeypatch, tmp_path):
    first = tmp_path / "first.env"
    second = tmp_path / "second.env"
    first.write_text(
        "AI_SERVER_LLM_PROVIDER=from-file\nAI_SERVER_LLM_MODEL=first-model\nAI_SERVER_LLM_API_KEY=file-secret\n",
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


def test_bitrix_oauth_bot_settings_can_be_loaded(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_BOT_AUTH_MODE", "oauth")
    monkeypatch.setenv("BITRIX_BOT_OAUTH_USER_ID", "9")

    settings = get_settings()

    assert settings.bitrix_bot_uses_oauth is True
    assert settings.bitrix_bot_oauth_user_id == 9


def test_scheduler_can_be_disabled(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_SCHEDULER_ENABLED", "false")

    settings = get_settings()

    assert settings.scheduler_enabled is False


def test_diagnost_can_be_disabled(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("DIAGNOST_ENABLED", "false")

    settings = get_settings()

    assert settings.diagnost_enabled is False


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


def test_bitrix_portal_base_url_prefers_explicit_domain(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("BITRIX_DOMAIN", "portal.bitrix24.ru")
    monkeypatch.setenv("BITRIX_REST_WEBHOOK_URL", "https://other.bitrix24.ru/rest/1/token/")

    settings = get_settings()

    assert settings.bitrix_portal_base_url == "https://portal.bitrix24.ru"


def test_bitrix_portal_base_url_falls_back_to_webhook_domain(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.delenv("BITRIX_DOMAIN", raising=False)
    monkeypatch.setenv("BITRIX_REST_WEBHOOK_URL", "https://asutp-expert.bitrix24.ru/rest/9/token/")

    settings = get_settings()

    assert settings.bitrix_portal_base_url == "https://asutp-expert.bitrix24.ru"


def test_bitrix_portal_base_url_empty_when_unconfigured(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.delenv("BITRIX_DOMAIN", raising=False)
    monkeypatch.delenv("BITRIX_REST_WEBHOOK_URL", raising=False)

    settings = get_settings()

    assert settings.bitrix_portal_base_url == ""


def test_technical_footer_allowed_user_ids(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.setenv("AI_SERVER_TECH_FOOTER_ALLOWED_USER_IDS", "1,9")

    settings = get_settings()

    assert settings.resolved_tech_footer_allowed_user_ids == [1, 9]


def test_search_index_task_comment_settings(monkeypatch):
    monkeypatch.setenv("AI_SERVER_ENV_FILE", "")
    monkeypatch.delenv("SEARCH_INDEX_INCLUDE_TASK_COMMENTS", raising=False)
    monkeypatch.delenv("SEARCH_INDEX_TASK_COMMENT_LIMIT", raising=False)

    defaults = get_settings()

    assert defaults.search_index_include_task_comments is True
    assert defaults.search_index_task_comment_limit == 20

    monkeypatch.setenv("SEARCH_INDEX_INCLUDE_TASK_COMMENTS", "false")
    monkeypatch.setenv("SEARCH_INDEX_TASK_COMMENT_LIMIT", "5")

    overridden = get_settings()

    assert overridden.search_index_include_task_comments is False
    assert overridden.search_index_task_comment_limit == 5
