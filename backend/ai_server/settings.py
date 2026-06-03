import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ai_server.runtime import runtime_paths


@dataclass(frozen=True)
class Settings:
    bitrix_bot_id: int | None
    bitrix_bot_token: str
    bitrix_bot_auth_mode: str
    bitrix_bot_webhook_url: str
    bitrix_oauth_client_id: str
    bitrix_oauth_client_secret: str
    bitrix_oauth_enabled: bool
    bitrix_oauth_required_for_writes: bool
    bitrix_oauth_token_endpoint: str
    bitrix_rest_webhook_url: str
    bitrix_projects_webhook_url: str
    public_base_url: str
    webhook_secret: str
    webhook_event_queue_enabled: bool
    webhook_event_worker_enabled: bool
    webhook_event_queue_interval_seconds: int
    webhook_event_queue_worker_count: int
    webhook_event_queue_claim_scan_limit: int
    webhook_event_queue_max_attempts: int
    webhook_event_queue_retry_base_seconds: int
    webhook_event_queue_retry_max_seconds: int
    webhook_event_queue_stale_processing_seconds: int
    agent_dry_run: bool
    var_dir: Path

    @property
    def bitrix_configured(self) -> bool:
        return bool(self.bitrix_rest_webhook_url)

    @property
    def bitrix_bot_uses_oauth(self) -> bool:
        return self.bitrix_bot_auth_mode.strip().lower() == "oauth"

    @property
    def bitrix_oauth_configured(self) -> bool:
        return bool(self.bitrix_oauth_client_id and self.bitrix_oauth_client_secret)

    @property
    def bitrix_oauth_db_path(self) -> Path:
        return runtime_paths(self.var_dir).bitrix_oauth_db

    @property
    def resolved_bot_webhook_url(self) -> str:
        if self.bitrix_bot_webhook_url:
            return self._with_webhook_secret(self.bitrix_bot_webhook_url)
        if self.public_base_url:
            return self._with_webhook_secret(self.public_base_url.rstrip("/") + "/bitrix/events")
        return ""

    @property
    def webhook_event_queue_path(self) -> Path:
        return runtime_paths(self.var_dir).webhook_event_queue_db

    def _with_webhook_secret(self, url: str) -> str:
        if not self.webhook_secret:
            return url
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["secret"] = self.webhook_secret
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def get_settings() -> Settings:
    paths = runtime_paths()
    return Settings(
        bitrix_bot_id=_env_int("BITRIX_BOT_ID"),
        bitrix_bot_token=_env("BITRIX_BOT_TOKEN"),
        bitrix_bot_auth_mode=_env("BITRIX_BOT_AUTH_MODE", "webhook"),
        bitrix_bot_webhook_url=_env("BITRIX_BOT_WEBHOOK_URL"),
        bitrix_oauth_client_id=_env("BITRIX_OAUTH_CLIENT_ID"),
        bitrix_oauth_client_secret=_env("BITRIX_OAUTH_CLIENT_SECRET"),
        bitrix_oauth_enabled=_env_bool("BITRIX_OAUTH_ENABLED", True),
        bitrix_oauth_required_for_writes=_env_bool("BITRIX_OAUTH_REQUIRED_FOR_WRITES", True),
        bitrix_oauth_token_endpoint=_env("BITRIX_OAUTH_TOKEN_ENDPOINT", "https://oauth.bitrix.info/oauth/token/"),
        bitrix_rest_webhook_url=_env("BITRIX_REST_WEBHOOK_URL"),
        bitrix_projects_webhook_url=_env("BITRIX_PROJECTS_WEBHOOK_URL"),
        public_base_url=_env("PUBLIC_BASE_URL"),
        webhook_secret=_env("WEBHOOK_SECRET"),
        webhook_event_queue_enabled=_env_bool("WEBHOOK_EVENT_QUEUE_ENABLED", True),
        webhook_event_worker_enabled=_env_bool("AI_SERVER_WEBHOOK_EVENT_WORKER_ENABLED", False),
        webhook_event_queue_interval_seconds=_env_int("WEBHOOK_EVENT_QUEUE_INTERVAL_SECONDS", 2) or 2,
        webhook_event_queue_worker_count=_env_int("WEBHOOK_EVENT_QUEUE_WORKER_COUNT", 1) or 1,
        webhook_event_queue_claim_scan_limit=_env_int("WEBHOOK_EVENT_QUEUE_CLAIM_SCAN_LIMIT", 50) or 50,
        webhook_event_queue_max_attempts=_env_int("WEBHOOK_EVENT_QUEUE_MAX_ATTEMPTS", 8) or 8,
        webhook_event_queue_retry_base_seconds=_env_int("WEBHOOK_EVENT_QUEUE_RETRY_BASE_SECONDS", 10) or 10,
        webhook_event_queue_retry_max_seconds=_env_int("WEBHOOK_EVENT_QUEUE_RETRY_MAX_SECONDS", 300) or 300,
        webhook_event_queue_stale_processing_seconds=_env_int("WEBHOOK_EVENT_QUEUE_STALE_PROCESSING_SECONDS", 300) or 300,
        agent_dry_run=_env_bool("AGENT_DRY_RUN", False),
        var_dir=paths.root,
    )


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default

