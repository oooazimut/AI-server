import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ai_server.registry import PROJECT_ROOT
from ai_server.runtime import runtime_paths


_LOADED_ENV_FILE_SPEC: str | None = None
_LOADED_ENV_KEYS: set[str] = set()


@dataclass(frozen=True)
class Settings:
    bitrix_bot_id: int | None
    bitrix_bot_token: str
    bitrix_bot_auth_mode: str
    bitrix_bot_oauth_user_id: int | None
    bitrix_bot_webhook_url: str
    bitrix_domain: str
    bitrix_oauth_client_id: str
    bitrix_oauth_client_secret: str
    bitrix_oauth_enabled: bool
    bitrix_oauth_required_for_writes: bool
    bitrix_oauth_db_path_override: str
    bitrix_oauth_token_endpoint: str
    bitrix_rest_webhook_url: str
    bitrix_projects_webhook_url: str
    public_base_url: str
    webhook_secret: str
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: str
    llm_temperature: float | None
    llm_max_tokens: int
    deepseek_api_key: str
    deepseek_balance_base_url: str
    deepseek_balance_timeout_seconds: float
    tech_footer_enabled: bool
    tech_footer_allowed_user_ids: str
    tech_footer_balance_enabled: bool
    tech_footer_balance_cache_seconds: int
    webhook_event_queue_enabled: bool
    webhook_event_worker_enabled: bool
    webhook_event_queue_interval_seconds: int
    webhook_event_queue_worker_count: int
    webhook_event_queue_claim_scan_limit: int
    webhook_event_queue_max_attempts: int
    webhook_event_queue_retry_base_seconds: int
    webhook_event_queue_retry_max_seconds: int
    webhook_event_queue_stale_processing_seconds: int
    search_index_max_tasks: int
    search_index_max_projects: int
    search_index_max_storages: int
    search_index_max_disk_items: int
    search_index_max_task_attachments: int
    search_index_disk_max_depth: int
    search_index_include_disk: bool
    search_index_include_task_attachments: bool
    search_content_enabled: bool
    search_content_keep_local_files: bool
    search_content_max_files: int
    search_content_max_bytes: int
    search_content_max_chars: int
    search_content_allowed_extensions: str
    search_background_indexer_enabled: bool
    search_background_initial_delay_seconds: int
    search_background_metadata_interval_seconds: int
    search_background_content_interval_seconds: int
    search_delta_indexer_enabled: bool
    search_delta_interval_seconds: int
    search_delta_folders_per_run: int
    search_delta_max_children_per_folder: int
    search_background_lock_stale_seconds: int
    search_webhook_indexer_enabled: bool
    search_webhook_content_enabled: bool
    quality_control_webhook_enabled: bool
    quality_control_webhook_auto_managed_only: bool
    quality_control_dry_run: bool
    quality_control_notify_only: bool
    quality_control_director_user_id: int | None
    quality_control_additional_notify_user_ids: str
    quality_control_notify_responsible: bool
    quality_control_notify_director: bool
    quality_control_actor_user_id: int | None
    quality_control_smart_enabled: bool
    quality_control_exempt_responsible_user_ids: str
    quality_control_auto_manage_project_id: int | None
    supervisor_enabled: bool
    supervisor_dry_run: bool
    supervisor_interval_seconds: int
    supervisor_initial_delay_seconds: int
    supervisor_max_tasks: int
    supervisor_max_tasks_per_user: int
    supervisor_admin_user_ids: str
    supervisor_notify_responsibles: bool
    supervisor_reminder_cooldown_hours: int
    reconcile_enabled: bool
    reconcile_interval_seconds: int
    reconcile_initial_delay_seconds: int
    reconcile_tasks_enabled: bool
    reconcile_task_lookback_hours: int
    reconcile_task_limit: int
    reconcile_disk_delta_enabled: bool
    vehicle_usage_manager_user_id: int | None
    vehicle_usage_dialog_id: str
    vehicle_usage_staff_roster: str
    vehicle_usage_dry_run: bool
    attachment_max_bytes: int
    stt_provider: str
    transcription_max_bytes: int
    yandex_api_key: str
    yandex_iam_token: str
    yandex_folder_id: str
    yandex_speechkit_base_url: str
    yandex_speechkit_lang: str
    yandex_speechkit_max_bytes: int
    yandex_speechkit_convert_to_ogg: bool
    ffmpeg_path: str
    agent_write_allowed_user_ids: str
    agent_limited_task_create_project_id: int | None
    agent_limited_task_create_user_ids: str
    agent_private_disk_path_markers: str
    agent_private_disk_restricted_user_ids: str
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
    def llm_configured(self) -> bool:
        return bool(self.llm_provider and self.llm_model and self.llm_api_key)

    @property
    def bitrix_oauth_db_path(self) -> Path:
        if self.bitrix_oauth_db_path_override:
            return Path(self.bitrix_oauth_db_path_override)
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

    @property
    def dialog_state_path(self) -> Path:
        return runtime_paths(self.var_dir).dialog_state_db

    @property
    def bitrix_write_audit_log_path(self) -> Path:
        return runtime_paths(self.var_dir).bitrix_write_audit_log

    @property
    def quality_control_state_path(self) -> Path:
        return runtime_paths(self.var_dir).quality_control_state

    @property
    def supervisor_state_path(self) -> Path:
        return runtime_paths(self.var_dir).supervisor_state

    @property
    def vehicle_usage_db_path(self) -> Path:
        return runtime_paths(self.var_dir).vehicle_usage_db

    @property
    def attachment_storage_dir(self) -> Path:
        return runtime_paths(self.var_dir).attachments_dir

    @property
    def transcription_configured(self) -> bool:
        return self.stt_provider == "yandex_speechkit" and bool(
            self.yandex_api_key or (self.yandex_iam_token and self.yandex_folder_id)
        )

    @property
    def resolved_supervisor_admin_user_ids(self) -> list[int]:
        return _id_list(self.supervisor_admin_user_ids)

    @property
    def resolved_agent_write_allowed_user_ids(self) -> list[int]:
        return _id_list(self.agent_write_allowed_user_ids)

    @property
    def resolved_agent_limited_task_create_user_ids(self) -> list[int]:
        return _id_list(self.agent_limited_task_create_user_ids)

    @property
    def resolved_agent_private_disk_path_markers(self) -> list[str]:
        return [
            item.strip()
            for item in self.agent_private_disk_path_markers.replace(";", ",").split(",")
            if item.strip()
        ]

    @property
    def resolved_agent_private_disk_restricted_user_ids(self) -> list[int]:
        return _id_list(self.agent_private_disk_restricted_user_ids)

    @property
    def resolved_tech_footer_allowed_user_ids(self) -> list[int]:
        return _id_list(self.tech_footer_allowed_user_ids)

    @property
    def resolved_quality_control_director_user_ids(self) -> list[int]:
        ids: list[int] = []
        if self.quality_control_director_user_id:
            ids.append(self.quality_control_director_user_id)
        ids.extend(_id_list(self.quality_control_additional_notify_user_ids))
        return _unique_ints(ids)

    @property
    def resolved_quality_control_exempt_responsible_user_ids(self) -> list[int]:
        return _id_list(self.quality_control_exempt_responsible_user_ids)

    @property
    def search_index_path(self) -> Path:
        return runtime_paths(self.var_dir).search_index_db

    @property
    def search_background_state_path(self) -> Path:
        return runtime_paths(self.var_dir).search_indexer_state

    @property
    def search_background_lock_path(self) -> Path:
        return runtime_paths(self.var_dir).search_indexer_lock

    @property
    def search_content_storage_dir(self) -> Path:
        return runtime_paths(self.var_dir).search_content_dir

    @property
    def resolved_search_content_allowed_extensions(self) -> set[str]:
        return {
            extension if extension.startswith(".") else f".{extension}"
            for extension in (
                part.strip().lower()
                for part in self.search_content_allowed_extensions.replace(";", ",").split(",")
            )
            if extension
        }

    def _with_webhook_secret(self, url: str) -> str:
        if not self.webhook_secret:
            return url
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["secret"] = self.webhook_secret
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def get_settings() -> Settings:
    _load_env_files()
    paths = runtime_paths()
    return Settings(
        bitrix_bot_id=_env_int("BITRIX_BOT_ID"),
        bitrix_bot_token=_env("BITRIX_BOT_TOKEN"),
        bitrix_bot_auth_mode=_env("BITRIX_BOT_AUTH_MODE", "webhook"),
        bitrix_bot_oauth_user_id=_env_int("BITRIX_BOT_OAUTH_USER_ID"),
        bitrix_bot_webhook_url=_env("BITRIX_BOT_WEBHOOK_URL"),
        bitrix_domain=_env("BITRIX_DOMAIN"),
        bitrix_oauth_client_id=_env("BITRIX_OAUTH_CLIENT_ID"),
        bitrix_oauth_client_secret=_env("BITRIX_OAUTH_CLIENT_SECRET"),
        bitrix_oauth_enabled=_env_bool("BITRIX_OAUTH_ENABLED", True),
        bitrix_oauth_required_for_writes=_env_bool("BITRIX_OAUTH_REQUIRED_FOR_WRITES", True),
        bitrix_oauth_db_path_override=_env("BITRIX_OAUTH_DB_PATH"),
        bitrix_oauth_token_endpoint=_env("BITRIX_OAUTH_TOKEN_ENDPOINT", "https://oauth.bitrix.info/oauth/token/"),
        bitrix_rest_webhook_url=_env("BITRIX_REST_WEBHOOK_URL"),
        bitrix_projects_webhook_url=_env("BITRIX_PROJECTS_WEBHOOK_URL"),
        public_base_url=_env("PUBLIC_BASE_URL"),
        webhook_secret=_env("WEBHOOK_SECRET"),
        llm_provider=_env("AI_SERVER_LLM_PROVIDER", _env("LLM_PROVIDER", "deepseek")),
        llm_model=_env("AI_SERVER_LLM_MODEL", _env("LLM_MODEL", "deepseek-v4-flash")),
        llm_base_url=_env("AI_SERVER_LLM_BASE_URL", _env("LLM_BASE_URL")),
        llm_api_key=_env("AI_SERVER_LLM_API_KEY", _env("LLM_API_KEY")),
        llm_temperature=_env_float("AI_SERVER_LLM_TEMPERATURE", _env_float("LLM_TEMPERATURE")),
        llm_max_tokens=_env_int("AI_SERVER_LLM_MAX_TOKENS", _env_int("LLM_MAX_TOKENS", 3000)) or 3000,
        deepseek_api_key=_deepseek_api_key(),
        deepseek_balance_base_url=_env("AI_SERVER_DEEPSEEK_BALANCE_BASE_URL", "https://api.deepseek.com"),
        deepseek_balance_timeout_seconds=_env_float("AI_SERVER_DEEPSEEK_BALANCE_TIMEOUT_SECONDS", 10.0) or 10.0,
        tech_footer_enabled=_env_bool("AI_SERVER_TECH_FOOTER_ENABLED", True),
        tech_footer_allowed_user_ids=_env("AI_SERVER_TECH_FOOTER_ALLOWED_USER_IDS"),
        tech_footer_balance_enabled=_env_bool("AI_SERVER_TECH_FOOTER_BALANCE_ENABLED", True),
        tech_footer_balance_cache_seconds=_env_int("AI_SERVER_TECH_FOOTER_BALANCE_CACHE_SECONDS", 300) or 300,
        webhook_event_queue_enabled=_env_bool("WEBHOOK_EVENT_QUEUE_ENABLED", True),
        webhook_event_worker_enabled=_env_bool("AI_SERVER_WEBHOOK_EVENT_WORKER_ENABLED", False),
        webhook_event_queue_interval_seconds=_env_int("WEBHOOK_EVENT_QUEUE_INTERVAL_SECONDS", 2) or 2,
        webhook_event_queue_worker_count=_env_int("WEBHOOK_EVENT_QUEUE_WORKER_COUNT", 1) or 1,
        webhook_event_queue_claim_scan_limit=_env_int("WEBHOOK_EVENT_QUEUE_CLAIM_SCAN_LIMIT", 50) or 50,
        webhook_event_queue_max_attempts=_env_int("WEBHOOK_EVENT_QUEUE_MAX_ATTEMPTS", 8) or 8,
        webhook_event_queue_retry_base_seconds=_env_int("WEBHOOK_EVENT_QUEUE_RETRY_BASE_SECONDS", 10) or 10,
        webhook_event_queue_retry_max_seconds=_env_int("WEBHOOK_EVENT_QUEUE_RETRY_MAX_SECONDS", 300) or 300,
        webhook_event_queue_stale_processing_seconds=_env_int("WEBHOOK_EVENT_QUEUE_STALE_PROCESSING_SECONDS", 300) or 300,
        search_index_max_tasks=_env_int("SEARCH_INDEX_MAX_TASKS", 5000) or 5000,
        search_index_max_projects=_env_int("SEARCH_INDEX_MAX_PROJECTS", 200) or 200,
        search_index_max_storages=_env_int("SEARCH_INDEX_MAX_STORAGES", 500) or 500,
        search_index_max_disk_items=_env_int("SEARCH_INDEX_MAX_DISK_ITEMS", 50000) or 50000,
        search_index_max_task_attachments=_env_int("SEARCH_INDEX_MAX_TASK_ATTACHMENTS", 5000) or 5000,
        search_index_disk_max_depth=_env_int("SEARCH_INDEX_DISK_MAX_DEPTH", 6) or 6,
        search_index_include_disk=_env_bool("SEARCH_INDEX_INCLUDE_DISK", True),
        search_index_include_task_attachments=_env_bool("SEARCH_INDEX_INCLUDE_TASK_ATTACHMENTS", True),
        search_content_enabled=_env_bool("SEARCH_CONTENT_ENABLED", True),
        search_content_keep_local_files=_env_bool("SEARCH_CONTENT_KEEP_LOCAL_FILES", False),
        search_content_max_files=_env_int("SEARCH_CONTENT_MAX_FILES", 80) or 80,
        search_content_max_bytes=_env_int("SEARCH_CONTENT_MAX_BYTES", 20 * 1024 * 1024) or (20 * 1024 * 1024),
        search_content_max_chars=_env_int("SEARCH_CONTENT_MAX_CHARS", 40_000) or 40_000,
        search_content_allowed_extensions=_env("SEARCH_CONTENT_ALLOWED_EXTENSIONS", ".txt,.csv,.doc,.docx,.xlsx,.xls,.pdf"),
        search_background_indexer_enabled=_env_bool("SEARCH_BACKGROUND_INDEXER_ENABLED", False),
        search_background_initial_delay_seconds=_env_int("SEARCH_BACKGROUND_INITIAL_DELAY_SECONDS", 60) or 60,
        search_background_metadata_interval_seconds=_env_int("SEARCH_BACKGROUND_METADATA_INTERVAL_SECONDS", 6 * 60 * 60) or (6 * 60 * 60),
        search_background_content_interval_seconds=_env_int("SEARCH_BACKGROUND_CONTENT_INTERVAL_SECONDS", 10 * 60) or (10 * 60),
        search_delta_indexer_enabled=_env_bool("SEARCH_DELTA_INDEXER_ENABLED", True),
        search_delta_interval_seconds=_env_int("SEARCH_DELTA_INTERVAL_SECONDS", 5 * 60) or (5 * 60),
        search_delta_folders_per_run=_env_int("SEARCH_DELTA_FOLDERS_PER_RUN", 15) or 15,
        search_delta_max_children_per_folder=_env_int("SEARCH_DELTA_MAX_CHILDREN_PER_FOLDER", 1000) or 1000,
        search_background_lock_stale_seconds=_env_int("SEARCH_BACKGROUND_LOCK_STALE_SECONDS", 2 * 60 * 60) or (2 * 60 * 60),
        search_webhook_indexer_enabled=_env_bool("SEARCH_WEBHOOK_INDEXER_ENABLED", False),
        search_webhook_content_enabled=_env_bool("SEARCH_WEBHOOK_CONTENT_ENABLED", True),
        quality_control_webhook_enabled=_env_bool("QUALITY_CONTROL_WEBHOOK_ENABLED", False),
        quality_control_webhook_auto_managed_only=_env_bool("QUALITY_CONTROL_WEBHOOK_AUTO_MANAGED_ONLY", True),
        quality_control_dry_run=_env_bool("QUALITY_CONTROL_DRY_RUN", True),
        quality_control_notify_only=_env_bool("QUALITY_CONTROL_NOTIFY_ONLY", False),
        quality_control_director_user_id=_env_int("QUALITY_CONTROL_DIRECTOR_USER_ID"),
        quality_control_additional_notify_user_ids=_env("QUALITY_CONTROL_ADDITIONAL_NOTIFY_USER_IDS"),
        quality_control_notify_responsible=_env_bool("QUALITY_CONTROL_NOTIFY_RESPONSIBLE", True),
        quality_control_notify_director=_env_bool("QUALITY_CONTROL_NOTIFY_DIRECTOR", True),
        quality_control_actor_user_id=_env_int("QUALITY_CONTROL_ACTOR_USER_ID"),
        quality_control_smart_enabled=_env_bool("QUALITY_CONTROL_SMART_ENABLED", True),
        quality_control_exempt_responsible_user_ids=_env("QUALITY_CONTROL_EXEMPT_RESPONSIBLE_USER_IDS"),
        quality_control_auto_manage_project_id=_env_int("QUALITY_CONTROL_AUTO_MANAGE_PROJECT_ID"),
        supervisor_enabled=_env_bool("SUPERVISOR_ENABLED", False),
        supervisor_dry_run=_env_bool("SUPERVISOR_DRY_RUN", True),
        supervisor_interval_seconds=_env_int("SUPERVISOR_INTERVAL_SECONDS", 60 * 60) or (60 * 60),
        supervisor_initial_delay_seconds=_env_int("SUPERVISOR_INITIAL_DELAY_SECONDS", 60) or 60,
        supervisor_max_tasks=_env_int("SUPERVISOR_MAX_TASKS", 50) or 50,
        supervisor_max_tasks_per_user=_env_int("SUPERVISOR_MAX_TASKS_PER_USER", 10) or 10,
        supervisor_admin_user_ids=_env("SUPERVISOR_ADMIN_USER_IDS"),
        supervisor_notify_responsibles=_env_bool("SUPERVISOR_NOTIFY_RESPONSIBLES", False),
        supervisor_reminder_cooldown_hours=_env_int("SUPERVISOR_REMINDER_COOLDOWN_HOURS", 12) or 12,
        reconcile_enabled=_env_bool("RECONCILE_ENABLED", False),
        reconcile_interval_seconds=_env_int("RECONCILE_INTERVAL_SECONDS", 15 * 60) or (15 * 60),
        reconcile_initial_delay_seconds=_env_int("RECONCILE_INITIAL_DELAY_SECONDS", 120) or 120,
        reconcile_tasks_enabled=_env_bool("RECONCILE_TASKS_ENABLED", True),
        reconcile_task_lookback_hours=_env_int("RECONCILE_TASK_LOOKBACK_HOURS", 24) or 24,
        reconcile_task_limit=_env_int("RECONCILE_TASK_LIMIT", 500) or 500,
        reconcile_disk_delta_enabled=_env_bool("RECONCILE_DISK_DELTA_ENABLED", True),
        vehicle_usage_manager_user_id=_env_int("VEHICLE_USAGE_MANAGER_USER_ID"),
        vehicle_usage_dialog_id=_env("VEHICLE_USAGE_DIALOG_ID"),
        vehicle_usage_staff_roster=_env("VEHICLE_USAGE_STAFF_ROSTER"),
        vehicle_usage_dry_run=_env_bool("VEHICLE_USAGE_DRY_RUN", True),
        attachment_max_bytes=_env_int("ATTACHMENT_MAX_BYTES", 30 * 1024 * 1024) or (30 * 1024 * 1024),
        stt_provider=_env("STT_PROVIDER", "yandex_speechkit"),
        transcription_max_bytes=_env_int("TRANSCRIPTION_MAX_BYTES", 25 * 1024 * 1024) or (25 * 1024 * 1024),
        yandex_api_key=_env("YANDEX_API_KEY"),
        yandex_iam_token=_env("YANDEX_IAM_TOKEN"),
        yandex_folder_id=_env("YANDEX_FOLDER_ID"),
        yandex_speechkit_base_url=_env("YANDEX_SPEECHKIT_BASE_URL", "https://stt.api.cloud.yandex.net"),
        yandex_speechkit_lang=_env("YANDEX_SPEECHKIT_LANG", "ru-RU"),
        yandex_speechkit_max_bytes=_env_int("YANDEX_SPEECHKIT_MAX_BYTES", 1024 * 1024) or (1024 * 1024),
        yandex_speechkit_convert_to_ogg=_env_bool("YANDEX_SPEECHKIT_CONVERT_TO_OGG", True),
        ffmpeg_path=_env("FFMPEG_PATH", "ffmpeg"),
        agent_write_allowed_user_ids=_env("AGENT_WRITE_ALLOWED_USER_IDS"),
        agent_limited_task_create_project_id=_env_int("AGENT_LIMITED_TASK_CREATE_PROJECT_ID"),
        agent_limited_task_create_user_ids=_env("AGENT_LIMITED_TASK_CREATE_USER_IDS"),
        agent_private_disk_path_markers=_env("AGENT_PRIVATE_DISK_PATH_MARKERS", "Приватный доступ"),
        agent_private_disk_restricted_user_ids=_env("AGENT_PRIVATE_DISK_RESTRICTED_USER_IDS"),
        agent_dry_run=_env_bool("AGENT_DRY_RUN", False),
        var_dir=paths.root,
    )


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _deepseek_api_key() -> str:
    explicit = _env("AI_SERVER_DEEPSEEK_API_KEY", _env("DEEPSEEK_API_KEY"))
    if explicit:
        return explicit
    provider = _env("AI_SERVER_LLM_PROVIDER", _env("LLM_PROVIDER", "deepseek")).casefold()
    if provider == "deepseek":
        return _env("AI_SERVER_LLM_API_KEY", _env("LLM_API_KEY"))
    return ""


def _load_env_files() -> None:
    global _LOADED_ENV_FILE_SPEC

    raw_spec = os.getenv("AI_SERVER_ENV_FILE", ".env,.env.local")
    if raw_spec == _LOADED_ENV_FILE_SPEC:
        return

    for raw_path in _split_env_file_spec(raw_spec):
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        _load_env_file(path)
    _LOADED_ENV_FILE_SPEC = raw_spec


def _split_env_file_spec(raw_spec: str) -> list[str]:
    return [item.strip() for item in raw_spec.replace(";", ",").split(",") if item.strip()]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if key in os.environ and key not in _LOADED_ENV_KEYS:
            continue
        os.environ[key] = value
        _LOADED_ENV_KEYS.add(key)


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


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


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _id_list(raw: str) -> list[int]:
    ids: list[int] = []
    for item in raw.replace(";", ",").split(","):
        value = item.strip()
        if not value:
            continue
        try:
            ids.append(int(value))
        except ValueError:
            continue
    return ids


def _unique_ints(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
