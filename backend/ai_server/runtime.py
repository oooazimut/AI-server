import os
from dataclasses import dataclass
from pathlib import Path

from ai_server.registry import PROJECT_ROOT


DEFAULT_VAR_DIR = PROJECT_ROOT / "var"


@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @property
    def attachments_dir(self) -> Path:
        return self.root / "attachments"

    @property
    def bitrix_oauth_db(self) -> Path:
        return self.root / "bitrix_oauth.sqlite"

    @property
    def bitrix_write_audit_log(self) -> Path:
        return self.root / "bitrix_write_audit.jsonl"

    @property
    def dialog_state_db(self) -> Path:
        return self.root / "dialog_state.sqlite"

    @property
    def document_drafts_dir(self) -> Path:
        return self.root / "document_drafts"

    @property
    def embedding_models_dir(self) -> Path:
        return self.root / "embedding_models"

    @property
    def quality_control_state(self) -> Path:
        return self.root / "quality_control_state.json"

    @property
    def learning_events_log(self) -> Path:
        return self.root / "learning_events.jsonl"

    @property
    def search_content_dir(self) -> Path:
        return self.root / "search_content"

    @property
    def search_index_db(self) -> Path:
        return self.root / "search_index.sqlite"

    @property
    def search_indexer_lock(self) -> Path:
        return self.root / "search_indexer.lock"

    @property
    def search_indexer_state(self) -> Path:
        return self.root / "search_indexer_state.json"

    @property
    def supervisor_state(self) -> Path:
        return self.root / "supervisor_state.json"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def vehicle_usage_db(self) -> Path:
        return self.root / "vehicle_usage.sqlite"

    @property
    def webhook_event_queue_db(self) -> Path:
        return self.root / "webhook_event_queue.sqlite"


def runtime_var_dir() -> Path:
    raw = os.getenv("AI_SERVER_VAR_DIR", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_VAR_DIR


def runtime_paths(root: Path | str | None = None) -> RuntimePaths:
    resolved = Path(root).expanduser() if root is not None else runtime_var_dir()
    return RuntimePaths(root=resolved)


def ensure_runtime_dirs(paths: RuntimePaths | None = None) -> None:
    selected = paths or runtime_paths()
    for directory in (
        selected.root,
        selected.attachments_dir,
        selected.document_drafts_dir,
        selected.embedding_models_dir,
        selected.search_content_dir,
        selected.tmp_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
