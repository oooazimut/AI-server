from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ai_server.models import AgentManifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedIndexEntry:
    id: str
    title: str
    file: str
    reason: str
    content: str
    matched_context_keys: list[str]
    matched_keywords: list[str]
    matched_statuses: list[str]
    match_reasons: list[str]
    priority: int = 0


def load_index_entries(
    manifest: AgentManifest,
    *,
    index_path: Path | None,
    section: str,
    request: str,
    context: dict[str, Any] | None = None,
    statuses: list[str] | None = None,
    default_file_for_id: Callable[[str], str] | None = None,
    entry_label: str = "entry",
) -> list[LoadedIndexEntry]:
    if index_path is None or not index_path.exists():
        return []

    index = _read_yaml(index_path)
    raw_entries = index.get(section)
    if not isinstance(raw_entries, list):
        return []

    loaded: list[LoadedIndexEntry] = []
    fallback_entries: list[LoadedIndexEntry] = []
    context = context or {}
    statuses = statuses or []
    request_text = request.casefold()

    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue

        use_when = raw_entry.get("use_when") if isinstance(raw_entry.get("use_when"), dict) else {}
        matched_context_keys = _matched_context_keys(use_when.get("context_keys"), context)
        matched_keywords = _matched_keywords(use_when.get("request_topics"), request_text)
        matched_statuses = _matched_statuses(use_when.get("statuses"), statuses)
        always_load = bool(use_when.get("always_load"))
        fallback = bool(use_when.get("fallback"))
        default_for_orchestrator = bool(use_when.get("default_for_orchestrator")) and manifest.kind == "orchestrator"
        match_reasons = _match_reasons(
            always_load=always_load,
            default_for_orchestrator=default_for_orchestrator,
            matched_context_keys=matched_context_keys,
            matched_keywords=matched_keywords,
            matched_statuses=matched_statuses,
        )

        if fallback:
            entry = _load_entry(
                raw_entry,
                index_path=index_path,
                default_file_for_id=default_file_for_id,
                entry_label=entry_label,
                matched_context_keys=[],
                matched_keywords=[],
                matched_statuses=[],
                match_reasons=["fallback"],
            )
            if entry is not None:
                fallback_entries.append(entry)
            continue

        if not (always_load or default_for_orchestrator or matched_context_keys or matched_keywords or matched_statuses):
            continue

        entry = _load_entry(
            raw_entry,
            index_path=index_path,
            default_file_for_id=default_file_for_id,
            entry_label=entry_label,
            matched_context_keys=matched_context_keys,
            matched_keywords=matched_keywords,
            matched_statuses=matched_statuses,
            match_reasons=match_reasons,
        )
        if entry is not None:
            loaded.append(entry)

    selected = loaded if loaded else fallback_entries
    return sorted(selected, key=lambda entry: entry.priority, reverse=True)


def _load_entry(
    raw_entry: dict[str, Any],
    *,
    index_path: Path,
    default_file_for_id: Callable[[str], str] | None,
    entry_label: str,
    matched_context_keys: list[str],
    matched_keywords: list[str],
    matched_statuses: list[str],
    match_reasons: list[str],
) -> LoadedIndexEntry | None:
    entry_id = str(raw_entry.get("id") or "").strip()
    file_name = str(raw_entry.get("file") or "").strip()
    if not file_name and entry_id and default_file_for_id is not None:
        file_name = default_file_for_id(entry_id)

    entry_path = _resolve_relative(index_path, file_name)
    if entry_path is None or not entry_path.exists():
        logger.warning("%s file not found for %s: %s", entry_label, raw_entry.get("id"), file_name)
        return None

    try:
        content = entry_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Failed to load %s file %s: %s", entry_label, entry_path, exc)
        return None

    return LoadedIndexEntry(
        id=entry_id or entry_path.stem,
        title=str(raw_entry.get("title") or _title_from_markdown(content) or entry_path.stem),
        file=file_name,
        reason=str(raw_entry.get("load_reason") or ""),
        content=content,
        matched_context_keys=matched_context_keys,
        matched_keywords=matched_keywords,
        matched_statuses=matched_statuses,
        match_reasons=match_reasons,
        priority=_int(raw_entry.get("priority")),
    )


def format_loaded_index_entries(entries: list[LoadedIndexEntry], *, heading: str, id_label: str) -> str:
    if not entries:
        return ""
    sections = [heading]
    for entry in entries:
        sections.append(
            "\n".join(
                [
                    f"## {entry.title}",
                    f"{id_label}: {entry.id}",
                    f"file: {entry.file}",
                    f"reason: {entry.reason}",
                    "",
                    entry.content,
                ]
            )
        )
    return "\n\n".join(sections)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
    except Exception as exc:
        logger.warning("Failed to load index %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_relative(base_file: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else base_file.parent / path


def _matched_context_keys(raw_keys: object, context: dict[str, Any]) -> list[str]:
    if not isinstance(raw_keys, list):
        return []
    return [str(key) for key in raw_keys if str(key) in context]


def _matched_keywords(raw_keywords: object, request_text: str) -> list[str]:
    if not isinstance(raw_keywords, list):
        return []
    return [str(keyword) for keyword in raw_keywords if str(keyword).casefold() in request_text]


def _matched_statuses(raw_statuses: object, statuses: list[str]) -> list[str]:
    if not isinstance(raw_statuses, list):
        return []
    known = {str(status) for status in statuses}
    return [str(status) for status in raw_statuses if str(status) in known]


def _match_reasons(
    *,
    always_load: bool,
    default_for_orchestrator: bool,
    matched_context_keys: list[str],
    matched_keywords: list[str],
    matched_statuses: list[str],
) -> list[str]:
    reasons: list[str] = []
    if always_load:
        reasons.append("always_load")
    if default_for_orchestrator:
        reasons.append("default_for_orchestrator")
    if matched_context_keys:
        reasons.append("context_keys")
    if matched_keywords:
        reasons.append("request_topics")
    if matched_statuses:
        reasons.append("statuses")
    return reasons


def _title_from_markdown(content: str) -> str | None:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
