from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_server.index_loader import LoadedIndexEntry, format_loaded_index_entries, load_index_entries
from ai_server.models import AgentManifest
from ai_server.registry import resolve_project_path


@dataclass(frozen=True)
class LoadedRule:
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


def load_rules_for_task(
    manifest: AgentManifest,
    *,
    request: str,
    context: dict[str, Any] | None = None,
    statuses: list[str] | None = None,
) -> list[LoadedRule]:
    entries = load_index_entries(
        manifest,
        index_path=_rule_index_path(manifest),
        section="rules",
        request=request,
        context=context,
        statuses=statuses,
        entry_label="Rule",
    )
    return [_to_loaded_rule(entry) for entry in entries]


def format_loaded_rules(rules: list[LoadedRule]) -> str:
    return format_loaded_index_entries(
        [_to_index_entry(rule) for rule in rules],
        heading="Подгруженные главы правил для текущей задачи:",
        id_label="rule_id",
    )


def _rule_index_path(manifest: AgentManifest) -> Path | None:
    if not manifest.instructions_file:
        return None
    instructions_path = resolve_project_path(manifest.instructions_file)
    if instructions_path is None:
        return None
    return instructions_path.parent / "rule_index.yaml"


def _to_loaded_rule(entry: LoadedIndexEntry) -> LoadedRule:
    return LoadedRule(
        id=entry.id,
        title=entry.title,
        file=entry.file,
        reason=entry.reason,
        content=entry.content,
        matched_context_keys=entry.matched_context_keys,
        matched_keywords=entry.matched_keywords,
        matched_statuses=entry.matched_statuses,
        match_reasons=entry.match_reasons,
        priority=entry.priority,
    )


def _to_index_entry(rule: LoadedRule) -> LoadedIndexEntry:
    return LoadedIndexEntry(
        id=rule.id,
        title=rule.title,
        file=rule.file,
        reason=rule.reason,
        content=rule.content,
        matched_context_keys=rule.matched_context_keys,
        matched_keywords=rule.matched_keywords,
        matched_statuses=rule.matched_statuses,
        match_reasons=rule.match_reasons,
        priority=rule.priority,
    )
