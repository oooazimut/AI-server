from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_server.index_loader import LoadedIndexEntry, format_loaded_index_entries, load_index_entries
from ai_server.models import AgentManifest
from ai_server.registry import agent_package_path, resolve_project_path


@dataclass(frozen=True)
class LoadedSkill:
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


def load_skills_for_task(
    manifest: AgentManifest,
    *,
    request: str,
    context: dict[str, Any] | None = None,
    statuses: list[str] | None = None,
) -> list[LoadedSkill]:
    entries = load_index_entries(
        manifest,
        index_path=_skill_index_path(manifest),
        section="skills",
        request=request,
        context=context,
        statuses=statuses,
        default_file_for_id=lambda skill_id: f"skills/{skill_id}.md",
        entry_label="Skill",
    )
    return [_to_loaded_skill(entry) for entry in entries]


def format_loaded_skills(skills: list[LoadedSkill]) -> str:
    return format_loaded_index_entries(
        [_to_index_entry(skill) for skill in skills],
        heading="Подгруженные скилы для текущей задачи:",
        id_label="skill_id",
    )


def _skill_index_path(manifest: AgentManifest) -> Path | None:
    if manifest.skills_path:
        skills_path = resolve_project_path(manifest.skills_path)
        if skills_path is not None:
            return skills_path.parent / "skill_index.yaml"
    return agent_package_path(manifest.id) / "skill_index.yaml"


def _to_loaded_skill(entry: LoadedIndexEntry) -> LoadedSkill:
    return LoadedSkill(
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


def _to_index_entry(skill: LoadedSkill) -> LoadedIndexEntry:
    return LoadedIndexEntry(
        id=skill.id,
        title=skill.title,
        file=skill.file,
        reason=skill.reason,
        content=skill.content,
        matched_context_keys=skill.matched_context_keys,
        matched_keywords=skill.matched_keywords,
        matched_statuses=skill.matched_statuses,
        match_reasons=skill.match_reasons,
        priority=skill.priority,
    )
