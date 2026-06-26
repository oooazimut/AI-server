from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ai_server.models import AgentManifest
from ai_server.registry import agent_package_path, resolve_project_path
from ai_server.skills import SkillStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedSkill:
    id: str
    title: str
    file: str
    reason: str
    content: str
    matched_context_keys: list[str]
    matched_keywords: list[str]
    priority: int = 0


def load_skills_for_task(
    manifest: AgentManifest,
    *,
    request: str,
    context: dict[str, Any] | None = None,
) -> list[LoadedSkill]:
    index_path = _skill_index_path(manifest)
    if index_path is None or not index_path.exists():
        return []

    index = _read_yaml(index_path)
    raw_skills = index.get("skills")
    if not isinstance(raw_skills, list):
        return []

    store = SkillStore()
    context = context or {}
    request_text = request.casefold()
    loaded: list[LoadedSkill] = []

    for raw_skill in raw_skills:
        if not isinstance(raw_skill, dict):
            continue
        use_when = raw_skill.get("use_when") if isinstance(raw_skill.get("use_when"), dict) else {}
        matched_context_keys = _matched_context_keys(use_when.get("context_keys"), context)
        matched_keywords = _matched_keywords(use_when.get("request_topics"), request_text)
        always_load = bool(use_when.get("always_load"))

        if not (always_load or matched_context_keys or matched_keywords):
            continue

        skill_id = str(raw_skill.get("id") or "").strip()
        if not skill_id:
            continue
        skill = store.read_skill(manifest, skill_id)
        if skill is None or skill.content is None:
            logger.warning("Skill file not found for %s", skill_id)
            continue

        loaded.append(
            LoadedSkill(
                id=skill.id,
                title=str(raw_skill.get("title") or skill.title),
                file=str(raw_skill.get("file") or f"skills/{skill.id}.md"),
                reason=str(raw_skill.get("load_reason") or ""),
                content=skill.content,
                matched_context_keys=matched_context_keys,
                matched_keywords=matched_keywords,
                priority=_int(raw_skill.get("priority")),
            )
        )

    return sorted(loaded, key=lambda skill: skill.priority, reverse=True)


def format_loaded_skills(skills: list[LoadedSkill]) -> str:
    if not skills:
        return ""
    sections = ["Подгруженные скилы для текущей Bitrix-задачи:"]
    for skill in skills:
        sections.append(
            "\n".join(
                [
                    f"## {skill.title}",
                    f"skill_id: {skill.id}",
                    f"file: {skill.file}",
                    f"reason: {skill.reason}",
                    "",
                    skill.content,
                ]
            )
        )
    return "\n\n".join(sections)


def _skill_index_path(manifest: AgentManifest) -> Path | None:
    if manifest.skills_path:
        skills_path = resolve_project_path(manifest.skills_path)
        if skills_path is not None:
            return skills_path.parent / "skill_index.yaml"
    return agent_package_path(manifest.id) / "skill_index.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
    except Exception as exc:
        logger.warning("Failed to load skill index %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _matched_context_keys(raw_keys: object, context: dict[str, Any]) -> list[str]:
    if not isinstance(raw_keys, list):
        return []
    return [str(key) for key in raw_keys if str(key) in context]


def _matched_keywords(raw_keywords: object, request_text: str) -> list[str]:
    if not isinstance(raw_keywords, list):
        return []
    return [str(keyword) for keyword in raw_keywords if str(keyword).casefold() in request_text]


def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
