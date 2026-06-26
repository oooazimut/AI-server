from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ai_server.models import AgentManifest
from ai_server.registry import resolve_project_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedRule:
    id: str
    title: str
    file: str
    reason: str
    content: str
    matched_context_keys: list[str]
    matched_keywords: list[str]
    priority: int = 0


def load_rules_for_task(
    manifest: AgentManifest,
    *,
    request: str,
    context: dict[str, Any] | None = None,
    statuses: list[str] | None = None,
) -> list[LoadedRule]:
    index_path = _rule_index_path(manifest)
    if index_path is None or not index_path.exists():
        return []

    index = _read_yaml(index_path)
    raw_rules = index.get("rules")
    if not isinstance(raw_rules, list):
        return []

    loaded: list[LoadedRule] = []
    context = context or {}
    statuses = statuses or []
    request_text = request.casefold()

    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        use_when = raw_rule.get("use_when") if isinstance(raw_rule.get("use_when"), dict) else {}
        matched_context_keys = _matched_context_keys(use_when.get("context_keys"), context)
        matched_keywords = _matched_keywords(use_when.get("request_topics"), request_text)
        matched_statuses = _matched_statuses(use_when.get("statuses"), statuses)
        default_for_orchestrator = bool(use_when.get("default_for_orchestrator")) and manifest.kind == "orchestrator"

        if not (default_for_orchestrator or matched_context_keys or matched_keywords or matched_statuses):
            continue

        file_name = str(raw_rule.get("file") or "").strip()
        rule_path = _resolve_relative(index_path, file_name)
        if rule_path is None or not rule_path.exists():
            logger.warning("Rule file not found for %s: %s", raw_rule.get("id"), file_name)
            continue

        try:
            content = rule_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Failed to load rule file %s: %s", rule_path, exc)
            continue

        loaded.append(
            LoadedRule(
                id=str(raw_rule.get("id") or rule_path.stem),
                title=str(raw_rule.get("title") or rule_path.stem),
                file=file_name,
                reason=str(raw_rule.get("load_reason") or ""),
                content=content,
                matched_context_keys=matched_context_keys,
                matched_keywords=matched_keywords + matched_statuses,
                priority=_int(raw_rule.get("priority")),
            )
        )

    return sorted(loaded, key=lambda rule: rule.priority, reverse=True)


def format_loaded_rules(rules: list[LoadedRule]) -> str:
    if not rules:
        return ""
    sections = ["Подгруженные главы правил для текущей задачи:"]
    for rule in rules:
        sections.append(
            "\n".join(
                [
                    f"## {rule.title}",
                    f"rule_id: {rule.id}",
                    f"file: {rule.file}",
                    f"reason: {rule.reason}",
                    "",
                    rule.content,
                ]
            )
        )
    return "\n\n".join(sections)


def _rule_index_path(manifest: AgentManifest) -> Path | None:
    if not manifest.instructions_file:
        return None
    instructions_path = resolve_project_path(manifest.instructions_file)
    if instructions_path is None:
        return None
    return instructions_path.parent / "rule_index.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream) or {}
    except Exception as exc:
        logger.warning("Failed to load rule index %s: %s", path, exc)
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


def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
