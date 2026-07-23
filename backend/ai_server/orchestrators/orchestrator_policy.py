"""Load business policies owned by the internal orchestrator."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_POLICY_PATH = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "internal_orchestrator"
    / "policies"
    / "bitrix_commands.json"
)


@lru_cache(maxsize=1)
def bitrix_policy_pack() -> dict[str, Any]:
    value = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "orchestrator.bitrix_policy.v1":
        raise RuntimeError("ORCHESTRATOR_BITRIX_POLICY_VERSION_INVALID")
    if (
        value.get("authority") != "internal_orchestrator"
        or not isinstance(value.get("rules"), list)
        or not isinstance(value.get("defaults"), dict)
        or not isinstance(value.get("request_verbs"), dict)
        or not isinstance(value.get("templates"), dict)
    ):
        raise RuntimeError("ORCHESTRATOR_BITRIX_POLICY_INVALID")
    required_defaults = {
        "result_limit",
        "warehouse_page_size",
        "task_deadline_working_days",
        "task_deadline_hour",
        "calendar_start_working_days",
        "calendar_start_hour",
        "calendar_duration_minutes",
    }
    if set(value["defaults"]) != required_defaults or any(
        type(value["defaults"][key]) is not int or value["defaults"][key] < 1
        for key in required_defaults
    ):
        raise RuntimeError("ORCHESTRATOR_BITRIX_POLICY_DEFAULTS_INVALID")
    required_templates = {
        "task_description",
        "task_close_unconfirmed_item",
        "task_close_missing_fields",
    }
    if set(value["templates"]) != required_templates:
        raise RuntimeError("ORCHESTRATOR_BITRIX_POLICY_TEMPLATES_INVALID")
    required_verb_groups = {"search_or_show", "search_noun"}
    if set(value["request_verbs"]) != required_verb_groups or any(
        not isinstance(value["request_verbs"][key], list)
        or not value["request_verbs"][key]
        or any(not isinstance(item, str) or not item.strip() for item in value["request_verbs"][key])
        for key in required_verb_groups
    ):
        raise RuntimeError("ORCHESTRATOR_BITRIX_POLICY_VERBS_INVALID")
    return value


def bitrix_policy_defaults() -> dict[str, int]:
    return {key: int(value) for key, value in bitrix_policy_pack()["defaults"].items()}


def bitrix_policy_templates() -> dict[str, Any]:
    return dict(bitrix_policy_pack()["templates"])


def bitrix_policy_request_verbs() -> dict[str, list[str]]:
    return {
        key: [str(item) for item in values]
        for key, values in bitrix_policy_pack()["request_verbs"].items()
    }


def selected_bitrix_policy(request: str) -> dict[str, Any]:
    pack = bitrix_policy_pack()
    normalized = request.casefold().replace("ё", "е")
    selected = []
    for rule in pack["rules"]:
        markers = [str(item).casefold().replace("ё", "е") for item in rule.get("markers") or []]
        if not markers or any(marker in normalized for marker in markers):
            selected.append(rule)
    return {
        "schema_version": pack["schema_version"],
        "authority": pack["authority"],
        "defaults": dict(pack["defaults"]),
        "request_verbs": bitrix_policy_request_verbs(),
        "templates": dict(pack["templates"]),
        "rules": selected,
    }


__all__ = [
    "bitrix_policy_defaults",
    "bitrix_policy_pack",
    "bitrix_policy_request_verbs",
    "bitrix_policy_templates",
    "selected_bitrix_policy",
]
