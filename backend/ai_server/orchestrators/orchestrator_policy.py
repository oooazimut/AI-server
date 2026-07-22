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
    if value.get("authority") != "internal_orchestrator" or not isinstance(value.get("rules"), list):
        raise RuntimeError("ORCHESTRATOR_BITRIX_POLICY_INVALID")
    return value


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
        "rules": selected,
    }


__all__ = ["bitrix_policy_pack", "selected_bitrix_policy"]
