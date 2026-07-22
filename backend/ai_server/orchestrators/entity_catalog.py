"""Orchestrator-owned, proactively refreshed Bitrix entity directory.

The directory is a semantic aid, not a source of object contents or access
rights. It is refreshed outside user requests. Bitrix tools still perform
current-user ACL/OAuth checks before returning facts or mutating anything.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime
from typing import Any

from ai_server.utils import MOSCOW_TZ


def normalize_entity_text(value: object) -> str:
    text = str(value or "").casefold().replace("\u0451", "\u0435")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\u0430-\u044f0-9]+", " ", text)).strip()


def _surname_stem(value: str) -> str:
    normalized = normalize_entity_text(value)
    token = normalized.split(" ")[0] if normalized else ""
    # Russian surname cases: Borisova/Borisovu/Borisovym -> Borisov.
    for suffix, replacement in (
        ("\u043e\u0432\u043e\u0439", "\u043e\u0432"),
        ("\u0435\u0432\u043e\u0439", "\u0435\u0432"),
        ("\u0438\u043d\u043e\u0439", "\u0438\u043d"),
        ("\u043e\u0432\u0430", "\u043e\u0432"),
        ("\u0435\u0432\u0430", "\u0435\u0432"),
        ("\u0438\u043d\u0430", "\u0438\u043d"),
        ("\u043e\u0433\u043e", ""),
        ("\u0435\u043c\u0443", ""),
        ("\u044b\u043c", ""),
        ("\u043e\u043c", ""),
        ("\u0443", ""),
        ("\u0430", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def _aliases(*values: object) -> list[str]:
    aliases: set[str] = set()
    for value in values:
        normalized = normalize_entity_text(value)
        if not normalized:
            continue
        aliases.add(normalized)
        stem = _surname_stem(normalized)
        if stem:
            aliases.add(stem)
    return sorted(aliases)


def _first(item: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _int(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _user_entry(item: dict[str, Any]) -> dict[str, Any] | None:
    entity_id = _int(_first(item, "ID", "id"))
    if entity_id is None:
        return None
    first_name = str(_first(item, "NAME", "name") or "").strip()
    last_name = str(_first(item, "LAST_NAME", "lastName", "last_name") or "").strip()
    second_name = str(_first(item, "SECOND_NAME", "secondName", "second_name") or "").strip()
    full_name = " ".join(part for part in (last_name, first_name, second_name) if part).strip()
    initials = "".join(f"{part[0]}." for part in (first_name, second_name) if part)
    short_name = f"{last_name} {initials}".strip()
    return {
        "id": entity_id,
        "name": full_name or short_name or f"User #{entity_id}",
        "aliases": _aliases(full_name, short_name, last_name, f"{first_name} {last_name}"),
    }


def _project_entry(item: dict[str, Any]) -> dict[str, Any] | None:
    entity_id = _int(_first(item, "ID", "id"))
    name = str(_first(item, "NAME", "name") or "").strip()
    if entity_id is None or not name:
        return None
    # Only globally visible/open projects belong in the shared semantic
    # directory. Private discovery remains fail-closed in Bitrix.
    opened = str(_first(item, "OPENED", "opened", "VISIBLE", "visible") or "").upper()
    if opened not in {"Y", "YES", "TRUE", "1"}:
        return None
    return {"id": entity_id, "name": name, "aliases": _aliases(name)}


def _warehouse_entry(item: dict[str, Any]) -> dict[str, Any] | None:
    entity_id = _int(_first(item, "ID", "id"))
    name = str(_first(item, "TITLE", "title", "NAME", "name") or "").strip()
    if entity_id is None or not name:
        return None
    address = str(_first(item, "ADDRESS", "address") or "").strip()
    return {"id": entity_id, "name": name, "address": address, "aliases": _aliases(name, address)}


class OrchestratorEntityCatalog:
    """In-memory versioned directory refreshed independently of chat traffic."""

    def __init__(
        self,
        bitrix: Any,
        *,
        refresh_interval_seconds: int = 900,
        user_limit: int = 2000,
        project_limit: int = 500,
        warehouse_limit: int = 1000,
    ) -> None:
        self._bitrix = bitrix
        self._refresh_interval_seconds = max(60, int(refresh_interval_seconds))
        self._user_limit = max(1, int(user_limit))
        self._project_limit = max(1, int(project_limit))
        self._warehouse_limit = max(1, int(warehouse_limit))
        self._lock = asyncio.Lock()
        self._snapshot: dict[str, Any] = {
            "schema_version": "orchestrator.entity_catalog.v1",
            "version": "missing",
            "updated_at": None,
            "status": "missing",
            "users": [],
            "projects": [],
            "warehouses": [],
        }

    async def refresh(self) -> dict[str, Any]:
        async with self._lock:
            try:
                users_raw, projects_raw, warehouses_raw = await asyncio.gather(
                    self._bitrix.list_all_users(limit=self._user_limit),
                    self._bitrix.search_projects("", limit=self._project_limit),
                    self._bitrix.list_catalog_stores(limit=self._warehouse_limit),
                )
                users = [entry for item in users_raw if (entry := _user_entry(item)) is not None]
                projects = [entry for item in projects_raw if (entry := _project_entry(item)) is not None]
                warehouses = [entry for item in warehouses_raw if (entry := _warehouse_entry(item)) is not None]
                payload = {
                    "schema_version": "orchestrator.entity_catalog.v1",
                    "updated_at": datetime.now(MOSCOW_TZ).isoformat(),
                    "status": "ready",
                    "users": sorted(users, key=lambda item: (item["name"].casefold(), item["id"])),
                    "projects": sorted(projects, key=lambda item: (item["name"].casefold(), item["id"])),
                    "warehouses": sorted(warehouses, key=lambda item: (item["name"].casefold(), item["id"])),
                }
                digest = hashlib.sha256(
                    json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                self._snapshot = {**payload, "version": digest}
            except Exception as exc:
                self._snapshot = {
                    **self._snapshot,
                    "status": "stale" if self._snapshot.get("updated_at") else "error",
                    "last_error": f"{type(exc).__name__}: {exc}",
                }
            return self.snapshot()

    async def run_periodic(self) -> None:
        while True:
            await asyncio.sleep(self._refresh_interval_seconds)
            await self.refresh()

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._snapshot, ensure_ascii=False))

    def view_for_request(self, request: str, *, per_type_limit: int = 30) -> dict[str, Any]:
        snapshot = self.snapshot()
        if snapshot.get("status") not in {"ready", "stale"}:
            return snapshot
        normalized = normalize_entity_text(request)
        tokens = {token for token in normalized.split() if len(token) >= 3}

        def score(item: dict[str, Any]) -> tuple[int, str, int]:
            aliases = [normalize_entity_text(value) for value in item.get("aliases") or []]
            matches = 0
            for token in tokens:
                stem = _surname_stem(token)
                if any(token in alias or (stem and stem in alias) for alias in aliases):
                    matches += 1
            return (-matches, str(item.get("name") or "").casefold(), int(item.get("id") or 0))

        result = {key: value for key, value in snapshot.items() if key not in {"users", "projects", "warehouses"}}
        for key in ("users", "projects", "warehouses"):
            items = list(snapshot.get(key) or [])
            ranked = sorted(items, key=score)
            matched = [item for item in ranked if score(item)[0] < 0]
            result[key] = (matched or ranked)[: max(1, per_type_limit)]
        return result


def resolve_entity(
    catalog: dict[str, Any],
    entity_type: str,
    value: object,
) -> tuple[dict[str, Any] | None, bool]:
    """Return (single match, ambiguous) from a local catalog view."""

    normalized = normalize_entity_text(value)
    if not normalized:
        return None, False
    stem = _surname_stem(normalized)
    exact: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for item in catalog.get(entity_type) or []:
        aliases = [normalize_entity_text(alias) for alias in item.get("aliases") or []]
        if normalized in aliases or (stem and stem in aliases):
            exact.append(item)
        elif any(normalized in alias or alias in normalized or (stem and stem in alias) for alias in aliases):
            partial.append(item)
    matches = exact or partial
    if len(matches) == 1:
        return matches[0], False
    return None, len(matches) > 1


def find_entities_in_text(
    catalog: dict[str, Any],
    entity_type: str,
    value: object,
) -> list[dict[str, Any]]:
    """Find catalog entities explicitly named in free text."""

    normalized = normalize_entity_text(value)
    tokens = normalized.split()
    matches: list[dict[str, Any]] = []
    for item in catalog.get(entity_type) or []:
        aliases = [normalize_entity_text(alias) for alias in item.get("aliases") or []]
        found = False
        for alias in aliases:
            if not alias:
                continue
            if " " in alias and re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", normalized):
                found = True
                break
            alias_stem = _surname_stem(alias)
            if any(token == alias or _surname_stem(token) == alias_stem for token in tokens):
                found = True
                break
        if found:
            matches.append(item)
    return matches


__all__ = [
    "OrchestratorEntityCatalog",
    "find_entities_in_text",
    "normalize_entity_text",
    "resolve_entity",
]
