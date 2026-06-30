from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_server.settings import Settings, get_settings

_OPEN_INDEX = "stage1_open_chunks.jsonl"
_PROTECTED_INDEX = "stage1_protected_chunks.jsonl"
_ACCESS_LEVELS = {"open", "protected", "secret", "internal", "restricted_review"}
_ACCESS_LEVEL_ALIASES = {
    "open": "open",
    "internal": "open",
    "public": "open",
    "protected": "protected",
    "restricted": "protected",
    "restricted_review": "protected",
    "closed": "protected",
    "secret": "secret",
}


@dataclass(frozen=True)
class SecureOrgDataSearchResult:
    title: str
    path: str
    access_level: str
    snippet: str
    score: int
    source: str

    def as_dict(self, *, include_path: bool) -> dict[str, Any]:
        data = {
            "title": self.title,
            "access_level": self.access_level,
            "snippet": self.snippet,
            "score": self.score,
            "source": self.source,
        }
        if include_path:
            data["path"] = self.path
        return data


class SecureOrgDataStore:
    """Read-only adapter over the local KB metadata/index files.

    It does not decide whether data is sensitive by content. Access level comes
    from explicit metadata/index placement only.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        data_root: Path | str | None = None,
        metadata_dir: Path | str | None = None,
        index_dir: Path | str | None = None,
        protected_user_ids: str | None = None,
        secret_user_ids: str | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.db_path = self._settings.secure_org_data_db_path
        self.data_root = _optional_path(data_root if data_root is not None else self._settings.secure_org_data_root)
        self.metadata_dir = _optional_path(
            metadata_dir if metadata_dir is not None else self._settings.secure_org_data_metadata_dir
        )
        if index_dir is not None:
            configured_index_dir = index_dir
        elif metadata_dir is not None:
            configured_index_dir = ""
        else:
            configured_index_dir = self._settings.secure_org_data_index_dir
        self.index_dir = _optional_path(configured_index_dir) or (
            self.metadata_dir / "content_index" if self.metadata_dir is not None else None
        )
        self._protected_user_ids = _id_set(
            protected_user_ids
            if protected_user_ids is not None
            else self._settings.secure_org_data_protected_user_ids
        )
        self._secret_user_ids = _id_set(
            secret_user_ids if secret_user_ids is not None else self._settings.secure_org_data_secret_user_ids
        )
        self._access_overrides = self._load_access_overrides()

    def status(self) -> dict[str, Any]:
        index_files = self._index_files()
        return {
            "data_root": str(self.data_root) if self.data_root else "",
            "metadata_dir": str(self.metadata_dir) if self.metadata_dir else "",
            "index_dir": str(self.index_dir) if self.index_dir else "",
            "db_path": str(self.db_path),
            "index_exists": bool(self.index_dir and self.index_dir.exists()),
            "open_index_exists": index_files["open"].exists() if index_files.get("open") else False,
            "protected_index_exists": index_files["protected"].exists() if index_files.get("protected") else False,
        }

    def search(
        self,
        query: str,
        *,
        user_id: int | None = None,
        limit: int = 5,
        include_paths: bool = True,
    ) -> dict[str, Any]:
        query = query.strip()
        if not query:
            return {"configured": True, "query": query, "results": [], "error": "query is required"}
        if not self.index_dir:
            return {
                "configured": False,
                "query": query,
                "results": [],
                "error": "SECURE_ORG_DATA_INDEX_DIR or SECURE_ORG_DATA_METADATA_DIR is not configured",
            }

        index_files = self._index_files()
        if not index_files["open"].exists() and not index_files["protected"].exists():
            return {
                "configured": False,
                "query": query,
                "results": [],
                "status": self.status(),
                "error": "secure org data content index is missing",
            }

        limit = max(1, min(limit, 20))
        can_read_protected = self._can_read_protected(user_id)
        can_read_secret = self._can_read_secret(user_id)
        denied = {"protected": 0, "secret": 0}
        candidates: list[SecureOrgDataSearchResult] = []

        candidates.extend(
            self._search_index_file(
                index_files["open"],
                query=query,
                default_access_level="open",
                include_paths=include_paths,
                denied=denied,
                can_read_protected=can_read_protected,
                can_read_secret=can_read_secret,
            )
        )
        candidates.extend(
            self._search_index_file(
                index_files["protected"],
                query=query,
                default_access_level="protected",
                include_paths=include_paths,
                denied=denied,
                can_read_protected=can_read_protected,
                can_read_secret=can_read_secret,
            )
        )
        candidates.sort(key=lambda item: (-item.score, item.title.casefold(), item.path.casefold()))
        selected = candidates[:limit]
        return {
            "configured": True,
            "query": query,
            "limit": limit,
            "access": {
                "protected_allowed": can_read_protected,
                "secret_allowed": can_read_secret,
                "denied_counts": denied,
            },
            "results": [
                result.as_dict(include_path=include_paths and result.access_level != "secret") for result in selected
            ],
        }

    def _search_index_file(
        self,
        path: Path,
        *,
        query: str,
        default_access_level: str,
        include_paths: bool,
        denied: dict[str, int],
        can_read_protected: bool,
        can_read_secret: bool,
    ) -> list[SecureOrgDataSearchResult]:
        if not path.exists():
            return []
        results: list[SecureOrgDataSearchResult] = []
        tokens = _tokens(query)
        query_norm = _norm(query)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                haystack = _norm(" ".join(_strings(item)))
                score = _score(haystack, query_norm=query_norm, tokens=tokens)
                if score <= 0:
                    continue
                item_path = _path_from_item(item)
                item_access_level = _access_level_from_item(item)
                access_level = self._access_level_for_path(
                    item_path,
                    default_access_level=item_access_level or default_access_level,
                )
                if access_level == "secret" and not can_read_secret:
                    denied["secret"] += 1
                    continue
                if access_level == "protected" and not can_read_protected:
                    denied["protected"] += 1
                    continue
                content = _content_from_item(item)
                results.append(
                    SecureOrgDataSearchResult(
                        title=_title_from_item(item, item_path),
                        path=item_path if include_paths else "",
                        access_level=access_level,
                        snippet=_snippet(content or haystack, query_norm=query_norm, tokens=tokens),
                        score=score,
                        source=path.name,
                    )
                )
        return results

    def _index_files(self) -> dict[str, Path]:
        base = self.index_dir or Path()
        return {
            "open": base / _OPEN_INDEX,
            "protected": base / _PROTECTED_INDEX,
        }

    def _load_access_overrides(self) -> dict[str, str]:
        overrides: dict[str, str] = {}
        if not self.metadata_dir:
            return overrides
        for filename in ("file_access_overrides.json", "folder_roles.json"):
            path = self.metadata_dir / filename
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                continue
            _collect_access_levels(payload, overrides)
        return overrides

    def _access_level_for_path(self, path: str, *, default_access_level: str) -> str:
        normalized_path = _path_key(path)
        best_match = ""
        best_level = ""
        for key, level in self._access_overrides.items():
            normalized_key = _path_key(key)
            matches = normalized_path == normalized_key or normalized_path.startswith(
                normalized_key.rstrip("/\\") + "/"
            )
            if matches and len(normalized_key) > len(best_match):
                best_match = normalized_key
                best_level = level
        return best_level or default_access_level

    def _can_read_protected(self, user_id: int | None) -> bool:
        return "*" in self._protected_user_ids or (user_id is not None and user_id in self._protected_user_ids)

    def _can_read_secret(self, user_id: int | None) -> bool:
        return "*" in self._secret_user_ids or (user_id is not None and user_id in self._secret_user_ids)


def _optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    return Path(raw).expanduser() if raw else None


def _id_set(raw: str | None) -> set[int | str]:
    result: set[int | str] = set()
    for item in str(raw or "").replace(";", ",").split(","):
        value = item.strip()
        if not value:
            continue
        if value == "*":
            result.add("*")
            continue
        try:
            result.add(int(value))
        except ValueError:
            continue
    return result


def _strings(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            found.extend(_strings(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_strings(item))
    elif isinstance(value, (str, int, float)):
        text = str(value)
        if text:
            found.append(text)
    return found


def _content_from_item(item: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk", "snippet", "body"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return " ".join(_strings(item))


def _path_from_item(item: dict[str, Any]) -> str:
    for key in ("path", "file_path", "source_path", "relativePath", "relative_path", "document_path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _title_from_item(item: dict[str, Any], path: str) -> str:
    for key in ("title", "name", "fileName", "file_name", "filename"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return Path(path).name if path else "Найденный фрагмент"


def _tokens(value: str) -> list[str]:
    return [token for token in re.findall(r"[0-9a-zа-яё_\\.-]{2,}", _norm(value)) if token]


def _norm(value: str) -> str:
    return value.casefold().replace("ё", "е")


def _score(haystack: str, *, query_norm: str, tokens: list[str]) -> int:
    score = 0
    if query_norm and query_norm in haystack:
        score += 10
    for token in tokens:
        if token in haystack:
            score += 1 + min(haystack.count(token), 5)
    return score


def _snippet(content: str, *, query_norm: str, tokens: list[str], max_chars: int = 320) -> str:
    if not content:
        return ""
    normalized = _norm(content)
    positions = [normalized.find(query_norm)] if query_norm else []
    positions.extend(normalized.find(token) for token in tokens)
    positions = [pos for pos in positions if pos >= 0]
    start = max(0, min(positions) - 80) if positions else 0
    snippet = content[start : start + max_chars].strip()
    if start > 0:
        snippet = "..." + snippet
    if start + max_chars < len(content):
        snippet += "..."
    return snippet


def _path_key(value: str) -> str:
    return str(value or "").strip().replace("\\", "/").casefold()


def _collect_access_levels(value: Any, output: dict[str, str]) -> None:
    if isinstance(value, dict):
        level = _extract_access_level(value)
        path = _extract_path(value)
        if path and level:
            output[path] = level
        for key, item in value.items():
            if isinstance(item, str) and _normalize_access_level(item):
                output[str(key)] = _normalize_access_level(item)
            else:
                _collect_access_levels(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_access_levels(item, output)


def _extract_access_level(value: dict[str, Any]) -> str:
    for key in ("access", "accessLevel", "access_level", "groupAccess", "role", "visibility"):
        item = value.get(key)
        if isinstance(item, str):
            normalized = _normalize_access_level(item)
            if normalized:
                return normalized
    return ""


def _extract_path(value: dict[str, Any]) -> str:
    for key in ("path", "file_path", "folder_path", "relativePath", "relative_path"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return ""


def _access_level_from_item(item: dict[str, Any]) -> str:
    return _extract_access_level(item)


def _normalize_access_level(value: str) -> str:
    return _ACCESS_LEVEL_ALIASES.get(value.strip().casefold(), "")
