from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

SEARCH_STOP_WORDS = {
    "а",
    "в",
    "во",
    "все",
    "всем",
    "всех",
    "всю",
    "где",
    "для",
    "документ",
    "документы",
    "документа",
    "документов",
    "и",
    "или",
    "любые",
    "мне",
    "на",
    "найди",
    "найти",
    "папка",
    "папки",
    "папку",
    "по",
    "поищи",
    "покажи",
    "портал",
    "портале",
    "порталу",
    "проект",
    "проекта",
    "проектам",
    "проектами",
    "проектах",
    "проектов",
    "проекты",
    "список",
    "файл",
    "файлы",
    "файлов",
}


# ---------------------------------------------------------------------------
# Text cleaning and search normalization
# ---------------------------------------------------------------------------


def normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text).lower().replace("ё", "е"))


def clean_text(text: object) -> str:
    if text is None:
        return ""
    value = re.sub(r"\[/?[a-zA-Z0-9_]+(?:=[^\]]*)?\]", "", str(text))
    value = re.sub(r"<[^>]+>", "", value)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def make_snippet(body: str, *, query: str = "", max_length: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", body).strip()
    if len(cleaned) <= max_length:
        return cleaned

    terms = flatten_unique(query_term_groups(query)) if query else []
    lowered = cleaned.lower().replace("ё", "е")
    positions = [lowered.find(term) for term in terms if term and lowered.find(term) >= 0]
    if positions:
        position = min(positions)
        start = max(0, position - max_length // 3)
        end = min(len(cleaned), start + max_length)
        start = max(0, end - max_length)
        snippet = cleaned[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(cleaned):
            snippet = snippet.rstrip() + "..."
        return snippet

    return cleaned[: max_length - 3].rstrip() + "..."


def body_with_content(body: str, content_text: str) -> str:
    base_body = body.split("\n\nТекст файла:\n", 1)[0].strip()
    if base_body:
        return f"{base_body}\n\nТекст файла:\n{content_text}"
    return f"Текст файла:\n{content_text}"


def content_text_from_body(body: str) -> str:
    marker = "\n\nТекст файла:\n"
    if marker in body:
        return body.split(marker, 1)[1].strip()
    if body.startswith("Текст файла:\n"):
        return body.split("Текст файла:\n", 1)[1].strip()
    return ""


# ---------------------------------------------------------------------------
# Search query processing
# ---------------------------------------------------------------------------


def query_terms(query: str) -> list[str]:
    terms = re.findall(r"[\w#№.-]+", normalize_search_text(query), flags=re.UNICODE)
    return [term for term in terms if len(term) > 1 and term not in SEARCH_STOP_WORDS]


def query_term_groups(query: str) -> list[list[str]]:
    return [search_variants(term) for term in query_terms(query)]


def search_variants(term: str) -> list[str]:
    variants = [term]
    if len(term) > 4 and term[-1:] in {"а", "ы", "и", "у", "е", "о"}:
        stem = term[:-1]
        variants.extend([stem, stem + "а", stem + "ы", stem + "и", stem + "у", stem + "е"])
    if len(term) > 5 and term.endswith(("ам", "ям", "ах", "ях", "ой", "ей", "ом", "ем")):
        stem = term[:-2]
        variants.extend([stem, stem + "а", stem + "ы", stem + "и", stem + "у", stem + "е"])
    return flatten_unique([variants])


def flatten_unique(groups: list[list[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            cleaned = value.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)
    return result


# ---------------------------------------------------------------------------
# File/extension helpers
# ---------------------------------------------------------------------------


def file_extension(name: str) -> str:
    return Path(name).suffix.lower()


def normalize_extensions(extensions: set[str]) -> set[str]:
    return {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
        if extension
    }


# ---------------------------------------------------------------------------
# Generic data helpers
# ---------------------------------------------------------------------------


def safe_json(value: object) -> dict[str, Any]:
    if not value:
        return {}
    try:
        data = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def safe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def to_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def first(data: dict[str, Any], *keys: str) -> object | None:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def attachment_ids(value: object) -> list[int]:
    if value is None or value == "":
        return []
    raw_items = value if isinstance(value, list) else [value]
    ids = []
    for item in raw_items:
        normalized = str(item).strip().removeprefix("n")
        if normalized.isdigit():
            ids.append(int(normalized))
    return ids


def normalize_url(value: str | None) -> str:
    if not value:
        return ""
    return quote(value.strip(), safe=":/?#[]@!$&'*,;=%")
