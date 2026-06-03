from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from ai_server.runtime import runtime_paths


@dataclass(frozen=True)
class PortalIndexStats:
    total_items: int
    by_type: dict[str, int]
    content_by_status: dict[str, int]
    last_indexed_at: str | None
    path: Path
    exists: bool


@dataclass(frozen=True)
class PortalSearchResult:
    entity_type: str
    entity_id: str
    title: str
    body: str
    url: str
    score: int
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "title": self.title,
            "body": self.body,
            "url": self.url,
            "score": self.score,
            "metadata": self.metadata,
        }


class PortalSearchIndex:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else runtime_paths().search_index_db

    @property
    def exists(self) -> bool:
        return self.path.exists()

    def ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS portal_search_items (
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    url TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source_updated_at TEXT,
                    last_seen_at TEXT,
                    indexed_at TEXT NOT NULL,
                    PRIMARY KEY (entity_type, entity_id)
                )
                """
            )
            self._ensure_column(connection, "portal_search_items", "last_seen_at", "TEXT")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_portal_search_type ON portal_search_items(entity_type)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_portal_search_indexed ON portal_search_items(indexed_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_portal_search_seen ON portal_search_items(last_seen_at)")

    def search(
        self,
        query: str,
        *,
        entity_types: set[str] | None = None,
        limit: int = 10,
    ) -> list[PortalSearchResult]:
        self.ensure_schema()
        term_groups = _query_term_groups(query)
        terms = _flatten_unique(term_groups)
        if not term_groups:
            return []

        where = []
        params: list[object] = []
        for group in term_groups:
            group_where = []
            for term in group:
                group_where.append("search_text LIKE ? ESCAPE '\\'")
                params.append(f"%{_escape_like(term)}%")
            where.append("(" + " OR ".join(group_where) + ")")
        if entity_types:
            placeholders = ",".join("?" for _ in entity_types)
            where.append(f"entity_type IN ({placeholders})")
            params.extend(sorted(entity_types))

        sql = (
            "SELECT entity_type, entity_id, title, body, url, metadata_json, search_text "
            "FROM portal_search_items WHERE "
            + " AND ".join(where)
            + " LIMIT 500"
        )
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()

        normalized_query = _normalize_search_text(query)
        scored: list[PortalSearchResult] = []
        for row in rows:
            metadata = _safe_json(row["metadata_json"])
            score = _score_result(
                normalized_query,
                terms,
                title=_normalize_search_text(row["title"]),
                body=_normalize_search_text(row["body"]),
                search_text=row["search_text"],
            )
            scored.append(
                PortalSearchResult(
                    entity_type=row["entity_type"],
                    entity_id=row["entity_id"],
                    title=row["title"],
                    body=row["body"],
                    url=row["url"],
                    score=score,
                    metadata=metadata,
                )
            )
        return sorted(scored, key=lambda item: (-item.score, item.entity_type, item.title))[:limit]

    def stats(self) -> PortalIndexStats:
        if not self.path.exists():
            return PortalIndexStats(
                total_items=0,
                by_type={},
                content_by_status={},
                last_indexed_at=None,
                path=self.path,
                exists=False,
            )
        self.ensure_schema()
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM portal_search_items").fetchone()[0]
            by_type_rows = connection.execute(
                "SELECT entity_type, COUNT(*) AS count FROM portal_search_items GROUP BY entity_type"
            ).fetchall()
            last_indexed_at = connection.execute("SELECT MAX(indexed_at) FROM portal_search_items").fetchone()[0]
            content_rows = connection.execute(
                """
                SELECT metadata_json
                FROM portal_search_items
                WHERE entity_type IN ('disk_file', 'task_attachment')
                """
            ).fetchall()

        content_by_status: dict[str, int] = {}
        for row in content_rows:
            status = _safe_json(row["metadata_json"]).get("content_index_status")
            if not status:
                continue
            status_key = str(status)
            content_by_status[status_key] = content_by_status.get(status_key, 0) + 1
        return PortalIndexStats(
            total_items=int(total),
            by_type={str(row["entity_type"]): int(row["count"]) for row in by_type_rows},
            content_by_status=content_by_status,
            last_indexed_at=str(last_indexed_at) if last_indexed_at else None,
            path=self.path,
            exists=True,
        )

    def get_item(self, *, entity_type: str, entity_id: object) -> PortalSearchResult | None:
        self.ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT entity_type, entity_id, title, body, url, metadata_json
                FROM portal_search_items
                WHERE entity_type = ? AND entity_id = ?
                """,
                (entity_type, str(entity_id)),
            ).fetchone()
        if not row:
            return None
        return _row_to_search_result(row)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def format_portal_search_results(results: list[PortalSearchResult], *, query: str) -> str:
    if not results:
        return f"В индексе портала ничего не нашёл по запросу: {query}"

    lines = [f"Нашёл по порталу: {len(results)}"]
    for result in results:
        label = _entity_type_label(result.entity_type)
        title = f"[{result.title}]({result.url})" if result.url else result.title
        snippet = _make_snippet(result.body, query=query)
        lines.append(f"- {label}: {title}")
        if snippet:
            lines.append(f"  {snippet}")
    return "\n".join(lines)


def format_portal_index_stats(stats: PortalIndexStats) -> str:
    if not stats.exists:
        return f"Локальный индекс портала пока не найден: {stats.path}"
    lines = [
        f"В индексе портала: {stats.total_items} объектов.",
        f"Последнее обновление: {stats.last_indexed_at or 'нет данных'}",
    ]
    for entity_type, count in sorted(stats.by_type.items()):
        lines.append(f"- {_entity_type_label(entity_type)}: {count}")
    if stats.content_by_status:
        lines.append("")
        lines.append("Текст документов:")
        for status, count in sorted(stats.content_by_status.items()):
            lines.append(f"- {_content_status_label(status)}: {count}")
    return "\n".join(lines)


def entity_types_for_scope(scope: str) -> set[str] | None:
    normalized = scope.strip().lower()
    if normalized in {"", "all"}:
        return None
    return {
        "documents": {"disk_file", "task_attachment"},
        "files": {"disk_file", "disk_folder", "disk_storage", "task_attachment"},
        "tasks": {"task", "task_attachment"},
        "projects": {"project"},
    }.get(normalized)


def _row_to_search_result(row: sqlite3.Row) -> PortalSearchResult:
    return PortalSearchResult(
        entity_type=row["entity_type"],
        entity_id=row["entity_id"],
        title=row["title"],
        body=row["body"],
        url=row["url"],
        score=0,
        metadata=_safe_json(row["metadata_json"]),
    )


def _score_result(
    normalized_query: str,
    terms: list[str],
    *,
    title: str,
    body: str,
    search_text: str,
) -> int:
    score = 0
    if normalized_query and normalized_query in title:
        score += 80
    if normalized_query and normalized_query in body:
        score += 30
    for term in terms:
        if term in title:
            score += 12
        if term in body:
            score += 4
        if term in search_text:
            score += 1
    return score


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[\w#№.-]+", _normalize_search_text(query), flags=re.UNICODE)
    return [term for term in terms if len(term) > 1 and term not in SEARCH_STOP_WORDS]


def _query_term_groups(query: str) -> list[list[str]]:
    return [_search_variants(term) for term in _query_terms(query)]


def _search_variants(term: str) -> list[str]:
    variants = [term]
    if len(term) > 4 and term[-1:] in {"а", "ы", "и", "у", "е", "о"}:
        stem = term[:-1]
        variants.extend([stem, stem + "а", stem + "ы", stem + "и", stem + "у", stem + "е"])
    if len(term) > 5 and term.endswith(("ам", "ям", "ах", "ях", "ой", "ей", "ом", "ем")):
        stem = term[:-2]
        variants.extend([stem, stem + "а", stem + "ы", stem + "и", stem + "у", stem + "е"])
    return _flatten_unique([variants])


def _flatten_unique(groups: list[list[str]]) -> list[str]:
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


def _normalize_search_text(text: str) -> str:
    return re.sub(r"\s+", " ", _clean_text(text).lower().replace("ё", "е"))


def _clean_text(text: object) -> str:
    if text is None:
        return ""
    value = re.sub(r"\[/?[a-zA-Z0-9_]+(?:=[^\]]*)?\]", "", str(text))
    value = re.sub(r"<[^>]+>", "", value)
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _safe_json(value: str) -> dict[str, Any]:
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _make_snippet(body: str, *, query: str = "", max_length: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", body).strip()
    if len(cleaned) <= max_length:
        return cleaned

    terms = _flatten_unique(_query_term_groups(query)) if query else []
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


def _entity_type_label(entity_type: str) -> str:
    return {
        "task": "Задача",
        "task_attachment": "Вложение задачи",
        "project": "Проект",
        "disk_storage": "Хранилище",
        "disk_folder": "Папка",
        "disk_file": "Файл",
    }.get(entity_type, entity_type)


def _content_status_label(status: str) -> str:
    return {
        "none": "ещё не брал в обработку",
        "indexed": "текст извлечён",
        "unsupported": "формат пока не поддержан",
        "too_large": "слишком большой файл",
        "empty": "текст не найден",
        "failed": "ошибка извлечения",
        "no_download_url": "нет ссылки скачивания",
    }.get(status, status)


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

