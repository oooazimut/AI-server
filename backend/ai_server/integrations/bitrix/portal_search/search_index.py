from __future__ import annotations

from typing import Any, Protocol

from ai_server.integrations.bitrix.portal_search.text_utils import safe_int
from ai_server.integrations.bitrix.portal_search.types import (
    PortalContentReadiness,
    PortalIndexStats,
    PortalSearchResult,
)


class PortalSearchIndex(Protocol):
    def ensure_schema(self) -> Any: ...

    def upsert_item(
        self,
        *,
        entity_type: str,
        entity_id: object,
        title: str,
        body: str = "",
        url: str = "",
        metadata: dict[str, Any] | None = None,
        source_updated_at: str | None = None,
        preserve_content: bool = True,
    ) -> None: ...

    def delete_item(self, *, entity_type: str, entity_id: object) -> bool: ...

    def delete_stale_items(self, *, entity_types: set[str], seen_before: str) -> int: ...

    def search(
        self,
        query: str,
        *,
        entity_types: set[str] | None = None,
        limit: int = 10,
    ) -> list[PortalSearchResult]: ...

    def stats(self) -> PortalIndexStats: ...

    def get_item(self, *, entity_type: str, entity_id: object) -> PortalSearchResult | None: ...

    def item_snapshot(self, *, entity_type: str, entity_id: object) -> dict[str, Any] | None: ...

    def disk_delta_folder_candidates(
        self,
        *,
        cursor_type: str | None,
        cursor_id: str | None,
        limit: int,
    ) -> tuple[list[PortalSearchResult], str | None, str | None, bool]: ...

    def children_by_parent_id(self, parent_id: object) -> list[PortalSearchResult]: ...

    def content_candidates(self, *, limit: int) -> list[PortalSearchResult]: ...

    def content_readiness(self, *, allowed_extensions: set[str]) -> PortalContentReadiness: ...

    def update_item_body_metadata(
        self,
        *,
        entity_type: str,
        entity_id: object,
        body: str,
        metadata: dict[str, Any],
    ) -> None: ...

    def get_task_close_processing_state(self, *, task_id: object, state_key: str) -> dict[str, Any] | None: ...

    def upsert_task_close_processing_state(
        self,
        *,
        task_id: object,
        state_key: str,
        status: str,
        payload: dict[str, Any] | None = None,
        actor_user_id: int | None = None,
    ) -> None: ...

    def get_task_close_control_event(self, *, task_id: object, close_event_key: str) -> dict[str, Any] | None: ...

    def upsert_task_close_control_event(
        self,
        *,
        task_id: object,
        close_event_key: str,
        decision: str,
        reason: str = "",
        closed_at: str | None = None,
        responsible_id: int | None = None,
        closed_by_user_id: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None: ...


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


def _merge_content_metadata(
    *,
    base: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in existing.items():
        if key.startswith("content_"):
            merged[key] = value
    return merged


def _should_preserve_content(
    *,
    existing_metadata: dict[str, Any],
    new_metadata: dict[str, Any],
    existing_source_updated_at: object,
    new_source_updated_at: object,
) -> bool:
    if not existing_metadata.get("content_index_status"):
        return False
    existing_source = _to_str(existing_source_updated_at)
    new_source = _to_str(new_source_updated_at)
    if existing_source or new_source:
        return existing_source == new_source
    existing_size = safe_int(existing_metadata.get("size"))
    new_size = safe_int(new_metadata.get("size"))
    return existing_size is not None and existing_size == new_size


def _to_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)
