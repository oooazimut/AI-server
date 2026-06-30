"""Tools-layer interface for the local Bitrix portal search index.

Provides PortalSearchPort Protocol and re-exports formatting utilities so that
agents/bitrix24/ can import from tools/ instead of integrations/ directly.
"""

from __future__ import annotations

from typing import Any, Protocol

from ai_server.integrations.bitrix.portal_search.formatting import (
    entity_types_for_scope,
    format_portal_search_results,
)

__all__ = [
    "PortalSearchPort",
    "entity_types_for_scope",
    "format_portal_search_results",
]


class PortalSearchPort(Protocol):
    """Narrow port for agents: full-text search over the local SQLite index."""

    def search(
        self,
        query: str,
        *,
        entity_types: set[str] | None = None,
        limit: int = 10,
    ) -> list[Any]: ...

    def stats(self) -> Any: ...  # returns PortalIndexStats: .exists: bool, .path: Path
