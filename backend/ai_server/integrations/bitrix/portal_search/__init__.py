from __future__ import annotations

from ai_server.integrations.bitrix.portal_search.content_indexer import (
    sync_portal_content_index,
    sync_portal_content_item,
)
from ai_server.integrations.bitrix.portal_search.entity_syncs import (
    sync_catalog_product_item,
    sync_disk_file_item,
    sync_task_item,
)
from ai_server.integrations.bitrix.portal_search.file_cache import delete_portal_file_cache_path, portal_file_cache_path
from ai_server.integrations.bitrix.portal_search.formatting import (
    entity_types_for_scope,
    format_portal_content_sync_stats,
    format_portal_delta_sync_stats,
    format_portal_index_stats,
    format_portal_search_results,
    format_portal_sync_stats,
)
from ai_server.integrations.bitrix.portal_search.search_index import PortalSearchIndex
from ai_server.integrations.bitrix.portal_search.sync_manager import sync_disk_delta_index, sync_portal_index
from ai_server.integrations.bitrix.portal_search.types import (
    PortalContentReadiness,
    PortalContentSyncStats,
    PortalDeltaSyncStats,
    PortalIndexStats,
    PortalSearchResult,
    PortalSyncStats,
)

__all__ = [
    "PortalContentReadiness",
    "PortalContentSyncStats",
    "PortalDeltaSyncStats",
    "PortalIndexStats",
    "PortalSearchIndex",
    "PortalSearchResult",
    "PortalSyncStats",
    "delete_portal_file_cache_path",
    "entity_types_for_scope",
    "format_portal_content_sync_stats",
    "format_portal_delta_sync_stats",
    "format_portal_index_stats",
    "format_portal_search_results",
    "format_portal_sync_stats",
    "portal_file_cache_path",
    "sync_disk_delta_index",
    "sync_disk_file_item",
    "sync_task_item",
    "sync_catalog_product_item",
    "sync_portal_content_index",
    "sync_portal_content_item",
    "sync_portal_index",
]
