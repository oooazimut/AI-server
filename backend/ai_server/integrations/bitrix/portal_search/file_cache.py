from __future__ import annotations

import re
from pathlib import Path

from ai_server.integrations.bitrix.portal_search.types import PortalSearchResult
from ai_server.settings import Settings


def portal_file_cache_path(item: PortalSearchResult, settings: Settings) -> Path:
    from ai_server.integrations.bitrix.portal_search.text_utils import file_extension

    extension = file_extension(item.title) or ".bin"
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{item.entity_type}_{item.entity_id}")
    return settings.search_content_storage_dir / item.entity_type / f"{safe_id}{extension}"


def delete_portal_file_cache_path(path: Path, settings: Settings) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return

    storage_root = settings.search_content_storage_dir.resolve()
    parent = path.parent
    while True:
        try:
            if parent.resolve() == storage_root:
                break
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
