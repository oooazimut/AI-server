"""Import JSONL chunks from var/kartoteka/data into PostgreSQL kartoteka.file_index.

Usage:
    uv run python scripts/kartoteka_import.py
    uv run python scripts/kartoteka_import.py --data-dir /custom/path/to/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ai_server.integrations.postgres.kartoteka_agent import PostgresKartotekaStore
from ai_server.settings import get_settings

_ACCESS_MAP: dict[str, str] = {
    "internal": "open",
    "public": "open",
    "open": "open",
    "restricted_review": "protected",
    "restricted": "protected",
    "protected": "protected",
    "closed": "protected",
}

_JSONL_FILES = [
    ("stage1_open_chunks.jsonl", "open"),
    ("stage1_protected_chunks.jsonl", "protected"),
]


def _map_access(raw: str, default: str) -> str:
    return _ACCESS_MAP.get(raw.strip().casefold(), default)


def _import_file(store: PostgresKartotekaStore, jsonl_path: Path, default_level: str) -> int:
    count = 0
    errors = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  WARN line {lineno}: JSON error — {exc}", file=sys.stderr)
                errors += 1
                continue

            chunk_id = str(chunk.get("chunkId") or "").strip()
            if not chunk_id:
                print(f"  WARN line {lineno}: missing chunkId, skipping", file=sys.stderr)
                errors += 1
                continue

            raw_access = str(chunk.get("access") or default_level)
            access_level = _map_access(raw_access, default_level)

            store.upsert_chunk(
                chunk_id=chunk_id,
                chunk_index=int(chunk.get("chunkIndex") or 0),
                document_id=str(chunk.get("documentId") or ""),
                relative_path=str(chunk.get("relativePath") or ""),
                filename=str(chunk.get("name") or ""),
                extension=str(chunk.get("extension") or ""),
                text=str(chunk.get("text") or ""),
                access_level=access_level,
                group_id=str(chunk.get("groupId") or ""),
                group_name=str(chunk.get("groupName") or ""),
                size_bytes=int(chunk.get("sizeBytes") or 0),
                modified_time=str(chunk.get("modifiedTime") or ""),
                indexed_at=str(chunk.get("indexedAt") or ""),
            )
            count += 1

    if errors:
        print(f"  {errors} lines skipped due to errors")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Kartoteka JSONL index into PostgreSQL")
    parser.add_argument("--data-dir", help="Path to data/ directory (overrides KARTOTEKA_DATA_DIR)")
    args = parser.parse_args()

    settings = get_settings()

    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
    elif settings.kartoteka_data_dir:
        data_dir = Path(settings.kartoteka_data_dir).expanduser().resolve()
    else:
        data_dir = (settings.var_dir / "kartoteka" / "data").resolve()

    print(f"Data dir: {data_dir}")

    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if not settings.database_url:
        print("ERROR: DATABASE_URL is not set", file=sys.stderr)
        sys.exit(1)

    store = PostgresKartotekaStore(
        settings.database_url,
        protected_user_ids=settings.kartoteka_protected_user_ids,
        secret_user_ids=settings.kartoteka_secret_user_ids,
    )

    total = 0
    for filename, default_level in _JSONL_FILES:
        jsonl_path = data_dir / "content_index" / filename
        if not jsonl_path.exists():
            print(f"SKIP (not found): {jsonl_path}")
            continue
        size_kb = jsonl_path.stat().st_size // 1024
        print(f"\nImporting {filename} ({size_kb} KB, default access={default_level}) ...")
        count = _import_file(store, jsonl_path, default_level)
        print(f"  → {count} chunks imported")
        total += count

    print(f"\nDone. Total: {total} chunks imported into kartoteka.file_index")


if __name__ == "__main__":
    main()
