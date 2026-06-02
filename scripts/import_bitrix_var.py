from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ai_server.runtime_migration import (
    DEFAULT_AI_SERVER_VAR,
    DEFAULT_BITRIX_AGENT_VAR,
    EXCLUDED_RUNTIME_ITEMS,
    migrate_var,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Import BitrixAIAgent runtime var into AI-server var.")
    parser.add_argument("--source", type=Path, default=DEFAULT_BITRIX_AGENT_VAR)
    parser.add_argument("--target", type=Path, default=DEFAULT_AI_SERVER_VAR)
    parser.add_argument(
        "--profile",
        choices=("cutover", "index_only", "attachments", "documents"),
        default="cutover",
    )
    parser.add_argument("--execute", action="store_true", help="Actually copy files. Without this flag the command prints a plan.")
    parser.add_argument("--no-backup", action="store_true", help="Do not move existing target files into var/legacy/backups first.")
    args = parser.parse_args()

    result = migrate_var(
        args.source,
        args.target,
        profile=args.profile,
        execute=args.execute,
        backup_existing=not args.no_backup,
    )
    print(json.dumps({"execute": args.execute, "excluded": EXCLUDED_RUNTIME_ITEMS, "items": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
