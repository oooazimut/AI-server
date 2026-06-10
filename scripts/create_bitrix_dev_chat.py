from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
sys.path.insert(0, str(BACKEND_DIR))

from ai_server.integrations.bitrix.client import BitrixApiError, BitrixClient, BitrixConfigError
from ai_server.integrations.bitrix.oauth import BitrixOAuthError
from ai_server.settings import get_settings

DEFAULT_TITLE = "AI dev"
DEFAULT_USERS = "1,9"


def parse_user_ids(raw: str) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for item in raw.replace(";", ",").replace(" ", ",").split(","):
        value = item.strip()
        if not value:
            continue
        try:
            user_id = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid Bitrix user id: {value}") from None
        if user_id <= 0:
            raise argparse.ArgumentTypeError(f"Bitrix user id must be positive: {value}")
        if user_id not in seen:
            seen.add(user_id)
            result.append(user_id)
    if not result:
        raise argparse.ArgumentTypeError("At least one Bitrix user id is required")
    return result


def chat_reference(result: Any) -> dict[str, Any]:
    if isinstance(result, int):
        return {"chat_id": result, "dialog_id": f"chat{result}"}
    if not isinstance(result, dict):
        return {}

    nested_chat = result.get("chat")
    if isinstance(nested_chat, dict):
        reference = chat_reference(nested_chat)
        if reference:
            return reference
    recent_config = result.get("recentConfig")
    if isinstance(recent_config, dict):
        chat_id = recent_config.get("chatId") or recent_config.get("CHAT_ID") or recent_config.get("chat_id")
        if chat_id is not None:
            return {"chat_id": chat_id, "dialog_id": f"chat{chat_id}"}

    chat_id = result.get("chatId") or result.get("CHAT_ID") or result.get("chat_id") or result.get("id") or result.get("ID")
    dialog_id = result.get("dialogId") or result.get("DIALOG_ID") or result.get("dialog_id")
    reference: dict[str, Any] = {}
    if chat_id is not None:
        reference["chat_id"] = chat_id
    if dialog_id is not None:
        reference["dialog_id"] = dialog_id
    elif chat_id is not None:
        reference["dialog_id"] = f"chat{chat_id}"
    return reference


def sanitize_result(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_secret_key(str(key)):
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = sanitize_result(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_result(item) for item in value]
    return value


def env_file_paths(raw_values: list[str] | None) -> list[Path]:
    values = raw_values or [str(DEFAULT_ENV_FILE)]
    paths: list[Path] = []
    for raw in values:
        for item in raw.replace(";", ",").split(","):
            value = item.strip()
            if value:
                paths.append(Path(value))
    return paths


def load_env_files(paths: list[Path]) -> list[str]:
    protected_keys = set(os.environ)
    loaded: list[str] = []
    for path in paths:
        loaded.extend(load_env_file(path, protected_keys=protected_keys))
    return loaded


def load_env_file(path: Path, *, protected_keys: set[str] | None = None) -> list[str]:
    if not path.exists():
        return []

    protected_keys = protected_keys or set()
    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key not in protected_keys:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def dry_run_plan(
    *,
    title: str,
    user_ids: list[int],
    description: str,
    color: str,
    message: str,
    bot_id: int | None,
    owner_id: int | None,
    loaded_env_keys: list[str],
) -> dict[str, Any]:
    settings = get_settings()
    fields: dict[str, Any] = {
        "title": title.strip(),
        "color": color,
        "userIds": user_ids,
    }
    if description:
        fields["description"] = description
    if message:
        fields["message"] = message
    if owner_id is not None:
        fields["ownerId"] = owner_id

    payload: dict[str, Any] = {
        "botId": bot_id or settings.bitrix_bot_id or "<BITRIX_BOT_ID>",
        "botToken": "<redacted>",
        "fields": fields,
    }
    return {
        "execute": False,
        "method": "imbot.v2.Chat.add",
        "payload": payload,
        "configured": {
            "bitrix_rest_webhook_url": bool(settings.bitrix_rest_webhook_url),
            "bitrix_bot_id": bool(bot_id or settings.bitrix_bot_id),
            "bitrix_bot_token": bool(settings.bitrix_bot_token),
        },
        "loaded_env_keys": _safe_env_key_list(loaded_env_keys),
        "next_command": (
            f'uv run python scripts/create_bitrix_dev_chat.py --title "{title}" '
            f"--users {','.join(str(user_id) for user_id in user_ids)} --execute"
        ),
    }


async def create_chat(args: argparse.Namespace) -> dict[str, Any]:
    client = BitrixClient()
    result = await client.create_bot_chat(
        title=args.title,
        user_ids=args.user_ids,
        description=args.description,
        color=args.color,
        message=args.message,
        bot_id=args.bot_id,
        owner_id=args.owner_id,
    )
    return {
        "execute": True,
        "method": "imbot.v2.Chat.add",
        "title": args.title,
        "user_ids": args.user_ids,
        "chat": chat_reference(result),
        "result": sanitize_result(result),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a Bitrix24 dev chat for the AI-server bot.")
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--users", default=parse_user_ids(DEFAULT_USERS), type=parse_user_ids)
    parser.add_argument("--description", default="AI-server development chat")
    parser.add_argument("--color", default="mint")
    parser.add_argument("--message", default="AI-server dev-контур готов к проверке.")
    parser.add_argument("--bot-id", type=int, default=None)
    parser.add_argument("--owner-id", type=int, default=None)
    parser.add_argument(
        "--oauth-db-path",
        type=Path,
        default=None,
        help="Use a specific Bitrix OAuth SQLite DB for OAuth bot mode.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=None,
        help=(
            "Load local environment variables before calling Bitrix24. "
            "Can be passed multiple times or as a comma/semicolon-separated list. Defaults to .env."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Actually create the chat in Bitrix24.")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()
    loaded_env_keys = load_env_files(env_file_paths(args.env_file))
    if args.oauth_db_path:
        os.environ["BITRIX_OAUTH_DB_PATH"] = str(args.oauth_db_path)
        loaded_env_keys.append("BITRIX_OAUTH_DB_PATH")

    if not args.execute:
        data = dry_run_plan(
            title=args.title,
            user_ids=args.users,
            description=args.description,
            color=args.color,
            message=args.message,
            bot_id=args.bot_id,
            owner_id=args.owner_id,
            loaded_env_keys=loaded_env_keys,
        )
    else:
        args.user_ids = args.users
        try:
            data = asyncio.run(create_chat(args))
        except (BitrixApiError, BitrixConfigError, BitrixOAuthError) as exc:
            data = {
                "execute": True,
                "ok": False,
                "method": "imbot.v2.Chat.add",
                "error": type(exc).__name__,
                "message": str(exc),
                "loaded_env_keys": _safe_env_key_list(loaded_env_keys),
            }
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 1

    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def _safe_env_key_list(keys: list[str]) -> list[str]:
    allowed_prefixes = ("BITRIX_", "PUBLIC_BASE_URL", "WEBHOOK_", "AGENT_", "AI_SERVER_", "LLM_")
    return sorted({key for key in keys if key.startswith(allowed_prefixes)})


def _is_secret_key(key: str) -> bool:
    normalized = key.replace("_", "").replace("-", "").casefold()
    return normalized in {
        "token",
        "accesstoken",
        "refreshtoken",
        "bottoken",
        "clientsecret",
        "secret",
    }


if __name__ == "__main__":
    raise SystemExit(main())
