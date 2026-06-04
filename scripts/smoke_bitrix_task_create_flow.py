from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from ai_server.channels.bitrix import BitrixWebhookProcessor
from ai_server.integrations.bitrix.oauth import BitrixOAuthService
from ai_server.retrieval import HybridKnowledgeRetriever
from ai_server.settings import get_settings
from scripts.create_bitrix_dev_chat import env_file_paths, load_env_files, sanitize_result


class SmokeEmbeddingProvider:
    name = "smoke_embeddings"

    def __init__(self, *, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> dict[int, float]:
        tokens = re.findall(r"[0-9a-zа-яё_\.]{2,}", text.casefold().replace("ё", "е"))
        counts = Counter(tokens)
        vector: dict[int, float] = {}
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            vector[index] = vector.get(index, 0.0) + (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {index: value / norm for index, value in vector.items()}


def bitrix_message_payload(
    *,
    text: str,
    chat_id: int,
    dialog_id: str,
    user_id: int,
    bot_id: int | None,
    message_id: int,
) -> dict:
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "data": {
            "bot": {"id": bot_id},
            "chat": {"id": chat_id, "dialogId": dialog_id},
            "message": {"id": message_id, "authorId": user_id, "text": text},
            "user": {"id": user_id},
        },
    }


async def run_flow(args: argparse.Namespace) -> dict:
    settings = get_settings()
    processor = BitrixWebhookProcessor(
        bitrix_oauth=BitrixOAuthService(),
        bitrix_retriever=HybridKnowledgeRetriever(embedding_provider=SmokeEmbeddingProvider()),
    )
    draft_result = await processor.process(
        bitrix_message_payload(
            text=args.text,
            chat_id=args.chat_id,
            dialog_id=args.dialog_id,
            user_id=args.user_id,
            bot_id=args.bot_id or settings.bitrix_bot_id,
            message_id=args.message_id,
        )
    )
    result = {
        "draft": draft_result,
        "confirmed": None,
    }
    if args.confirm:
        confirm_result = await processor.process(
            bitrix_message_payload(
                text="да",
                chat_id=args.chat_id,
                dialog_id=args.dialog_id,
                user_id=args.user_id,
                bot_id=args.bot_id or settings.bitrix_bot_id,
                message_id=args.message_id + 1,
            )
        )
        result["confirmed"] = confirm_result
    return sanitize_result(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test Bitrix task creation through the Bitrix channel processor.")
    parser.add_argument("--env-file", action="append", default=None)
    parser.add_argument("--oauth-db-path", type=Path, default=None)
    parser.add_argument("--chat-id", type=int, default=3955)
    parser.add_argument("--dialog-id", default="chat3955")
    parser.add_argument("--user-id", type=int, default=9)
    parser.add_argument("--bot-id", type=int, default=None)
    parser.add_argument("--message-id", type=int, default=900001)
    parser.add_argument("--text", default="Создай задачу на меня AI dev smoke test без срока")
    parser.add_argument("--confirm", action="store_true", help="Also send the confirmation turn and create a real Bitrix task.")
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()
    loaded_env_keys = load_env_files(env_file_paths(args.env_file))
    if args.oauth_db_path:
        import os

        os.environ["BITRIX_OAUTH_DB_PATH"] = str(args.oauth_db_path)
        loaded_env_keys.append("BITRIX_OAUTH_DB_PATH")
    if "AGENT_DRY_RUN" not in loaded_env_keys:
        import os

        os.environ["AGENT_DRY_RUN"] = "false"

    result = asyncio.run(run_flow(args))
    result["loaded_env_keys"] = sorted({key for key in loaded_env_keys if key.startswith(("BITRIX_", "AGENT_", "WEBHOOK_", "PUBLIC_BASE_URL"))})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
