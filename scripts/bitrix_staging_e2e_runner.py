from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from ai_server.settings import get_settings  # noqa: E402
from scripts.create_bitrix_dev_chat import env_file_paths, load_env_files  # noqa: E402

DEFAULT_EVENTS_URL = "http://127.0.0.1:8001/bitrix/events"
DEFAULT_STATUS_URL = "http://127.0.0.1:8001/bitrix/webhook-events/status"


@dataclass(frozen=True)
class TestCase:
    test_id: str
    text: str
    timeout_seconds: int = 180
    expect_any: tuple[str, ...] = ()
    reject_any: tuple[str, ...] = ()
    kind: str = "read"


SAFE_TESTS: dict[str, list[TestCase]] = {
    "smoke": [
        TestCase(
            test_id="BITRIX-SMOKE-01",
            text="Битрикс тест связи. Ответь коротко: test ok.",
            timeout_seconds=90,
            expect_any=("test ok",),
            kind="smoke",
        ),
    ],
    "tasks": [
        TestCase(
            test_id="BITRIX-TASK-SEARCH-01",
            text="Битрикс покажи мои задачи. Покажи не больше 10.",
            timeout_seconds=210,
            expect_any=("задач", "задачи", "нет"),
        ),
    ],
    "warehouse": [
        TestCase(
            test_id="BITRIX-WAREHOUSE-BORISOV-01",
            text="Битрикс покажи остатки по складу Борисов. Покажи не больше 10 позиций.",
            timeout_seconds=240,
            expect_any=("склад", "остат", "Борис"),
            reject_any=("количество не указано",),
        ),
    ],
    "project": [
        TestCase(
            test_id="BITRIX-PROJECT-HYPHEN-01",
            text="Битрикс найди проект Ларгус-2.",
            timeout_seconds=210,
            expect_any=("Ларгус", "проект"),
            reject_any=("не найден", "не нашел", "не нашёл"),
        ),
    ],
    "project_space": [
        TestCase(
            test_id="BITRIX-PROJECT-SPACE-01",
            text="Битрикс найди проект Ларгус 2.",
            timeout_seconds=210,
            expect_any=("Ларгус", "проект"),
            reject_any=("не найден", "не нашел", "не нашёл"),
        ),
    ],
}

STATEFUL_TESTS: dict[str, list[TestCase]] = {
    "draft": [
        TestCase(
            test_id="BITRIX-TASK-DRAFT-01",
            text=(
                "Битрикс создай задачу на меня: подготовить тестовый отчет. "
                "Срок без срока. Не создавай сразу, только покажи черновик для подтверждения."
            ),
            timeout_seconds=240,
            expect_any=("черновик", "подтверд", "задач"),
            kind="draft",
        ),
    ],
}


def request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    timeout: int = 30,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": "HTTP_ERROR", "status": exc.code, "body": raw[:1000]}
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "details": str(exc)[:1000]}


def post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return request_json(url, payload, headers=headers, method="POST")


def bitrix_rest(rest_webhook_url: str, method: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = rest_webhook_url.rstrip("/") + "/" + method + ".json"
    return post_json(url, payload)


def get_messages(rest_webhook_url: str, dialog_id: str) -> list[dict[str, Any]]:
    response = bitrix_rest(rest_webhook_url, "im.dialog.messages.get", {"DIALOG_ID": dialog_id, "LIMIT": 40})
    result = response.get("result")
    if isinstance(result, dict):
        messages = result.get("messages") or result.get("MESSAGES") or result.get("items") or []
    elif isinstance(result, list):
        messages = result
    else:
        messages = []
    if isinstance(messages, dict):
        messages = list(messages.values())
    return [item for item in messages if isinstance(item, dict)]


def message_id(message: dict[str, Any]) -> int | None:
    raw = message.get("id") or message.get("ID")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def message_text(message: dict[str, Any]) -> str:
    for key in ("text", "TEXT", "message", "MESSAGE"):
        value = message.get(key)
        if isinstance(value, str):
            return value
    return json.dumps(message, ensure_ascii=False, sort_keys=True)


def send_event(
    *,
    events_url: str,
    secret: str,
    run_id: str,
    test: TestCase,
    bot_id: int,
    chat_id: int,
    dialog_id: str,
    user_id: int,
) -> dict[str, Any]:
    synthetic_message_id = int(datetime.now().strftime("%Y%m%d%H%M%S%f")[:17])
    payload = {
        "event": "ONIMBOTV2MESSAGEADD",
        "data": {
            "bot": {"id": bot_id},
            "chat": {"id": chat_id, "dialogId": dialog_id},
            "message": {
                "id": synthetic_message_id,
                "authorId": user_id,
                "text": f"[{run_id} {test.test_id}] {test.text}",
            },
            "user": {"id": user_id},
        },
    }
    return post_json(events_url, payload, headers={"X-Agent-Secret": secret})


def run_test(args: argparse.Namespace, *, run_id: str, test: TestCase) -> dict[str, Any]:
    preflight_status = wait_queue_idle(args.status_url, timeout_seconds=args.preflight_idle_timeout)
    if not preflight_status.get("ok"):
        return {
            "test_id": test.test_id,
            "kind": test.kind,
            "ok": False,
            "stage": "preflight_queue_busy",
            "worker_latest": compact_status(preflight_status.get("status") or {}),
        }

    before_messages = get_messages(args.rest_webhook_url, args.dialog_id)
    before_ids = {item for item in (message_id(message) for message in before_messages) if item is not None}
    enqueue = send_event(
        events_url=args.events_url,
        secret=args.secret,
        run_id=run_id,
        test=test,
        bot_id=args.bot_id,
        chat_id=args.chat_id,
        dialog_id=args.dialog_id,
        user_id=args.user_id,
    )
    if not enqueue.get("ok"):
        return {"test_id": test.test_id, "ok": False, "stage": "enqueue", "enqueue": enqueue}
    event_id = enqueue.get("event_id")

    deadline = time.time() + test.timeout_seconds
    last_status: dict[str, Any] = {}
    last_messages: list[dict[str, Any]] = []
    last_new_messages: list[dict[str, Any]] = []
    event_was_processed = False
    while time.time() < deadline:
        time.sleep(args.poll_interval)
        last_status = request_json(args.status_url, timeout=10)
        event_was_processed = event_was_processed or event_processed(last_status, event_id)
        messages = get_messages(args.rest_webhook_url, args.dialog_id)
        last_messages = messages
        new_messages = [
            message for message in messages if message_id(message) is not None and message_id(message) not in before_ids
        ]
        if not new_messages:
            continue
        if not event_was_processed:
            continue
        last_new_messages = new_messages

        response_messages = matching_response_messages(new_messages, test)
        if not response_messages:
            continue

        joined = "\n".join(message_text(message) for message in response_messages)
        lower_joined = joined.casefold()
        matched = not test.expect_any or any(item.casefold() in lower_joined for item in test.expect_any)
        rejected = any(item.casefold() in lower_joined for item in test.reject_any)
        ok = matched and not rejected
        return {
            "test_id": test.test_id,
            "kind": test.kind,
            "ok": ok,
            "stage": "response",
            "event_id": event_id,
            "event_processed": event_was_processed,
            "new_message_ids": [message_id(message) for message in new_messages],
            "new_message_count": len(new_messages),
            "response_message_ids": [message_id(message) for message in response_messages],
            "expect_any": list(test.expect_any),
            "reject_any": list(test.reject_any),
            "matched": matched,
            "rejected": rejected,
            "response_preview": joined[:1200],
            "worker_latest": compact_status(last_status),
        }

    return {
        "test_id": test.test_id,
        "kind": test.kind,
        "ok": False,
        "stage": "timeout",
        "event_id": event_id,
        "event_processed": event_was_processed,
        "last_message_count": len(last_messages),
        "last_new_message_ids": [message_id(message) for message in last_new_messages],
        "last_new_response_preview": "\n".join(message_text(message) for message in last_new_messages)[:1200],
        "worker_latest": compact_status(last_status),
    }


def wait_queue_idle(status_url: str, *, timeout_seconds: float, poll_interval: float = 3.0) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last_status: dict[str, Any] = {}
    while time.time() < deadline:
        last_status = request_json(status_url, timeout=10)
        if queue_is_idle(last_status):
            return {"ok": True, "status": last_status}
        time.sleep(poll_interval)
    return {"ok": False, "status": last_status}


def queue_is_idle(status: dict[str, Any]) -> bool:
    queue = status.get("queue") if isinstance(status.get("queue"), dict) else {}
    return int(queue.get("pending") or 0) == 0 and int(queue.get("processing") or 0) == 0


def event_processed(status: dict[str, Any], event_id: object) -> bool:
    if event_id in (None, ""):
        return True
    normalized = str(event_id)
    latest = status.get("latest_events") if isinstance(status.get("latest_events"), list) else []
    for item in latest:
        if not isinstance(item, dict) or str(item.get("id") or "") != normalized:
            continue
        return str(item.get("status") or "").casefold() in {"done", "failed"}
    worker = status.get("worker") if isinstance(status.get("worker"), dict) else {}
    queue_idle = queue_is_idle(status)
    return queue_idle and str(worker.get("last_event_id") or "") == normalized


def matching_response_messages(messages: list[dict[str, Any]], test: TestCase) -> list[dict[str, Any]]:
    if not test.expect_any:
        return messages
    needles = [item.casefold() for item in test.expect_any]
    return [message for message in messages if any(needle in message_text(message).casefold() for needle in needles)]


def compact_status(status: dict[str, Any]) -> dict[str, Any]:
    worker = status.get("worker") if isinstance(status.get("worker"), dict) else {}
    queue = status.get("queue") if isinstance(status.get("queue"), dict) else {}
    latest = status.get("latest_events") if isinstance(status.get("latest_events"), list) else []
    latest_event = latest[0] if latest and isinstance(latest[0], dict) else {}
    return {
        "worker_running": worker.get("running"),
        "worker_processed": worker.get("processed"),
        "worker_errors": worker.get("errors"),
        "worker_last_event_id": worker.get("last_event_id"),
        "worker_last_error": worker.get("last_error"),
        "queue": {key: queue.get(key) for key in ("pending", "processing", "done", "failed", "dbsize")},
        "latest_event": {
            "id": latest_event.get("id"),
            "event_type": latest_event.get("event_type"),
            "status": latest_event.get("status"),
        },
    }


def resolve_status_url(events_url: str) -> str:
    parsed = urllib.parse.urlparse(events_url)
    if parsed.path.endswith("/bitrix/events"):
        path = parsed.path[: -len("/bitrix/events")] + "/bitrix/webhook-events/status"
        return urllib.parse.urlunparse(parsed._replace(path=path, query="", fragment=""))
    return DEFAULT_STATUS_URL


def chat_id_from_dialog(dialog_id: str) -> int:
    if dialog_id.startswith("chat") and dialog_id[4:].isdigit():
        return int(dialog_id[4:])
    return 0


def tests_for_suite(suite: str, *, include_draft: bool) -> list[TestCase]:
    selected: list[TestCase] = []
    if suite == "all":
        for tests in SAFE_TESTS.values():
            selected.extend(tests)
    else:
        selected.extend(SAFE_TESTS.get(suite, []))
    if include_draft:
        selected.extend(STATEFUL_TESTS["draft"])
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic Bitrix staging E2E checks through AI Dev.")
    parser.add_argument("--env-file", action="append", default=None)
    parser.add_argument("--events-url", default=os.getenv("BITRIX_E2E_EVENTS_URL", DEFAULT_EVENTS_URL))
    parser.add_argument("--status-url", default=os.getenv("BITRIX_E2E_STATUS_URL", ""))
    parser.add_argument("--rest-webhook-url", default=os.getenv("BITRIX_E2E_REST_WEBHOOK_URL", ""))
    parser.add_argument("--secret", default=os.getenv("BITRIX_E2E_WEBHOOK_SECRET", ""))
    parser.add_argument("--dialog-id", default=os.getenv("BITRIX_E2E_DIALOG_ID", "chat4321"))
    parser.add_argument("--chat-id", type=int, default=int(os.getenv("BITRIX_E2E_CHAT_ID", "0") or 0))
    parser.add_argument("--user-id", type=int, default=int(os.getenv("BITRIX_E2E_USER_ID", "13") or 13))
    parser.add_argument("--bot-id", type=int, default=int(os.getenv("BITRIX_E2E_BOT_ID", "0") or 0))
    parser.add_argument("--suite", choices=["all", *sorted(SAFE_TESTS)], default="all")
    parser.add_argument("--include-draft", action="store_true", help="Also run the stateful local draft test.")
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--preflight-idle-timeout", type=float, default=120.0)
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args()
    loaded_env_keys = load_env_files(env_file_paths(args.env_file))
    settings = get_settings()

    args.status_url = args.status_url or resolve_status_url(args.events_url)
    args.rest_webhook_url = args.rest_webhook_url or settings.bitrix_rest_webhook_url
    args.secret = args.secret or settings.webhook_secret
    args.bot_id = args.bot_id or settings.bitrix_bot_id
    args.chat_id = args.chat_id or chat_id_from_dialog(args.dialog_id)

    missing = [
        name
        for name, value in {
            "rest_webhook_url": args.rest_webhook_url,
            "secret": args.secret,
            "bot_id": args.bot_id,
            "chat_id": args.chat_id,
            "dialog_id": args.dialog_id,
            "user_id": args.user_id,
        }.items()
        if not value
    ]
    if missing:
        print(json.dumps({"ok": False, "error": "missing_config", "missing": missing}, ensure_ascii=False, indent=2))
        return 1

    run_id = "AI-TEST-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tests = tests_for_suite(args.suite, include_draft=args.include_draft)
    results = [run_test(args, run_id=run_id, test=test) for test in tests]
    payload = {
        "ok": all(item.get("ok") for item in results),
        "suite": args.suite,
        "include_draft": args.include_draft,
        "run_id": run_id,
        "events_url": args.events_url,
        "status_url": args.status_url,
        "rest_webhook_configured": bool(args.rest_webhook_url),
        "dialog_id": args.dialog_id,
        "user_id": args.user_id,
        "bot_id": args.bot_id,
        "loaded_env_keys": sorted({key for key in loaded_env_keys if key.startswith(("BITRIX_", "WEBHOOK_"))}),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
