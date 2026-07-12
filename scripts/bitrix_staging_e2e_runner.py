from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_DIR))

from ai_server.integrations.bitrix.chat_parser import make_dialog_key  # noqa: E402
from ai_server.integrations.postgres.bitrix_agent import PostgresBitrixAgentStore  # noqa: E402
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
    expect_all: tuple[str, ...] = ()
    reject_any: tuple[str, ...] = ()
    kind: str = "read"
    required_task_id_arg: str = ""


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

OPTIONAL_READ_TESTS: dict[str, list[TestCase]] = {
    "tasks_advanced": [
        TestCase(
            test_id="BITRIX-TASK-ADVANCED-COMMENT-01",
            text=("Битрикс найди закрытые задачи, где в комментариях есть слово понаблюдать. Покажи не больше 5."),
            timeout_seconds=300,
            expect_all=("задач",),
            expect_any=("понаблюдать", "не найден", "нет"),
            reject_any=("вы добавлены наблюдателем", "вы назначены исполнителем"),
        ),
    ],
}

STATEFUL_TESTS: dict[str, list[TestCase]] = {
    "drafts": [
        TestCase(
            test_id="BITRIX-TASK-DRAFT-01",
            text=(
                "Битрикс создай задачу на меня: подготовить тестовый отчет. "
                "Не создавай сразу, только покажи черновик для подтверждения."
            ),
            timeout_seconds=240,
            expect_all=("черновик", "задач", "срок", "если всё верно"),
            reject_any=("Срок: Без срока", "/company/personal/user/"),
            kind="draft",
        ),
        TestCase(
            test_id="BITRIX-TASK-DRAFT-DISCARD-01",
            text="Битрикс отмени черновик задачи.",
            timeout_seconds=180,
            expect_all=("черновик", "задач", "удал"),
            kind="draft_cleanup",
        ),
        TestCase(
            test_id="BITRIX-CALENDAR-DRAFT-01",
            text=(
                "Битрикс напомни мне завтра позвонить Борисову. "
                "Не добавляй сразу, только покажи черновик календаря для подтверждения."
            ),
            timeout_seconds=240,
            expect_all=("черновик", "календар", "позвонить", "если всё верно"),
            reject_any=("/company/personal/user/",),
            kind="draft",
        ),
        TestCase(
            test_id="BITRIX-CALENDAR-DRAFT-DISCARD-01",
            text="Битрикс отмени черновик события календаря.",
            timeout_seconds=180,
            expect_all=("черновик", "событ", "календар", "удал"),
            kind="draft_cleanup",
        ),
    ],
    "drafts_project": [
        TestCase(
            test_id="BITRIX-TASK-PROJECT-DRAFT-01",
            text=(
                "Битрикс создай задачу в проекте Ларгус-2 на меня: проверить документы. "
                "Не создавай сразу, только покажи черновик для подтверждения."
            ),
            timeout_seconds=300,
            expect_all=("черновик", "задач", "проект", "Ларгус", "срок", "если всё верно"),
            reject_any=("Срок: Без срока", "/company/personal/user/"),
            kind="draft",
        ),
        TestCase(
            test_id="BITRIX-TASK-PROJECT-DRAFT-DISCARD-01",
            text="Битрикс отмени черновик задачи.",
            timeout_seconds=180,
            expect_all=("черновик", "задач", "удал"),
            kind="draft_cleanup",
        ),
    ],
    "task_close": [
        TestCase(
            test_id="BITRIX-TASK-CLOSE-DRAFT-01",
            text="Битрикс закрой задачу {task_close_task_id}.",
            timeout_seconds=300,
            expect_all=("Черновик закрытия задачи", "Пункты задачи", "Оборудование", "Итог", "Действия"),
            reject_any=("Задача закрыта", "Задача отмечена выполненной"),
            kind="task_close_draft",
            required_task_id_arg="task_close_task_id",
        ),
        TestCase(
            test_id="BITRIX-TASK-CLOSE-DISCARD-01",
            text="Битрикс отмени черновик закрытия задачи.",
            timeout_seconds=180,
            expect_all=("Черновик закрытия задачи", "удал"),
            kind="task_close_cleanup",
            required_task_id_arg="task_close_task_id",
        ),
    ],
}
STATEFUL_TESTS["draft"] = STATEFUL_TESTS["drafts"]

READ_ONLY_SUITE_ALIASES: dict[str, tuple[str, ...]] = {
    "quick": ("smoke", "project"),
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


def create_task_close_test_task(args: argparse.Namespace, *, run_id: str) -> dict[str, Any]:
    title = f"AI Dev task close E2E {run_id}"
    description = (
        "1. Проверить тестовый сценарий закрытия задачи.\n"
        "2. Зафиксировать, что бот не закрывает задачу без результата работ."
    )
    response = bitrix_rest(
        args.rest_webhook_url,
        "tasks.task.add",
        {
            "fields": {
                "TITLE": title,
                "DESCRIPTION": description,
                "RESPONSIBLE_ID": args.user_id,
            }
        },
    )
    task_id = extract_task_id(response)
    return {
        "ok": task_id is not None,
        "task_id": task_id,
        "title": title,
        "response": compact_rest_response(response),
    }


def delete_task_close_test_task(args: argparse.Namespace) -> dict[str, Any]:
    task_id = optional_int(getattr(args, "created_task_close_task_id", None))
    if task_id is None:
        return {"ok": True, "skipped": True}
    response = bitrix_rest(args.rest_webhook_url, "tasks.task.delete", {"taskId": task_id})
    return {
        "ok": bool(response.get("result") is not None or response.get("ok") is not False),
        "task_id": task_id,
        "response": compact_rest_response(response),
    }


def get_task_status(args: argparse.Namespace, task_id: object) -> dict[str, Any]:
    task_id_int = optional_int(task_id)
    if task_id_int is None:
        return {"ok": False, "error": "missing_task_id"}
    response = bitrix_rest(
        args.rest_webhook_url,
        "tasks.task.get",
        {"taskId": task_id_int, "select": ["ID", "TITLE", "STATUS", "CLOSED_DATE"]},
    )
    task = extract_task(response)
    status = optional_int((task or {}).get("status") or (task or {}).get("STATUS"))
    return {
        "ok": task is not None,
        "task_id": task_id_int,
        "status": status,
        "is_closed": status == 5 or bool((task or {}).get("closedDate") or (task or {}).get("CLOSED_DATE")),
        "response": compact_rest_response(response),
    }


def extract_task_id(response: dict[str, Any]) -> int | None:
    task = extract_task(response)
    if task:
        return optional_int(task.get("id") or task.get("ID"))
    result = response.get("result")
    if isinstance(result, dict):
        return optional_int(result.get("taskId") or result.get("TASK_ID") or result.get("id") or result.get("ID"))
    return optional_int(result)


def extract_task(response: dict[str, Any]) -> dict[str, Any] | None:
    result = response.get("result")
    if isinstance(result, dict):
        for key in ("task", "TASK", "item", "ITEM"):
            value = result.get(key)
            if isinstance(value, dict):
                return value
        if "id" in result or "ID" in result:
            return result
    return None


def optional_int(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def compact_rest_response(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("ok") is False:
        return {key: response.get(key) for key in ("ok", "error", "status", "details", "body")}
    result = response.get("result")
    if isinstance(result, dict):
        return {"result_keys": sorted(str(key) for key in result)[:20]}
    return {"has_result": "result" in response, "result_type": type(result).__name__}


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
    text: str,
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
                "text": f"[{run_id} {test.test_id}] {text}",
            },
            "user": {"id": user_id},
        },
    }
    return post_json(events_url, payload, headers={"X-Agent-Secret": secret})


def run_test(args: argparse.Namespace, *, run_id: str, test: TestCase) -> dict[str, Any]:
    missing_arg = missing_required_test_arg(args, test)
    if missing_arg:
        return {"test_id": test.test_id, "kind": test.kind, "ok": False, "stage": "missing_arg", "missing": missing_arg}

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
    text = render_test_text(test, args=args)
    enqueue = send_event(
        events_url=args.events_url,
        secret=args.secret,
        run_id=run_id,
        test=test,
        text=text,
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
        evaluation = evaluate_response_text(joined, test)
        draft_state = verify_draft_state(args, test)
        task_state = verify_task_state(args, test)
        ok = (
            evaluation["matched"]
            and not evaluation["rejected"]
            and draft_state.get("ok", True)
            and task_state.get("ok", True)
        )
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
            "expect_all": list(test.expect_all),
            "reject_any": list(test.reject_any),
            **evaluation,
            "draft_state": draft_state,
            "task_state": task_state,
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
    expectations = (*test.expect_any, *test.expect_all)
    if not expectations:
        return messages
    needles = [item.casefold() for item in expectations]
    return [message for message in messages if any(needle in message_text(message).casefold() for needle in needles)]


def evaluate_response_text(text: str, test: TestCase) -> dict[str, bool]:
    lower_text = text.casefold()
    matched_any = not test.expect_any or any(item.casefold() in lower_text for item in test.expect_any)
    matched_all = all(item.casefold() in lower_text for item in test.expect_all)
    rejected = any(item.casefold() in lower_text for item in test.reject_any)
    return {
        "matched": matched_any and matched_all,
        "matched_any": matched_any,
        "matched_all": matched_all,
        "rejected": rejected,
    }


def render_test_text(test: TestCase, *, args: argparse.Namespace) -> str:
    values = {
        "task_close_task_id": getattr(args, "task_close_task_id", ""),
    }
    try:
        return test.text.format(**values)
    except KeyError:
        return test.text


def missing_required_test_arg(args: argparse.Namespace, test: TestCase) -> str:
    if not test.required_task_id_arg:
        return ""
    value = getattr(args, test.required_task_id_arg, "")
    return "" if optional_int(value) is not None else test.required_task_id_arg


def expected_pending_draft(test: TestCase) -> bool | None:
    if test.kind in {"draft", "task_close_draft"}:
        return True
    if test.kind in {"draft_cleanup", "task_close_cleanup"}:
        return False
    return None


def expected_draft_type(test: TestCase) -> str:
    if test.kind in {"task_close_draft", "task_close_cleanup"}:
        return "task_close"
    if test.kind in {"draft", "draft_cleanup"}:
        return ""
    return ""


def verify_draft_state(args: argparse.Namespace, test: TestCase) -> dict[str, Any]:
    expected = expected_pending_draft(test)
    if expected is None or not getattr(args, "verify_draft_store", True):
        return {"checked": False, "ok": True}
    database_url = str(getattr(args, "database_url", "") or "")
    dialog_key = str(getattr(args, "dialog_key", "") or "")
    if not database_url:
        return {"checked": True, "ok": False, "error": "missing_database_url", "expected_present": expected}
    if not dialog_key:
        return {"checked": True, "ok": False, "error": "missing_dialog_key", "expected_present": expected}
    try:
        draft = asyncio.run(load_pending_task_draft(database_url, dialog_key))
    except Exception as exc:
        return {
            "checked": True,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "expected_present": expected,
            "dialog_key": dialog_key,
        }
    present = bool(draft)
    expected_type = expected_draft_type(test)
    draft_type = draft.get("_draft_type") if isinstance(draft, dict) else None
    draft_task_id = optional_int(draft.get("task_id")) if isinstance(draft, dict) else None
    expected_task_id = (
        optional_int(getattr(args, "task_close_task_id", None)) if expected_type == "task_close" else None
    )
    type_ok = not expected or not expected_type or draft_type == expected_type
    task_ok = not expected or expected_task_id is None or draft_task_id == expected_task_id
    return {
        "checked": True,
        "ok": present is expected and type_ok and task_ok,
        "expected_present": expected,
        "present": present,
        "dialog_key": dialog_key,
        "expected_type": expected_type,
        "draft_type": draft_type,
        "expected_task_id": expected_task_id,
        "draft_task_id": draft_task_id,
        "has_fields": isinstance(draft, dict) and isinstance(draft.get("fields"), dict),
    }


def verify_task_state(args: argparse.Namespace, test: TestCase) -> dict[str, Any]:
    if test.kind != "task_close_draft":
        return {"checked": False, "ok": True}
    status = get_task_status(args, getattr(args, "task_close_task_id", ""))
    if not status.get("ok"):
        return {"checked": True, "ok": False, **status}
    return {
        "checked": True,
        "ok": not status.get("is_closed"),
        "expected_open": True,
        **status,
    }


async def load_pending_task_draft(database_url: str, dialog_key: str) -> dict[str, Any] | None:
    store = PostgresBitrixAgentStore(database_url)
    return await store.get_task_draft(dialog_key)


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


def make_run_id() -> str:
    return f"AI-TEST-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def default_lock_path(dialog_id: str) -> str:
    safe_dialog = "".join(char if char.isalnum() else "-" for char in dialog_id).strip("-") or "default"
    return str(Path(tempfile.gettempdir()) / f"ai-server-bitrix-e2e-{safe_dialog}.lock")


def acquire_dialog_lock(
    lock_path: str,
    *,
    timeout_seconds: float,
    stale_seconds: float,
    poll_interval: float = 1.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if stale_seconds > 0:
                try:
                    if time.time() - path.stat().st_mtime > stale_seconds:
                        path.unlink()
                        continue
                except OSError:
                    pass
            if time.time() >= deadline:
                return {"ok": False, "error": "dialog_lock_timeout", "lock_path": str(path)}
            time.sleep(poll_interval)
            continue
        except OSError as exc:
            return {"ok": False, "error": type(exc).__name__, "details": str(exc), "lock_path": str(path)}

        payload = {
            "pid": os.getpid(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        return {"ok": True, "fd": fd, "lock_path": str(path)}


def release_dialog_lock(lock: dict[str, Any] | None) -> None:
    if not lock or not lock.get("ok"):
        return
    fd = lock.get("fd")
    if isinstance(fd, int):
        try:
            os.close(fd)
        except OSError:
            pass
    lock_path = lock.get("lock_path")
    if isinstance(lock_path, str):
        try:
            Path(lock_path).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def tests_for_suite(suite: str, *, include_draft: bool) -> list[TestCase]:
    selected: list[TestCase] = []
    if suite == "release":
        for tests in SAFE_TESTS.values():
            selected.extend(tests)
        for tests in OPTIONAL_READ_TESTS.values():
            selected.extend(tests)
        selected.extend(STATEFUL_TESTS["drafts"])
        selected.extend(STATEFUL_TESTS["drafts_project"])
    elif suite == "all":
        for tests in SAFE_TESTS.values():
            selected.extend(tests)
    elif suite in READ_ONLY_SUITE_ALIASES:
        for aliased_suite in READ_ONLY_SUITE_ALIASES[suite]:
            selected.extend(SAFE_TESTS[aliased_suite])
    elif suite in OPTIONAL_READ_TESTS:
        selected.extend(OPTIONAL_READ_TESTS[suite])
    elif suite in STATEFUL_TESTS:
        selected.extend(STATEFUL_TESTS[suite])
    else:
        selected.extend(SAFE_TESTS.get(suite, []))
    if include_draft and suite != "release":
        selected.extend(STATEFUL_TESTS["drafts"])
    return selected


def suite_needs_task_close_task(tests: list[TestCase]) -> bool:
    return any(test.required_task_id_arg == "task_close_task_id" for test in tests)


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
    parser.add_argument(
        "--suite",
        choices=[
            "all",
            "release",
            *sorted(READ_ONLY_SUITE_ALIASES),
            *sorted(SAFE_TESTS),
            *sorted(OPTIONAL_READ_TESTS),
            *sorted(STATEFUL_TESTS),
        ],
        default="all",
    )
    parser.add_argument(
        "--include-draft",
        action="store_true",
        help="Also run the stateful local draft checks after the selected safe suite.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run remaining tests after a failure. By default the suite stops to avoid delayed-response mixups.",
    )
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--preflight-idle-timeout", type=float, default=120.0)
    parser.add_argument(
        "--verify-draft-store",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For stateful draft suites, verify pending draft presence in PostgreSQL.",
    )
    parser.add_argument(
        "--task-close-task-id",
        default=os.getenv("BITRIX_E2E_TASK_CLOSE_TASK_ID", ""),
        help="Existing staging task id for --suite task_close. If omitted, a temporary task is created and deleted.",
    )
    parser.add_argument(
        "--keep-created-task",
        action="store_true",
        help="Do not delete the temporary task created for --suite task_close.",
    )
    parser.add_argument(
        "--lock-path",
        default=os.getenv("BITRIX_E2E_LOCK_PATH", ""),
        help="Dialog-level lock path. Defaults to a temp file derived from dialog-id.",
    )
    parser.add_argument("--lock-timeout", type=float, default=float(os.getenv("BITRIX_E2E_LOCK_TIMEOUT", "600") or 600))
    parser.add_argument(
        "--stale-lock-seconds",
        type=float,
        default=float(os.getenv("BITRIX_E2E_STALE_LOCK_SECONDS", "1800") or 1800),
    )
    parser.add_argument("--disable-lock", action="store_true")
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
    args.database_url = settings.database_url
    args.dialog_key = make_dialog_key(chat_id=args.chat_id, dialog_id=args.dialog_id, user_id=args.user_id)

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

    args.lock_path = args.lock_path or default_lock_path(args.dialog_id)
    lock: dict[str, Any] | None = None
    if not args.disable_lock:
        lock = acquire_dialog_lock(
            args.lock_path,
            timeout_seconds=args.lock_timeout,
            stale_seconds=args.stale_lock_seconds,
        )
        if not lock.get("ok"):
            print(json.dumps(lock, ensure_ascii=False, indent=2))
            return 3

    run_id = make_run_id()
    tests = tests_for_suite(args.suite, include_draft=args.include_draft)
    setup: dict[str, Any] = {"task_close_test_task": {"ok": True, "skipped": True}}
    cleanup: dict[str, Any] = {"task_close_test_task": {"ok": True, "skipped": True}}
    if suite_needs_task_close_task(tests) and optional_int(args.task_close_task_id) is None:
        created = create_task_close_test_task(args, run_id=run_id)
        setup["task_close_test_task"] = created
        if not created.get("ok"):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "suite": args.suite,
                        "run_id": run_id,
                        "stage": "create_task_close_test_task",
                        "setup": setup,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            release_dialog_lock(lock)
            return 2
        args.task_close_task_id = str(created["task_id"])
        args.created_task_close_task_id = created["task_id"]
    else:
        args.created_task_close_task_id = None

    results = []
    stopped_after_failure = False
    try:
        for index, test in enumerate(tests):
            result = run_test(args, run_id=run_id, test=test)
            results.append(result)
            if not result.get("ok") and not args.continue_on_failure:
                for cleanup_test in cleanup_tests_after_failure(tests, index):
                    cleanup_result = run_test(args, run_id=run_id, test=cleanup_test)
                    results.append(cleanup_result)
                stopped_after_failure = True
                break
    finally:
        if not args.keep_created_task:
            cleanup["task_close_test_task"] = delete_task_close_test_task(args)
        release_dialog_lock(lock)
    payload = {
        "ok": all(item.get("ok") for item in results) and cleanup["task_close_test_task"].get("ok", True),
        "suite": args.suite,
        "include_draft": args.include_draft,
        "continue_on_failure": args.continue_on_failure,
        "stopped_after_failure": stopped_after_failure,
        "run_id": run_id,
        "events_url": args.events_url,
        "status_url": args.status_url,
        "lock_path": None if args.disable_lock else args.lock_path,
        "rest_webhook_configured": bool(args.rest_webhook_url),
        "dialog_id": args.dialog_id,
        "dialog_key": args.dialog_key,
        "user_id": args.user_id,
        "bot_id": args.bot_id,
        "setup": setup,
        "cleanup": cleanup,
        "loaded_env_keys": sorted({key for key in loaded_env_keys if key.startswith(("BITRIX_", "WEBHOOK_"))}),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 2


def cleanup_tests_after_failure(tests: list[TestCase], failed_index: int) -> list[TestCase]:
    if tests[failed_index].kind not in {"draft", "task_close_draft"}:
        return []
    cleanup_tests: list[TestCase] = []
    for test in tests[failed_index + 1 :]:
        if test.kind not in {"draft_cleanup", "task_close_cleanup"}:
            break
        cleanup_tests.append(test)
    return cleanup_tests


if __name__ == "__main__":
    raise SystemExit(main())
