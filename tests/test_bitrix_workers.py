from __future__ import annotations

from pathlib import Path

from ai_server.workers.bitrix.reconciler import reconcile_once
from ai_server.workers.bitrix.supervisor import run_task_supervisor_once
from ai_server.workers.bitrix.webhook_event_queue import WebhookEventQueue


def test_task_supervisor_dry_run_does_not_notify(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("SUPERVISOR_ADMIN_USER_IDS", "9")
    monkeypatch.setenv("SUPERVISOR_DRY_RUN", "true")
    bitrix = FakeSupervisorBitrix()

    result = anyio_run(run_task_supervisor_once(bitrix))

    assert result["overdue_tasks_seen"] == 1
    assert result["notifications_planned"] == 1
    assert result["notifications_sent"] == 0
    assert result["notifications"][0]["reason"] == "dry_run"
    assert bitrix.notifications == []


def test_reconciler_enqueues_task_updates_with_dedupe(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVER_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.setenv("RECONCILE_TASKS_ENABLED", "true")
    monkeypatch.setenv("RECONCILE_DISK_DELTA_ENABLED", "false")
    queue = WebhookEventQueue(tmp_path / "webhook_event_queue.sqlite")
    queue.ensure_schema()
    bitrix = FakeReconcileBitrix()

    first = anyio_run(reconcile_once(bitrix, queue, FakeSearchIndexer()))
    second = anyio_run(reconcile_once(bitrix, queue, FakeSearchIndexer()))

    assert first["tasks"]["seen"] == 1
    assert first["tasks"]["enqueued"] == 1
    assert second["tasks"]["duplicates"] == 1
    stats = queue.stats()
    assert stats["pending"] == 1
    assert stats["done"] == 0


class FakeSupervisorBitrix:
    def __init__(self) -> None:
        self.notifications = []

    async def list_all_tasks(self, *, filter_=None, select=None, order=None, limit=None):
        return [
            {
                "ID": "8413",
                "TITLE": "Просроченная задача",
                "RESPONSIBLE_ID": "15",
                "DEADLINE": "2026-06-01T10:00:00+03:00",
                "STATUS": "3",
            }
        ]

    async def get_user(self, user_id: int):
        return {"ID": user_id, "NAME": "Иван", "LAST_NAME": "Петров"}

    async def notify_user(self, *, user_id: int, message: str, tag: str = "ai_server", sub_tag: str = ""):
        self.notifications.append({"user_id": user_id, "message": message, "tag": tag, "sub_tag": sub_tag})
        return 1


class FakeReconcileBitrix:
    async def list_all_tasks(self, *, filter_=None, select=None, order=None, limit=None):
        return [
            {
                "ID": "8413",
                "TITLE": "Измененная задача",
                "STATUS": "3",
                "RESPONSIBLE_ID": "15",
                "GROUP_ID": "7",
                "CHANGED_DATE": "2026-06-05T11:30:00+03:00",
            }
        ]


class FakeSearchIndexer:
    async def run_delta_once(self):
        raise AssertionError("disk delta should be disabled in this test")


def anyio_run(awaitable):
    import anyio

    async def runner():
        return await awaitable

    return anyio.run(runner)
