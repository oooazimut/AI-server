from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anyio

from ai_server.workers.bitrix.dialog_lines import choose_auto_line_id, is_auto_line_candidate
from ai_server.workers.bitrix.webhook_event_queue import _route_event


def _run(coro):
    async def _runner():
        return await coro

    return anyio.run(_runner)


def _payload(text: str) -> dict[str, Any]:
    return {
        "event": "ONIMBOTV2MESSAGEADD",
        "data": {
            "bot": {"id": 231},
            "chat": {"id": 77, "dialogId": "chat77"},
            "message": {"id": 123, "authorId": 9, "text": text},
            "user": {"id": 9},
        },
    }


def _settings(*, enabled: bool = True, max_lines: int = 3) -> SimpleNamespace:
    return SimpleNamespace(bitrix_auto_lines_enabled=enabled, bitrix_auto_line_max=max_lines)


def test_auto_line_candidate_detects_independent_bitrix_requests():
    assert is_auto_line_candidate("Bitrix show warehouse Borisov")
    assert is_auto_line_candidate("покажи склад Борисов")
    assert not is_auto_line_candidate("да")
    assert not is_auto_line_candidate("уточни")


def test_choose_auto_line_treats_base_dialog_as_line_one():
    active = {"dialog:chat:77:user:9"}

    assert choose_auto_line_id(active, "chat:77:user:9", max_lines=3) == 2


def test_route_event_assigns_auto_line_when_base_dialog_is_active():
    queue = RecordingAgentQueue(active={"dialog:chat:77:user:9"})

    result = _run(
        _route_event(
            event_id=1,
            event_type="ONIMBOTV2MESSAGEADD",
            payload=_payload("Bitrix show warehouse Borisov"),
            agent_queue=queue,
            attachment_service=FakeAttachmentService(),
            transcriber=FakeTranscriber(),
            settings=_settings(),
        )
    )

    assert result["routed_to"] == "orchestrator"
    task = queue.published[0]["payload"]
    assert task["request"] == "Bitrix show warehouse Borisov"
    assert task["context"]["base_dialog_key"] == "chat:77:user:9"
    assert task["context"]["dialog_key"] == "chat:77:user:9:line:2"
    assert task["context"]["dialog_line_id"] == "2"
    assert task["context"]["dialog_line_label"] == "Линия 2"
    assert task["context"]["dialog_auto_line"] is True
    assert task["context"]["recipient_id"] == "chat77"


def test_route_event_keeps_followup_in_base_dialog_even_when_active():
    queue = RecordingAgentQueue(active={"dialog:chat:77:user:9"})

    _run(
        _route_event(
            event_id=1,
            event_type="ONIMBOTV2MESSAGEADD",
            payload=_payload("да"),
            agent_queue=queue,
            attachment_service=FakeAttachmentService(),
            transcriber=FakeTranscriber(),
            settings=_settings(),
        )
    )

    task = queue.published[0]["payload"]
    assert task["context"]["dialog_key"] == "chat:77:user:9"
    assert "dialog_line_id" not in task["context"]


class RecordingAgentQueue:
    def __init__(self, *, active: set[str] | None = None) -> None:
        self.active = active or set()
        self.published: list[dict[str, Any]] = []

    async def active_partition_keys(self, agent_id: str) -> set[str]:
        assert agent_id == "orchestrator"
        return set(self.active)

    async def publish(self, message: dict[str, Any]) -> None:
        self.published.append(message)


class FakeAttachmentService:
    async def download_message_files(self, message):
        return []


class FakeTranscriber:
    async def transcribe(self, attachment):
        raise AssertionError("no voice files expected")
