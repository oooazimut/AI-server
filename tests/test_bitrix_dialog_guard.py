from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anyio

from ai_server.models import AgentResult, AgentTask
from ai_server.orchestrators.internal import OrchestratorTransportRuntime
from ai_server.workers.bitrix.webhook_event_queue import _route_event
from tests.fakes import FakeOrchestratorStore


def _run(coro):
    async def _runner():
        return await coro

    return anyio.run(_runner)


def _settings(**overrides) -> SimpleNamespace:
    defaults = {
        "bitrix_dialog_guard_enabled": True,
        "bitrix_dialog_stuck_seconds": 90,
        "bitrix_dialog_pending_ttl_seconds": 600,
        "agent_dry_run": False,
        "bitrix_bot_id": 231,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


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


def _task(text: str = "Bitrix show warehouse Borisov") -> AgentTask:
    return AgentTask(
        task_id="task-new",
        request=text,
        user={"id": "9"},
        context={
            "dialog_key": "chat:77:user:9",
            "base_dialog_key": "chat:77:user:9",
            "dialog_id": "chat77",
            "recipient_id": "chat77",
            "channel_id": "bitrix24",
        },
    )


def test_route_event_assigns_numbered_branch_before_orchestrator_queue():
    store = FakeOrchestratorStore()
    queue = RecordingAgentQueue()

    first = _run(
        _route_event(
            event_id=1,
            event_type="ONIMBOTV2MESSAGEADD",
            payload=_payload("Покажи склад Борисов"),
            agent_queue=queue,
            attachment_service=FakeAttachmentService(),
            transcriber=FakeTranscriber(),
            settings=_settings(),
            orchestrator_store=store,
        )
    )
    second = _run(
        _route_event(
            event_id=2,
            event_type="ONIMBOTV2MESSAGEADD",
            payload=_payload("Покажи склад Карасев"),
            agent_queue=queue,
            attachment_service=FakeAttachmentService(),
            transcriber=FakeTranscriber(),
            settings=_settings(),
            orchestrator_store=store,
        )
    )

    assert first["routed_to"] == second["routed_to"] == "orchestrator"
    contexts = [item["payload"]["context"] for item in queue.published]
    assert [item["conversation_number"] for item in contexts] == [101, 102]
    assert contexts[0]["dialog_key"] != contexts[1]["dialog_key"]


def test_task_update_goes_only_to_search_refresh():
    queue = RecordingAgentQueue()
    payload = {"event": "ONTASKUPDATE", "data": {"FIELDS_AFTER": {"ID": "8413"}}}

    result = _run(
        _route_event(
            event_id=3,
            event_type="ONTASKUPDATE",
            payload=payload,
            agent_queue=queue,
            attachment_service=FakeAttachmentService(),
            transcriber=FakeTranscriber(),
            settings=_settings(),
        )
    )

    assert result["routed_to"] == "index_refresher"
    assert [item["to"] for item in queue.published] == ["index_refresher"]
    assert queue.published[0]["payload"] == payload


def test_comment_and_catalog_events_go_to_search_refresh():
    for event_type, fields in (
        ("ONTASKCOMMENTADD", {"TASK_ID": "8413"}),
        ("CATALOG.PRODUCT.ON.UPDATE", {"ID": "1001"}),
    ):
        queue = RecordingAgentQueue()
        payload = {"event": event_type, "data": {"FIELDS_AFTER": fields}}

        result = _run(
            _route_event(
                event_id=4,
                event_type=event_type,
                payload=payload,
                agent_queue=queue,
                attachment_service=FakeAttachmentService(),
                transcriber=FakeTranscriber(),
                settings=_settings(),
            )
        )

        assert result["routed_to"] == "index_refresher"
        assert [item["to"] for item in queue.published] == ["index_refresher"]


def test_orchestrator_suppresses_stale_outbound_answer():
    channel = RecordingChannel()
    guard = FakeDialogGuard(generation=1)
    orchestrator = OrchestratorTransportRuntime(
        SimpleNamespace(id="internal_orchestrator"),
        agent_tools=[],
        llm=None,
        channels={"bitrix24": channel},
        dialog_guard=guard,
    )
    task = _task("old request")
    task.context["dialog_cancel_generation"] = 0
    result = AgentResult(status="completed", agent_id="internal_orchestrator", answer="old answer")

    _run(orchestrator._send_to_channel(task, result))

    assert channel.messages == []


class FakeDialogGuard:
    def __init__(
        self,
        *,
        active: dict[str, Any] | None = None,
        pending: AgentTask | None = None,
        generation: int = 0,
    ) -> None:
        self.active = active
        self.pending = pending
        self.generation = generation

    async def current_generation(self, dialog_key: str) -> int:
        return self.generation

    async def increment_generation(self, dialog_key: str) -> int:
        self.generation += 1
        return self.generation

    async def get_active(self, dialog_key: str) -> dict[str, Any] | None:
        return self.active

    async def save_pending(self, task: AgentTask, *, ttl_seconds: int) -> None:
        self.pending = task

    async def get_pending(self, dialog_key: str) -> AgentTask | None:
        return self.pending

    async def pop_pending(self, dialog_key: str) -> AgentTask | None:
        task = self.pending
        self.pending = None
        return task

    async def task_is_stale(self, task: AgentTask) -> bool:
        return int(task.context.get("dialog_cancel_generation") or 0) < self.generation


class RecordingAgentQueue:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self.removed: list[tuple[str, str]] = []

    async def publish(self, message: dict[str, Any]) -> None:
        self.published.append(message)

    async def remove_pending_by_partition(self, agent_id: str, partition_key: str) -> int:
        self.removed.append((agent_id, partition_key))
        return 1


class RecordingBitrixSender:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, int | None]] = []

    async def send_bot_message(self, dialog_id: str, message: str, *, bot_id=None, keyboard=None) -> int:
        self.messages.append((dialog_id, message, bot_id))
        return 1


class RecordingChannel:
    channel_id = "bitrix24"

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def send(self, recipient_id: str, body: str) -> None:
        self.messages.append((recipient_id, body))


class FakeAttachmentService:
    async def download_message_files(self, message):
        return []


class FakeTranscriber:
    async def transcribe(self, attachment):
        raise AssertionError("no voice files expected")
