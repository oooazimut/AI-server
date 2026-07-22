from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anyio

from ai_server.models import AgentResult, AgentTask
from ai_server.orchestrators.internal import InternalOrchestrator
from ai_server.workers.bitrix.webhook_event_queue import _handle_dialog_guard, _route_event
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


def test_route_event_prompts_when_dialog_active_too_long():
    guard = FakeDialogGuard(active={"age_seconds": 120, "task_id": "old"})
    queue = RecordingAgentQueue()
    sender = RecordingBitrixSender()

    result = _run(
        _route_event(
            event_id=1,
            event_type="ONIMBOTV2MESSAGEADD",
            payload=_payload("Bitrix show warehouse Borisov"),
            agent_queue=queue,
            attachment_service=FakeAttachmentService(),
            transcriber=FakeTranscriber(),
            settings=_settings(),
            dialog_guard=guard,
            bitrix_sender=sender,
        )
    )

    assert result["routed_to"] == "dialog_guard"
    assert result["action"] == "stuck_prompt"
    assert queue.published == []
    assert guard.pending is not None
    assert "сбросить предыдущий запрос" in sender.messages[0][1]
    assert "выполнить после предыдущего" in sender.messages[0][1]


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


def test_dialog_guard_reset_publishes_pending_task_and_increments_generation():
    pending = _task("Bitrix find project Largus")
    guard = FakeDialogGuard(active={"age_seconds": 130, "task_id": "old"}, pending=pending)
    queue = RecordingAgentQueue()
    sender = RecordingBitrixSender()

    result = _run(
        _handle_dialog_guard(
            _task("сбросить предыдущий запрос"),
            agent_queue=queue,
            settings=_settings(),
            dialog_guard=guard,
            bitrix_sender=sender,
            conversation_trace=None,
            event_id=2,
            event_type="ONIMBOTV2MESSAGEADD",
            partition_key="chat:77:user:9",
        )
    )

    assert result["action"] == "reset_previous"
    assert guard.generation == 1
    assert queue.removed == [("orchestrator", "dialog:chat:77:user:9"), ("bitrix24", "dialog:chat:77:user:9")]
    assert queue.published[0]["payload"]["request"] == "Bitrix find project Largus"
    assert queue.published[0]["payload"]["context"]["dialog_cancel_generation"] == 1
    assert "Сбросил предыдущий запрос" in sender.messages[0][1]


def test_dialog_guard_wait_publishes_pending_after_previous():
    pending = _task("Bitrix find project Largus")
    guard = FakeDialogGuard(active={"age_seconds": 130, "task_id": "old"}, pending=pending)
    queue = RecordingAgentQueue()
    sender = RecordingBitrixSender()

    result = _run(
        _handle_dialog_guard(
            _task("дождаться предыдущего ответа"),
            agent_queue=queue,
            settings=_settings(),
            dialog_guard=guard,
            bitrix_sender=sender,
            conversation_trace=None,
            event_id=2,
            event_type="ONIMBOTV2MESSAGEADD",
            partition_key="chat:77:user:9",
        )
    )

    assert result["action"] == "wait_previous"
    assert guard.generation == 0
    assert queue.published[0]["payload"]["request"] == "Bitrix find project Largus"
    assert queue.published[0]["payload"]["context"]["dialog_cancel_generation"] == 0
    assert "после предыдущего" in sender.messages[0][1]


def test_dialog_guard_reprompts_unknown_choice():
    guard = FakeDialogGuard(pending=_task("Bitrix find project Largus"))
    queue = RecordingAgentQueue()
    sender = RecordingBitrixSender()

    result = _run(
        _handle_dialog_guard(
            _task("да"),
            agent_queue=queue,
            settings=_settings(),
            dialog_guard=guard,
            bitrix_sender=sender,
            conversation_trace=None,
            event_id=2,
            event_type="ONIMBOTV2MESSAGEADD",
            partition_key="chat:77:user:9",
        )
    )

    assert result["action"] == "clarify_choice"
    assert queue.published == []
    assert "Ответьте одной из фраз" in sender.messages[0][1]


def test_orchestrator_suppresses_stale_outbound_answer():
    channel = RecordingChannel()
    guard = FakeDialogGuard(generation=1)
    orchestrator = InternalOrchestrator(
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
