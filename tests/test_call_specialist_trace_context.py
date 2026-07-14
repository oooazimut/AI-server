from __future__ import annotations

import anyio

from ai_server.models import AgentManifest, AgentResult, AgentTask, UserContext
from ai_server.orchestrators.tools.call_specialist import CallSpecialistTool


class _CapturingSpecialist:
    def __init__(self) -> None:
        self.task: AgentTask | None = None

    async def handle(self, task: AgentTask) -> AgentResult:
        self.task = task
        return AgentResult(status="completed", agent_id="bitrix24", answer="ok")


def test_call_specialist_execute_with_task_preserves_trace_context():
    specialist = _CapturingSpecialist()
    tool = CallSpecialistTool(
        {"bitrix24": specialist},
        [AgentManifest(id="bitrix24", kind="specialist", name="Bitrix", description="")],
    )
    task = AgentTask(
        task_id="task-1",
        request="original",
        user=UserContext(id="27", channel="bitrix24", raw={"message_id": 127203, "chat_id": 3669}),
        context={
            "dialog_key": "chat:3669:user:27",
            "dialog_id": "27",
            "recipient_id": "27",
        },
    )

    async def run():
        return await tool.execute_with_task(
            {"specialist_id": "bitrix24", "request": "delegated"},
            task=task,
        )

    result = anyio.run(run)

    assert result.status == "ok"
    assert specialist.task is not None
    assert specialist.task.task_id == "task-1"
    assert specialist.task.request == "delegated"
    assert specialist.task.user.raw["message_id"] == 127203
    assert specialist.task.context["dialog_key"] == "chat:3669:user:27"
