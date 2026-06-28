from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request

from ..models import AgentTask, AgentTestRequest, UserContext
from ..orchestrators.internal import InternalOrchestrator

router = APIRouter()


@router.post("/orchestrator/test")
async def orchestrator_test(request: Request, body: AgentTestRequest) -> Any:
    manifests = request.app.state.manifests
    task = AgentTask(
        task_id=str(uuid4()),
        source="local_test",
        user=UserContext(id=body.user_id, channel=body.channel, raw={"dialog_id": body.dialog_id}),
        request=body.text,
    )
    result = await InternalOrchestrator(manifests, trace_recorder=request.app.state.trace_recorder).handle(task)
    request.app.state.learning_recorder.record_agent_result(
        task,
        result,
        metadata={"endpoint": "/orchestrator/test", "dialog_id": body.dialog_id},
    )
    return result


@router.post("/agent/test")
async def legacy_agent_test(request: Request, body: AgentTestRequest) -> Any:
    return await orchestrator_test(request, body)
