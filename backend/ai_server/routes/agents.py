from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request

from ..models import AgentTask, AgentTestRequest, UserContext
from ..orchestrators.internal import InternalOrchestrator
from ..technical_footer import TechnicalFooterService

router = APIRouter()


@router.post("/orchestrator/test")
async def orchestrator_test(request: Request, body: AgentTestRequest) -> Any:
    manifests = request.app.state.manifests
    orch_manifest = next((m for m in manifests if m.kind == "orchestrator"), None)
    task = AgentTask(
        task_id=str(uuid4()),
        source="local_test",
        user=UserContext(id=body.user_id, channel=body.channel, raw={"dialog_id": body.dialog_id}),
        request=body.text,
    )
    orchestrator = InternalOrchestrator.build(
        orch_manifest,
        manifests=manifests,
        settings=request.app.state.settings,
        bitrix_client=request.app.state.bitrix,
        portal_search_index=request.app.state.portal_search,
        bitrix_oauth=request.app.state.bitrix_oauth,
        bitrix_bot=request.app.state.bitrix,
        scheduler=getattr(request.app.state, "scheduler", None),
        footer_service=TechnicalFooterService(settings=request.app.state.settings),
        learning_recorder=request.app.state.learning_recorder,
        trace_recorder=request.app.state.trace_recorder,
    )
    result = await orchestrator.handle(task)
    request.app.state.learning_recorder.record_agent_result(
        task,
        result,
        metadata={"endpoint": "/orchestrator/test", "dialog_id": body.dialog_id},
    )
    return result


@router.post("/agent/test")
async def legacy_agent_test(request: Request, body: AgentTestRequest) -> Any:
    return await orchestrator_test(request, body)
