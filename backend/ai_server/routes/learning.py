from __future__ import annotations

from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Query, Request

from ..agents.diagnostic_agent import DiagnosticAgent
from ..learning import LearningEventRecorder
from ..models import AgentTask, LearningDiagnosticRequest, LearningFeedbackRequest, UserContext
from ..specialists import manifest_by_id
from ._common import request_secret, validate_webhook_secret

router = APIRouter()


@router.get("/learning/status")
def learning_status(request: Request) -> dict[str, Any]:
    return request.app.state.learning_recorder.stats()


@router.get("/learning/events")
def learning_events(
    request: Request,
    x_agent_secret: Annotated[str | None, Header(alias="X-Agent-Secret")] = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    validate_webhook_secret(request.app.state.settings, request_secret(request, x_agent_secret))
    recorder: LearningEventRecorder = request.app.state.learning_recorder
    return {"events": recorder.latest(limit=limit), "status": recorder.stats()}


@router.get("/learning/diagnostics/groups")
def learning_diagnostic_groups(
    request: Request,
    x_agent_secret: Annotated[str | None, Header(alias="X-Agent-Secret")] = None,
    limit: int = Query(default=100, ge=1, le=500),
    detailed: bool = Query(default=False),
) -> dict[str, Any]:
    validate_webhook_secret(request.app.state.settings, request_secret(request, x_agent_secret))
    recorder: LearningEventRecorder = request.app.state.learning_recorder
    return recorder.diagnostic_groups(limit=limit, detailed=detailed)


@router.post("/learning/feedback")
def learning_feedback(
    request: Request,
    body: LearningFeedbackRequest,
    x_agent_secret: Annotated[str | None, Header(alias="X-Agent-Secret")] = None,
) -> dict[str, Any]:
    validate_webhook_secret(request.app.state.settings, request_secret(request, x_agent_secret))
    recorder: LearningEventRecorder = request.app.state.learning_recorder
    return recorder.record_feedback(
        event_id=body.event_id,
        rating=body.rating,
        corrected_answer=body.corrected_answer,
        comment=body.comment,
        tags=body.tags,
        user_id=body.user_id,
        channel=body.channel,
    )


@router.post("/learning/diagnose")
async def learning_diagnose(
    request: Request,
    body: LearningDiagnosticRequest,
    x_agent_secret: Annotated[str | None, Header(alias="X-Agent-Secret")] = None,
) -> dict[str, Any]:
    validate_webhook_secret(request.app.state.settings, request_secret(request, x_agent_secret))
    recorder: LearningEventRecorder = request.app.state.learning_recorder
    target_event = recorder.get_event(body.event_id)
    if target_event is None:
        raise HTTPException(status_code=404, detail=f"Learning event not found: {body.event_id}")

    feedback_events = recorder.feedback_for(body.event_id, limit=10)
    if body.feedback_event_id:
        feedback_events = [event for event in feedback_events if event.get("id") == body.feedback_event_id]

    manifest = manifest_by_id(request.app.state.manifests, "diagnostic_agent")
    if manifest is None:
        raise HTTPException(status_code=404, detail="Diagnostic Agent manifest not found")

    diagnostic_llm = getattr(request.app.state, "diagnostic_llm", None)
    diagnostic_agent = DiagnosticAgent.build(manifest, diagnostic_llm=diagnostic_llm)
    task = AgentTask(
        task_id=str(uuid4()),
        source="learning_diagnose",
        user=UserContext(channel="diagnostics"),
        request=body.comment or f"Разбери learning event {body.event_id}",
        context={
            "event_id": body.event_id,
            "target_event": target_event,
            "feedback_events": feedback_events,
            "feedback": feedback_events[-1] if feedback_events else {},
            "rating": (feedback_events[-1].get("metadata") or {}).get("rating") if feedback_events else None,
            "comment": body.comment,
        },
    )
    result = await diagnostic_agent.handle(task)
    diagnostic_record = recorder.record_agent_result(
        task,
        result,
        event_type="diagnostic_report",
        metadata={
            "target_event_id": body.event_id,
            "feedback_event_ids": [event.get("id") for event in feedback_events],
            "feedback_event_id": body.feedback_event_id or "",
        },
    )
    return {
        "status": result.status,
        "answer": result.answer,
        "event_id": body.event_id,
        "feedback_events": [event.get("id") for event in feedback_events],
        "diagnostic_event": diagnostic_record,
        "diagnostic_agent": result.model_dump(),
    }
