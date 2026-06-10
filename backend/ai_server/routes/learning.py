from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header, Query, Request

from ..learning import LearningEventRecorder
from ..models import LearningFeedbackRequest
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
