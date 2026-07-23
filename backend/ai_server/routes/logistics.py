from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/logistics/vehicle-usage/status")
def logistics_vehicle_usage_status(request: Request) -> dict[str, Any]:
    store = getattr(request.app.state, "vehicle_usage_store", None)
    latest_requests = store.latest_requests(limit=10) if store is not None else []
    return {
        "status": dict(request.app.state.vehicle_usage_status),
        "latest_requests": latest_requests,
    }
