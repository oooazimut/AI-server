from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/logistics/vehicle-usage/status")
def logistics_vehicle_usage_status(request: Request) -> dict[str, Any]:
    store = getattr(request.app.state, "vehicle_usage_store", None)
    latest_requests = store.latest_requests(limit=10) if store is not None else []
    return {
        "status": dict(request.app.state.vehicle_usage_status),
        "latest_requests": latest_requests,
    }


@router.post("/logistics/vehicle-usage/run-once")
async def logistics_vehicle_usage_run_once(request: Request) -> dict[str, Any]:
    specialist = getattr(request.app.state, "logistics_specialist", None)
    if specialist is None:
        raise HTTPException(status_code=503, detail="Vehicle usage not enabled or logistics specialist not started")
    await specialist._morning_handler()
    return {"ok": True, "status": dict(request.app.state.vehicle_usage_status)}


@router.get("/agent/vehicles/status")
def legacy_vehicle_usage_status(request: Request) -> dict[str, Any]:
    return logistics_vehicle_usage_status(request)
