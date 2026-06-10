from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..workers.logistics.vehicle_usage import run_vehicle_usage_once

router = APIRouter()


@router.get("/logistics/vehicle-usage/status")
def logistics_vehicle_usage_status(request: Request) -> dict[str, Any]:
    from ..tools.vehicle_usage import VehicleUsageStore

    store = VehicleUsageStore()
    return {
        "status": dict(request.app.state.vehicle_usage_status),
        "latest_requests": store.latest_requests(limit=10),
    }


@router.post("/logistics/vehicle-usage/run-once")
async def logistics_vehicle_usage_run_once(request: Request) -> dict[str, Any]:
    result = await run_vehicle_usage_once(
        request.app.state.bitrix,
        status=request.app.state.vehicle_usage_status,
    )
    return {"ok": True, "result": result, "status": dict(request.app.state.vehicle_usage_status)}


@router.get("/agent/vehicles/status")
def legacy_vehicle_usage_status(request: Request) -> dict[str, Any]:
    return logistics_vehicle_usage_status(request)
