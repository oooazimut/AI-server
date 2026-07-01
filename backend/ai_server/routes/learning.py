from __future__ import annotations

from typing import Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/learning/status")
def learning_status() -> dict[str, Any]:
    return {"status": "learning_recorder removed — diagnostics now handled by ИИ-Диагност specialist"}
