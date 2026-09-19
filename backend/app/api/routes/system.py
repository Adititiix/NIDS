"""Phase 10: GET /api/status -- system status beyond the basic /api/health liveness check."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_repository
from app.api.schemas import SystemStatusResponse
from app.config import settings
from app.database.repository import DatabasePersistenceError, FlowDetectionRepository

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(repo: FlowDetectionRepository = Depends(get_repository)) -> SystemStatusResponse:
    """
    Reports configuration (environment, capture interface, active ML model
    name) plus live database reachability and totals. Unlike /api/health,
    this endpoint actually queries the database -- if that fails,
    `database_reachable` is reported as False rather than the request
    itself failing, since "is the DB up" is exactly the kind of thing a
    status page needs to report even when the answer is "no".
    """
    try:
        total_flows = repo.count_flows()
        total_detections = repo.count_detections()
        database_reachable = True
    except DatabasePersistenceError:
        total_flows = 0
        total_detections = 0
        database_reachable = False

    return SystemStatusResponse(
        environment=settings.environment,
        capture_interface=settings.capture_interface,
        ml_active_model_name=settings.ml_active_model_name,
        database_reachable=database_reachable,
        total_flows=total_flows,
        total_detections=total_detections,
    )
