"""Phase 10: GET /api/alerts, GET /api/alerts/{id}."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_repository
from app.api.schemas import AlertDetailResponse, AlertListItemResponse, alert_list_item_to_response, persisted_detection_to_response
from app.database.repository import DatabasePersistenceError, FlowDetectionRepository
from app.detection.models import Severity

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

_VALID_SEVERITIES = {s.value for s in Severity}


@router.get("", response_model=list[AlertListItemResponse])
async def list_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    severity: str | None = Query(default=None, description="Filter by severity: NONE, LOW, MEDIUM, HIGH, or CRITICAL."),
    repo: FlowDetectionRepository = Depends(get_repository),
) -> list[AlertListItemResponse]:
    if severity is not None and severity not in _VALID_SEVERITIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid severity '{severity}'. Must be one of {sorted(_VALID_SEVERITIES)}.",
        )

    try:
        items = repo.get_recent_alerts(limit=limit, offset=offset, severity=severity)
    except DatabasePersistenceError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    return [alert_list_item_to_response(item) for item in items]


@router.get("/{alert_id}", response_model=AlertDetailResponse)
async def get_alert(alert_id: int, repo: FlowDetectionRepository = Depends(get_repository)) -> AlertDetailResponse:
    try:
        detection = repo.get_detection(alert_id)
    except DatabasePersistenceError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    if detection is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")

    return persisted_detection_to_response(detection)
