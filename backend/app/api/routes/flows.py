"""Phase 10: GET /api/flows, GET /api/flows/{id}, GET /api/flows/{id}/alerts."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_repository
from app.api.schemas import AlertDetailResponse, FlowResponse, persisted_detection_to_response, persisted_flow_to_response
from app.database.repository import DatabasePersistenceError, FlowDetectionRepository

router = APIRouter(prefix="/api/flows", tags=["flows"])


@router.get("", response_model=list[FlowResponse])
async def list_flows(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repo: FlowDetectionRepository = Depends(get_repository),
) -> list[FlowResponse]:
    try:
        flows = repo.list_flows(limit=limit, offset=offset)
    except DatabasePersistenceError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    return [persisted_flow_to_response(flow) for flow in flows]


@router.get("/{flow_id}", response_model=FlowResponse)
async def get_flow(flow_id: int, repo: FlowDetectionRepository = Depends(get_repository)) -> FlowResponse:
    try:
        flow = repo.get_flow(flow_id)
    except DatabasePersistenceError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    if flow is None:
        raise HTTPException(status_code=404, detail=f"Flow {flow_id} not found.")

    return persisted_flow_to_response(flow)


@router.get("/{flow_id}/alerts", response_model=list[AlertDetailResponse])
async def get_flow_alerts(flow_id: int, repo: FlowDetectionRepository = Depends(get_repository)) -> list[AlertDetailResponse]:
    try:
        flow = repo.get_flow(flow_id)
        if flow is None:
            raise HTTPException(status_code=404, detail=f"Flow {flow_id} not found.")
        detections = repo.get_detections_for_flow(flow_id)
    except DatabasePersistenceError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    return [persisted_detection_to_response(detection) for detection in detections]
