"""Phase 10: GET /api/stats/detections, GET /api/stats/traffic."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_repository
from app.api.schemas import DetectionStatsResponse, TrafficStatsResponse
from app.database.repository import DatabasePersistenceError, FlowDetectionRepository

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/detections", response_model=DetectionStatsResponse)
async def get_detection_stats(repo: FlowDetectionRepository = Depends(get_repository)) -> DetectionStatsResponse:
    try:
        total = repo.count_detections()
        severity_counts = repo.count_detections_by_severity()
    except DatabasePersistenceError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    return DetectionStatsResponse(total_detections=total, severity_counts=severity_counts)


@router.get("/traffic", response_model=TrafficStatsResponse)
async def get_traffic_stats(repo: FlowDetectionRepository = Depends(get_repository)) -> TrafficStatsResponse:
    try:
        summary = repo.get_traffic_summary()
    except DatabasePersistenceError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    return TrafficStatsResponse(
        total_flows=summary.total_flows,
        total_packets=summary.total_packets,
        total_bytes=summary.total_bytes,
        protocol_counts=summary.protocol_counts,
        average_duration_seconds=summary.average_duration_seconds,
    )
