"""
Phase 10: typed request/response models for the FastAPI API layer.

These are presentation-layer models -- separate from the database's ORM
models (app.database.models) and from the detection pipeline's own
dataclasses (app.detection.models, app.ml.inference). Keeping them
separate means the API's public contract can stay stable even if internal
persistence or detection types change shape later.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


def derive_detection_source(rule_score: int, ml_predicted_class: str | None) -> str:
    """
    Classify which detector(s) actually contributed to a detection, for
    display purposes (e.g. a dashboard badge distinguishing "Rule",
    "ML", and "Hybrid" detections per the project's requirement to
    clearly distinguish detection sources).

    - "rule": only the rule engine contributed (no ML ran, or ML ran but
      predicted BENIGN while rules fired).
    - "ml": only ML contributed (rules evaluated but none fired, ML
      predicted ATTACK).
    - "hybrid": both the rule engine and ML contributed a non-zero signal.
    - "none": neither contributed (rules didn't fire, and ML said BENIGN
      or wasn't available) -- included for completeness; such a detection
      would have final_risk_score == 0.
    """
    rules_contributed = rule_score > 0
    ml_contributed = ml_predicted_class == "ATTACK"

    if rules_contributed and ml_contributed:
        return "hybrid"
    if rules_contributed:
        return "rule"
    if ml_contributed:
        return "ml"
    return "none"


class SystemStatusResponse(BaseModel):
    """GET /api/status"""

    environment: str
    capture_interface: str
    ml_active_model_name: str
    database_reachable: bool
    total_flows: int
    total_detections: int


class FlowResponse(BaseModel):
    """GET /api/flows, GET /api/flows/{id}"""

    id: int
    first_seen: datetime
    last_seen: datetime
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    protocol: str
    duration_seconds: float
    packet_count: int
    byte_count: int
    forward_packet_count: int
    backward_packet_count: int
    forward_byte_count: int
    backward_byte_count: int
    feature_snapshot: dict[str, float] | None = None


class AlertListItemResponse(BaseModel):
    """One row of GET /api/alerts."""

    id: int
    flow_id: int
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    protocol: str
    rule_score: int
    ml_predicted_class: str | None
    ml_probability: float | None
    final_risk_score: int
    severity: str
    detection_source: str


class AlertDetailResponse(BaseModel):
    """GET /api/alerts/{id} -- full detail including rule evidence, per-flow alerts, etc."""

    id: int
    flow_id: int
    timestamp: datetime
    rule_score: int
    triggered_rules: list[str]
    rule_evidence: dict[str, dict[str, float]]
    ml_model_type: str | None
    ml_predicted_class: str | None
    ml_probability: float | None
    ml_error: str | None
    final_risk_score: int
    severity: str
    rule_weight: float
    ml_weight: float
    detection_source: str


class DetectionStatsResponse(BaseModel):
    """GET /api/stats/detections"""

    total_detections: int
    severity_counts: dict[str, int] = Field(default_factory=dict)


class TrafficStatsResponse(BaseModel):
    """GET /api/stats/traffic"""

    total_flows: int
    total_packets: int
    total_bytes: int
    protocol_counts: dict[str, int] = Field(default_factory=dict)
    average_duration_seconds: float


# ---------------------------------------------------------------------------
# Converters: database read-view dataclasses -> API response models.
# Public (not underscore-prefixed) so routers can share them without
# reaching into each other's "private" helpers.
# ---------------------------------------------------------------------------


def alert_list_item_to_response(item) -> AlertListItemResponse:
    return AlertListItemResponse(
        id=item.id,
        flow_id=item.flow_id,
        timestamp=item.timestamp,
        src_ip=item.src_ip,
        dst_ip=item.dst_ip,
        src_port=item.src_port,
        dst_port=item.dst_port,
        protocol=item.protocol,
        rule_score=item.rule_score,
        ml_predicted_class=item.ml_predicted_class,
        ml_probability=item.ml_probability,
        final_risk_score=item.final_risk_score,
        severity=item.severity,
        detection_source=derive_detection_source(item.rule_score, item.ml_predicted_class),
    )


def persisted_detection_to_response(detection) -> AlertDetailResponse:
    return AlertDetailResponse(
        id=detection.id,
        flow_id=detection.flow_id,
        timestamp=detection.timestamp,
        rule_score=detection.rule_score,
        triggered_rules=detection.triggered_rules,
        rule_evidence=detection.rule_evidence,
        ml_model_type=detection.ml_model_type,
        ml_predicted_class=detection.ml_predicted_class,
        ml_probability=detection.ml_probability,
        ml_error=detection.ml_error,
        final_risk_score=detection.final_risk_score,
        severity=detection.severity,
        rule_weight=detection.rule_weight,
        ml_weight=detection.ml_weight,
        detection_source=derive_detection_source(detection.rule_score, detection.ml_predicted_class),
    )


def persisted_flow_to_response(flow) -> FlowResponse:
    return FlowResponse(
        id=flow.id,
        first_seen=flow.first_seen,
        last_seen=flow.last_seen,
        src_ip=flow.src_ip,
        dst_ip=flow.dst_ip,
        src_port=flow.src_port,
        dst_port=flow.dst_port,
        protocol=flow.protocol,
        duration_seconds=flow.duration_seconds,
        packet_count=flow.packet_count,
        byte_count=flow.byte_count,
        forward_packet_count=flow.forward_packet_count,
        backward_packet_count=flow.backward_packet_count,
        forward_byte_count=flow.forward_byte_count,
        backward_byte_count=flow.backward_byte_count,
        feature_snapshot=flow.feature_snapshot,
    )
