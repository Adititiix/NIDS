"""
Phase 9: PostgreSQL persistence schema.

Two tables:

- `flows`: one row per FinalizedFlow (Phase 3), with an optional JSON
  snapshot of its extracted feature vector (Phase 4) -- no raw packet
  payloads are ever stored, only flow-level metadata and derived features.
- `detections`: one row per HybridDetectionResult (Phase 8), foreign-keyed
  to the flow it was computed from, capturing the rule score, triggered
  rules + evidence, ML prediction/probability (if available), and the
  final fused risk score/severity.

Indexes are placed on the columns this project's own design docs identify
as commonly queried (docs/architecture.md's dashboard design references
timestamp, source IP, severity, and risk score) -- see the `index=True`
markers below.

This module defines the ORM tables (`FlowRecord`, `DetectionRecord`) AND
plain, frozen read-view dataclasses (`PersistedFlow`, `PersistedDetection`)
that the repository layer (app.database.repository) returns instead of raw
ORM instances. This keeps callers (a future API layer, tests, etc.)
decoupled from SQLAlchemy specifics and avoids DetachedInstanceError
pitfalls from touching ORM attributes after their session has closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all Phase 9 ORM models."""


class FlowRecord(Base):
    """One persisted FinalizedFlow. See app.flows.models.FinalizedFlow for the live equivalent."""

    __tablename__ = "flows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    src_ip: Mapped[str] = mapped_column(String(45), nullable=False, index=True)  # 45 chars fits a full IPv6 address
    dst_ip: Mapped[str] = mapped_column(String(45), nullable=False, index=True)
    src_port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dst_port: Mapped[int | None] = mapped_column(Integer, nullable=True)
    protocol: Mapped[str] = mapped_column(String(16), nullable=False)

    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    packet_count: Mapped[int] = mapped_column(Integer, nullable=False)
    byte_count: Mapped[int] = mapped_column(Integer, nullable=False)
    forward_packet_count: Mapped[int] = mapped_column(Integer, nullable=False)
    backward_packet_count: Mapped[int] = mapped_column(Integer, nullable=False)
    forward_byte_count: Mapped[int] = mapped_column(Integer, nullable=False)
    backward_byte_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # Optional full 41-feature snapshot (name -> value), e.g. from
    # FeatureVector.to_dict(). Deliberately NOT the raw packets -- just the
    # already-computed, already-aggregated feature values for this flow.
    feature_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    detections: Mapped[list["DetectionRecord"]] = relationship(back_populates="flow", cascade="all, delete-orphan")


class DetectionRecord(Base):
    """One persisted HybridDetectionResult. See app.detection.models.HybridDetectionResult for the live equivalent."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    flow_id: Mapped[int] = mapped_column(ForeignKey("flows.id"), nullable=False, index=True)

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    rule_score: Mapped[int] = mapped_column(Integer, nullable=False)
    triggered_rules: Mapped[list] = mapped_column(JSON, nullable=False)  # list[str] of rule names
    rule_evidence: Mapped[dict] = mapped_column(JSON, nullable=False)  # {rule_name: {feature: value, ...}}

    ml_model_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    ml_predicted_class: Mapped[str | None] = mapped_column(String(16), nullable=True)
    ml_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    ml_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    final_risk_score: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    rule_weight: Mapped[float] = mapped_column(Float, nullable=False)
    ml_weight: Mapped[float] = mapped_column(Float, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    flow: Mapped["FlowRecord"] = relationship(back_populates="detections")


# ---------------------------------------------------------------------------
# Read-view dataclasses -- what the repository layer actually returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PersistedFlow:
    """Plain, ORM-independent snapshot of a `FlowRecord` row."""

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
    feature_snapshot: dict | None


@dataclass(frozen=True, slots=True)
class PersistedDetection:
    """Plain, ORM-independent snapshot of a `DetectionRecord` row."""

    id: int
    flow_id: int
    timestamp: datetime
    rule_score: int
    triggered_rules: list[str]
    rule_evidence: dict
    ml_model_type: str | None
    ml_predicted_class: str | None
    ml_probability: float | None
    ml_error: str | None
    final_risk_score: int
    severity: str
    rule_weight: float
    ml_weight: float


@dataclass(frozen=True, slots=True)
class AlertListItem:
    """
    A detection joined with its flow's identifying fields (src/dst IP and
    port, protocol) -- added for Phase 10's alerts-list API endpoint, so a
    dashboard can render a useful alert row in one request instead of
    fetching each flow separately (N+1).
    """

    id: int
    flow_id: int
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    protocol: str
    rule_score: int
    ml_model_type: str | None
    ml_predicted_class: str | None
    ml_probability: float | None
    final_risk_score: int
    severity: str


@dataclass(frozen=True, slots=True)
class TrafficSummary:
    """Aggregate traffic statistics across all persisted flows -- added for Phase 10's /api/stats/traffic endpoint."""

    total_flows: int
    total_packets: int
    total_bytes: int
    protocol_counts: dict[str, int]
    average_duration_seconds: float
