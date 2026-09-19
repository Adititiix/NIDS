"""
Phase 9: flow/detection persistence repository.

This is the ONLY layer that translates between the project's domain
objects (`FinalizedFlow`, `FeatureVector`, `HybridDetectionResult`) and the
database. A future orchestrator (or Phase 10's API layer) calls
`FlowDetectionRepository.save_flow()` / `.save_detection()` /
`.get_recent_detections()` etc. and never needs to import SQLAlchemy,
construct a session, or handle a `SQLAlchemyError` directly -- every
database failure surfaces as the single typed `DatabasePersistenceError`
defined here, per the project's "capture/detection should not become
tightly coupled to database implementation details" requirement.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.database.models import (
    AlertListItem,
    DetectionRecord,
    FlowRecord,
    PersistedDetection,
    PersistedFlow,
    TrafficSummary,
)
from app.detection.models import HybridDetectionResult
from app.features.models import FeatureVector
from app.flows.models import FinalizedFlow

logger = logging.getLogger("nids.database.repository")


class DatabasePersistenceError(Exception):
    """
    Raised for any database failure encountered by this repository --
    connection errors, constraint violations, etc. Wraps the underlying
    SQLAlchemy exception (available via `__cause__`) so callers can log or
    inspect it without needing to import SQLAlchemy themselves.
    """


def _to_utc_datetime(unix_timestamp: float) -> datetime:
    return datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)


class FlowDetectionRepository:
    """
    Persists `FinalizedFlow` and `HybridDetectionResult` domain objects, and
    provides a small set of read queries for a future dashboard/API.

    Constructed with a `sessionmaker` (see `app.database.session.create_session_factory`)
    rather than a single shared `Session`, so each method opens and closes
    its own short-lived session -- appropriate for a repository that may be
    called from different threads/contexts (e.g. a future orchestrator
    processing one flow at a time) without session-sharing pitfalls.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save_flow(self, flow: FinalizedFlow, feature_vector: FeatureVector | None = None) -> int:
        """
        Persist a FinalizedFlow (optionally with its full feature snapshot)
        and return the new row's id. Raises DatabasePersistenceError on any
        database failure.
        """
        try:
            with self._session_factory() as session:
                record = FlowRecord(
                    first_seen=_to_utc_datetime(flow.first_seen),
                    last_seen=_to_utc_datetime(flow.last_seen),
                    src_ip=flow.origin_src_ip,
                    dst_ip=flow.origin_dst_ip,
                    src_port=flow.origin_src_port,
                    dst_port=flow.origin_dst_port,
                    protocol=flow.key.protocol.value,
                    duration_seconds=flow.duration_seconds,
                    packet_count=flow.packet_count,
                    byte_count=flow.byte_count,
                    forward_packet_count=flow.forward_packet_count,
                    backward_packet_count=flow.backward_packet_count,
                    forward_byte_count=flow.forward_byte_count,
                    backward_byte_count=flow.backward_byte_count,
                    feature_snapshot=feature_vector.to_dict() if feature_vector is not None else None,
                )
                session.add(record)
                session.commit()
                return record.id
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to save flow ({flow.origin_src_ip} -> {flow.origin_dst_ip}): {exc}") from exc

    def save_detection(self, flow_id: int, result: HybridDetectionResult) -> int:
        """
        Persist a HybridDetectionResult, foreign-keyed to an already-saved
        flow. Raises DatabasePersistenceError on any database failure,
        including a foreign-key violation if `flow_id` doesn't exist.
        """
        try:
            with self._session_factory() as session:
                record = DetectionRecord(
                    flow_id=flow_id,
                    timestamp=datetime.now(timezone.utc),
                    rule_score=result.rule_result.risk_score,
                    triggered_rules=list(result.rule_result.rule_names),
                    rule_evidence={trigger.rule_name: trigger.evidence for trigger in result.rule_result.triggered_rules},
                    ml_model_type=result.ml_result.model_type if result.ml_result is not None else None,
                    ml_predicted_class=result.ml_result.predicted_class if result.ml_result is not None else None,
                    ml_probability=result.ml_result.probability if result.ml_result is not None else None,
                    ml_error=result.ml_error,
                    final_risk_score=result.final_risk_score,
                    severity=result.final_severity.value,
                    rule_weight=result.rule_weight,
                    ml_weight=result.ml_weight,
                )
                session.add(record)
                session.commit()
                return record.id
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to save detection for flow_id={flow_id}: {exc}") from exc

    def get_flow(self, flow_id: int) -> PersistedFlow | None:
        try:
            with self._session_factory() as session:
                record = session.get(FlowRecord, flow_id)
                return None if record is None else _flow_record_to_view(record)
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to fetch flow_id={flow_id}: {exc}") from exc

    def get_detection(self, detection_id: int) -> PersistedDetection | None:
        try:
            with self._session_factory() as session:
                record = session.get(DetectionRecord, detection_id)
                return None if record is None else _detection_record_to_view(record)
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to fetch detection_id={detection_id}: {exc}") from exc

    def get_recent_detections(self, limit: int = 50) -> list[PersistedDetection]:
        """Most recent detections first, by timestamp -- the natural query for a live dashboard feed."""
        try:
            with self._session_factory() as session:
                stmt = select(DetectionRecord).order_by(DetectionRecord.timestamp.desc()).limit(limit)
                records = session.scalars(stmt).all()
                return [_detection_record_to_view(record) for record in records]
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to fetch recent detections: {exc}") from exc

    def get_detections_for_flow(self, flow_id: int) -> list[PersistedDetection]:
        try:
            with self._session_factory() as session:
                stmt = select(DetectionRecord).where(DetectionRecord.flow_id == flow_id).order_by(DetectionRecord.timestamp.asc())
                records = session.scalars(stmt).all()
                return [_detection_record_to_view(record) for record in records]
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to fetch detections for flow_id={flow_id}: {exc}") from exc

    def count_detections_by_severity(self) -> dict[str, int]:
        """Convenience aggregate for a future dashboard's severity-distribution chart."""
        try:
            with self._session_factory() as session:
                stmt = select(DetectionRecord.severity)
                severities = session.scalars(stmt).all()
                counts: dict[str, int] = {}
                for severity in severities:
                    counts[severity] = counts.get(severity, 0) + 1
                return counts
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to count detections by severity: {exc}") from exc

    # --- Added for Phase 10 (FastAPI/WebSocket API). All additive: nothing
    # above this line was changed. These use real SQL aggregates
    # (func.count/func.sum/func.avg, GROUP BY) rather than loading rows
    # into Python, unlike count_detections_by_severity() above -- that
    # method's Python-side counting was already a documented Phase 9
    # limitation and is left as-is rather than rewritten here.

    def count_flows(self) -> int:
        try:
            with self._session_factory() as session:
                return session.scalar(select(func.count()).select_from(FlowRecord)) or 0
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to count flows: {exc}") from exc

    def count_detections(self) -> int:
        try:
            with self._session_factory() as session:
                return session.scalar(select(func.count()).select_from(DetectionRecord)) or 0
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to count detections: {exc}") from exc

    def list_flows(self, limit: int = 50, offset: int = 0) -> list[PersistedFlow]:
        """Most recently first-seen flows first -- the natural listing order for a dashboard."""
        try:
            with self._session_factory() as session:
                stmt = select(FlowRecord).order_by(FlowRecord.first_seen.desc()).offset(offset).limit(limit)
                records = session.scalars(stmt).all()
                return [_flow_record_to_view(record) for record in records]
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to list flows: {exc}") from exc

    def get_recent_alerts(self, limit: int = 50, offset: int = 0, severity: str | None = None) -> list[AlertListItem]:
        """
        Detections joined with their flow's src/dst IP, port, and protocol
        -- the query the alerts-list API endpoint uses so a dashboard gets
        everything it needs to render a row in a single request.
        """
        try:
            with self._session_factory() as session:
                stmt = (
                    select(DetectionRecord, FlowRecord)
                    .join(FlowRecord, DetectionRecord.flow_id == FlowRecord.id)
                    .order_by(DetectionRecord.timestamp.desc())
                )
                if severity is not None:
                    stmt = stmt.where(DetectionRecord.severity == severity)
                stmt = stmt.offset(offset).limit(limit)
                rows = session.execute(stmt).all()
                return [_to_alert_list_item(detection, flow) for detection, flow in rows]
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to fetch recent alerts: {exc}") from exc

    def get_traffic_summary(self) -> TrafficSummary:
        """
        Aggregate traffic totals (flow count, packet/byte sums, average
        duration) plus a per-protocol flow count, computed via SQL
        aggregates rather than loading every flow into Python.
        """
        try:
            with self._session_factory() as session:
                totals_stmt = select(
                    func.count(FlowRecord.id),
                    func.coalesce(func.sum(FlowRecord.packet_count), 0),
                    func.coalesce(func.sum(FlowRecord.byte_count), 0),
                    func.coalesce(func.avg(FlowRecord.duration_seconds), 0.0),
                )
                total_flows, total_packets, total_bytes, average_duration = session.execute(totals_stmt).one()

                protocol_stmt = select(FlowRecord.protocol, func.count()).group_by(FlowRecord.protocol)
                protocol_counts = {protocol: count for protocol, count in session.execute(protocol_stmt).all()}

                return TrafficSummary(
                    total_flows=int(total_flows),
                    total_packets=int(total_packets),
                    total_bytes=int(total_bytes),
                    protocol_counts=protocol_counts,
                    average_duration_seconds=float(average_duration),
                )
        except SQLAlchemyError as exc:
            raise DatabasePersistenceError(f"Failed to compute traffic summary: {exc}") from exc


def _flow_record_to_view(record: FlowRecord) -> PersistedFlow:
    return PersistedFlow(
        id=record.id,
        first_seen=record.first_seen,
        last_seen=record.last_seen,
        src_ip=record.src_ip,
        dst_ip=record.dst_ip,
        src_port=record.src_port,
        dst_port=record.dst_port,
        protocol=record.protocol,
        duration_seconds=record.duration_seconds,
        packet_count=record.packet_count,
        byte_count=record.byte_count,
        forward_packet_count=record.forward_packet_count,
        backward_packet_count=record.backward_packet_count,
        forward_byte_count=record.forward_byte_count,
        backward_byte_count=record.backward_byte_count,
        feature_snapshot=record.feature_snapshot,
    )


def _detection_record_to_view(record: DetectionRecord) -> PersistedDetection:
    return PersistedDetection(
        id=record.id,
        flow_id=record.flow_id,
        timestamp=record.timestamp,
        rule_score=record.rule_score,
        triggered_rules=list(record.triggered_rules),
        rule_evidence=dict(record.rule_evidence),
        ml_model_type=record.ml_model_type,
        ml_predicted_class=record.ml_predicted_class,
        ml_probability=record.ml_probability,
        ml_error=record.ml_error,
        final_risk_score=record.final_risk_score,
        severity=record.severity,
        rule_weight=record.rule_weight,
        ml_weight=record.ml_weight,
    )


def _to_alert_list_item(detection: DetectionRecord, flow: FlowRecord) -> AlertListItem:
    return AlertListItem(
        id=detection.id,
        flow_id=detection.flow_id,
        timestamp=detection.timestamp,
        src_ip=flow.src_ip,
        dst_ip=flow.dst_ip,
        src_port=flow.src_port,
        dst_port=flow.dst_port,
        protocol=flow.protocol,
        rule_score=detection.rule_score,
        ml_model_type=detection.ml_model_type,
        ml_predicted_class=detection.ml_predicted_class,
        ml_probability=detection.ml_probability,
        final_risk_score=detection.final_risk_score,
        severity=detection.severity,
    )
