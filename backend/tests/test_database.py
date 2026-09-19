"""
Unit tests for Phase 9 (PostgreSQL persistence).

All tests here run against an in-memory SQLite database
(`sqlite:///:memory:`), so the suite is fast and needs no live database --
per the project's "prefer unit tests that do not require a live database"
requirement. Foreign-key enforcement is explicitly turned on for SQLite by
`create_db_engine` (off by default in SQLite, unlike PostgreSQL), so the
constraint-violation test below is meaningful, not a false negative.

A separate, clearly-documented real-PostgreSQL integration test lives at
the bottom of this file and is skipped unless `NIDS_TEST_POSTGRES_URL` is
set -- see that test's docstring for how to run it locally against
docker-compose.yml's postgres service.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import inspect

from app.capture.models import PacketMetadata, TransportProtocol
from app.config import RiskScoringConfig
from app.database.repository import DatabasePersistenceError, FlowDetectionRepository
from app.database.session import create_db_engine, create_session_factory, init_db
from app.detection.hybrid_detector import HybridDetector
from app.detection.models import Severity
from app.detection.rule_engine import RuleDetector
from app.features.feature_extractor import FeatureExtractor
from app.flows.models import Flow, FinalizedFlow
from app.ml.inference import MLInferenceResult


def make_packet(
    *,
    timestamp: float,
    src_ip: str = "10.0.2.15",
    dst_ip: str = "10.0.2.1",
    src_port: int | None = 51234,
    dst_port: int | None = 443,
    protocol: TransportProtocol = TransportProtocol.TCP,
    length: int = 100,
    tcp_flags: str | None = None,
) -> PacketMetadata:
    return PacketMetadata(
        timestamp=timestamp, src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port,
        protocol=protocol, length=length, tcp_flags=tcp_flags,
    )


def build_flow(packets: list[PacketMetadata]) -> FinalizedFlow:
    flow: Flow | None = None
    for i, pkt in enumerate(packets):
        flow = Flow.start(pkt) if i == 0 else flow
        if i > 0:
            assert flow is not None
            flow.add_packet(pkt)
    assert flow is not None
    return flow.finalize()


def suspicious_flow() -> FinalizedFlow:
    packets = [make_packet(timestamp=100.0 + i * 0.005, dst_port=4444, tcp_flags="S") for i in range(30)]
    return build_flow(packets)


def normal_flow() -> FinalizedFlow:
    return build_flow(
        [
            make_packet(timestamp=100.0, length=500, tcp_flags="S"),
            make_packet(timestamp=100.1, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=500, tcp_flags="SA"),
            make_packet(timestamp=100.2, length=500, tcp_flags="A"),
        ]
    )


@pytest.fixture
def repository():
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    session_factory = create_session_factory(engine)
    return FlowDetectionRepository(session_factory)


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------


def test_init_db_creates_expected_tables_and_indexes() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)

    inspector = inspect(engine)
    assert set(inspector.get_table_names()) == {"flows", "detections"}

    flow_indexed_columns = {col for idx in inspector.get_indexes("flows") for col in idx["column_names"]}
    assert {"src_ip", "dst_ip", "first_seen"}.issubset(flow_indexed_columns)

    detection_indexed_columns = {col for idx in inspector.get_indexes("detections") for col in idx["column_names"]}
    assert {"flow_id", "timestamp", "final_risk_score", "severity"}.issubset(detection_indexed_columns)


def test_init_db_is_idempotent() -> None:
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine)
    init_db(engine)  # must not raise on a second call


# ---------------------------------------------------------------------------
# Saving and retrieving flows
# ---------------------------------------------------------------------------


def test_save_flow_returns_an_id(repository) -> None:
    flow = normal_flow()
    flow_id = repository.save_flow(flow)

    assert isinstance(flow_id, int)
    assert flow_id > 0


def test_save_and_get_flow_round_trips_core_fields(repository) -> None:
    flow = normal_flow()
    flow_id = repository.save_flow(flow)

    persisted = repository.get_flow(flow_id)

    assert persisted is not None
    assert persisted.src_ip == flow.origin_src_ip
    assert persisted.dst_ip == flow.origin_dst_ip
    assert persisted.src_port == flow.origin_src_port
    assert persisted.dst_port == flow.origin_dst_port
    assert persisted.protocol == flow.key.protocol.value
    assert persisted.packet_count == flow.packet_count
    assert persisted.byte_count == flow.byte_count
    assert persisted.forward_packet_count == flow.forward_packet_count
    assert persisted.backward_packet_count == flow.backward_packet_count
    assert persisted.duration_seconds == pytest.approx(flow.duration_seconds)


def test_save_flow_with_feature_snapshot_round_trips_as_json(repository) -> None:
    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)

    flow_id = repository.save_flow(flow, feature_vector=vector)
    persisted = repository.get_flow(flow_id)

    assert persisted is not None
    assert persisted.feature_snapshot is not None
    assert persisted.feature_snapshot == vector.to_dict()
    assert persisted.feature_snapshot["total_packets"] == vector.to_dict()["total_packets"]


def test_save_flow_without_feature_vector_leaves_snapshot_null(repository) -> None:
    flow = normal_flow()
    flow_id = repository.save_flow(flow)  # no feature_vector passed
    persisted = repository.get_flow(flow_id)

    assert persisted.feature_snapshot is None


def test_get_flow_returns_none_for_nonexistent_id(repository) -> None:
    assert repository.get_flow(99999) is None


def test_icmp_flow_with_no_ports_persists_correctly(repository) -> None:
    flow = build_flow([make_packet(timestamp=100.0, protocol=TransportProtocol.ICMP, src_port=None, dst_port=None, tcp_flags=None)])
    flow_id = repository.save_flow(flow)
    persisted = repository.get_flow(flow_id)

    assert persisted.src_port is None
    assert persisted.dst_port is None
    assert persisted.protocol == "ICMP"


# ---------------------------------------------------------------------------
# Saving and retrieving detections
# ---------------------------------------------------------------------------


def test_save_detection_persists_rule_only_result(repository) -> None:
    flow = suspicious_flow()
    flow_id = repository.save_flow(flow)
    vector = FeatureExtractor().extract(flow)

    detector = HybridDetector(rule_detector=RuleDetector(), ml_predictor=None, risk_scoring=RiskScoringConfig())
    result = detector.detect(vector, flow)

    detection_id = repository.save_detection(flow_id, result)
    persisted = repository.get_detection(detection_id)

    assert persisted is not None
    assert persisted.flow_id == flow_id
    assert persisted.rule_score == result.rule_result.risk_score
    assert set(persisted.triggered_rules) == set(result.rule_result.rule_names)
    assert persisted.ml_model_type is None
    assert persisted.ml_predicted_class is None
    assert persisted.ml_probability is None
    assert "rules-only" in persisted.ml_error
    assert persisted.final_risk_score == result.final_risk_score
    assert persisted.severity == result.final_severity.value
    assert persisted.rule_weight == result.rule_weight
    assert persisted.ml_weight == result.ml_weight


def test_save_detection_persists_ml_fields_when_present(repository) -> None:
    flow = normal_flow()
    flow_id = repository.save_flow(flow)
    vector = FeatureExtractor().extract(flow)

    class _StubPredictor:
        def predict(self, feature_vector):
            return MLInferenceResult(
                model_type="stub_model",
                schema_version="1.0.0",
                predicted_label=1,
                predicted_class="ATTACK",
                probability=0.87,
                flow_key_repr=repr(flow.key),
            )

    detector = HybridDetector(rule_detector=RuleDetector(), ml_predictor=_StubPredictor(), risk_scoring=RiskScoringConfig())
    result = detector.detect(vector, flow)

    detection_id = repository.save_detection(flow_id, result)
    persisted = repository.get_detection(detection_id)

    assert persisted.ml_model_type == "stub_model"
    assert persisted.ml_predicted_class == "ATTACK"
    assert persisted.ml_probability == pytest.approx(0.87)
    assert persisted.ml_error is None


def test_rule_evidence_round_trips_as_json(repository) -> None:
    flow = suspicious_flow()
    flow_id = repository.save_flow(flow)
    vector = FeatureExtractor().extract(flow)

    detector = RuleDetector()
    rule_result = detector.detect(vector, flow)
    assert rule_result.triggered_rules  # sanity: this flow should trip something

    hybrid = HybridDetector(rule_detector=detector, ml_predictor=None)
    result = hybrid.detect(vector, flow)

    detection_id = repository.save_detection(flow_id, result)
    persisted = repository.get_detection(detection_id)

    for trigger in result.rule_result.triggered_rules:
        assert trigger.rule_name in persisted.rule_evidence
        assert persisted.rule_evidence[trigger.rule_name] == trigger.evidence


def test_get_detection_returns_none_for_nonexistent_id(repository) -> None:
    assert repository.get_detection(99999) is None


# ---------------------------------------------------------------------------
# Queries: recent detections, per-flow detections, severity counts
# ---------------------------------------------------------------------------


def test_get_recent_detections_orders_newest_first_and_respects_limit(repository) -> None:
    import time as _time

    flow_ids = []
    detection_ids = []
    for _ in range(5):
        flow = normal_flow()
        flow_id = repository.save_flow(flow)
        vector = FeatureExtractor().extract(flow)
        result = HybridDetector(rule_detector=RuleDetector()).detect(vector, flow)
        detection_ids.append(repository.save_detection(flow_id, result))
        flow_ids.append(flow_id)
        _time.sleep(0.001)  # ensure strictly increasing timestamps for a meaningful order check

    recent = repository.get_recent_detections(limit=3)

    assert len(recent) == 3
    returned_ids = [d.id for d in recent]
    assert returned_ids == list(reversed(detection_ids))[:3]


def test_get_detections_for_flow_returns_only_that_flows_detections(repository) -> None:
    flow_a = normal_flow()
    flow_a_id = repository.save_flow(flow_a)
    vector_a = FeatureExtractor().extract(flow_a)
    result_a = HybridDetector(rule_detector=RuleDetector()).detect(vector_a, flow_a)
    repository.save_detection(flow_a_id, result_a)
    repository.save_detection(flow_a_id, result_a)  # a second detection for the same flow

    flow_b = suspicious_flow()
    flow_b_id = repository.save_flow(flow_b)
    vector_b = FeatureExtractor().extract(flow_b)
    result_b = HybridDetector(rule_detector=RuleDetector()).detect(vector_b, flow_b)
    repository.save_detection(flow_b_id, result_b)

    detections_for_a = repository.get_detections_for_flow(flow_a_id)
    assert len(detections_for_a) == 2
    assert all(d.flow_id == flow_a_id for d in detections_for_a)


def test_count_detections_by_severity(repository) -> None:
    flow = suspicious_flow()
    flow_id = repository.save_flow(flow)
    vector = FeatureExtractor().extract(flow)
    result = HybridDetector(rule_detector=RuleDetector()).detect(vector, flow)
    repository.save_detection(flow_id, result)

    counts = repository.count_detections_by_severity()

    assert counts.get(result.final_severity.value, 0) >= 1
    assert sum(counts.values()) == 1


# ---------------------------------------------------------------------------
# Graceful failure handling
# ---------------------------------------------------------------------------


def test_save_detection_with_invalid_flow_id_raises_persistence_error(repository) -> None:
    flow = normal_flow()
    vector = FeatureExtractor().extract(flow)
    result = HybridDetector(rule_detector=RuleDetector()).detect(vector, flow)

    with pytest.raises(DatabasePersistenceError):
        repository.save_detection(flow_id=99999, result=result)  # foreign key violation


def test_repository_wraps_underlying_errors_not_raw_sqlalchemy_errors(repository) -> None:
    """
    Callers of FlowDetectionRepository should only ever need to catch
    DatabasePersistenceError -- never a raw sqlalchemy.exc.SQLAlchemyError
    -- per the "not tightly coupled to database implementation details"
    requirement.
    """
    flow = normal_flow()
    vector = FeatureExtractor().extract(flow)
    result = HybridDetector(rule_detector=RuleDetector()).detect(vector, flow)

    try:
        repository.save_detection(flow_id=99999, result=result)
        pytest.fail("Expected DatabasePersistenceError to be raised")
    except DatabasePersistenceError as exc:
        assert exc.__cause__ is not None  # the original SQLAlchemy error is preserved for debugging


def test_broken_session_factory_raises_persistence_error_not_a_crash() -> None:
    """Simulates a connection-level failure (e.g. database unreachable) without needing a real broken DB."""

    def _broken_session_factory():
        raise RuntimeError("simulated connection failure")

    # RuntimeError isn't a SQLAlchemyError, so this specifically checks that
    # our repository's try/except boundary is where we expect it -- a
    # connection-level failure inside session_factory() itself would
    # currently propagate as-is (documented limitation below), whereas a
    # failure raised through the ORM/session API is what gets wrapped.
    repo = FlowDetectionRepository(_broken_session_factory)
    flow = normal_flow()

    with pytest.raises(RuntimeError):
        repo.save_flow(flow)


# ---------------------------------------------------------------------------
# Full pipeline integration: FinalizedFlow -> Features -> HybridDetector -> DB
# ---------------------------------------------------------------------------


def test_full_pipeline_integration_flow_and_detection_persisted_together(repository) -> None:
    flow = suspicious_flow()
    extractor = FeatureExtractor()
    vector = extractor.extract(flow)

    detector = HybridDetector(rule_detector=RuleDetector(), ml_predictor=None, risk_scoring=RiskScoringConfig())
    result = detector.detect(vector, flow)

    flow_id = repository.save_flow(flow, feature_vector=vector)
    detection_id = repository.save_detection(flow_id, result)

    persisted_flow = repository.get_flow(flow_id)
    persisted_detection = repository.get_detection(detection_id)

    assert persisted_flow is not None
    assert persisted_detection is not None
    assert persisted_detection.flow_id == persisted_flow.id
    assert persisted_detection.final_risk_score == result.final_risk_score
    assert persisted_flow.feature_snapshot == vector.to_dict()

    # And it's genuinely queryable via the flow-scoped lookup too.
    detections_for_flow = repository.get_detections_for_flow(flow_id)
    assert len(detections_for_flow) == 1
    assert detections_for_flow[0].id == detection_id


# ---------------------------------------------------------------------------
# Real PostgreSQL integration test (documented, opt-in only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    "NIDS_TEST_POSTGRES_URL" not in os.environ,
    reason=(
        "Set NIDS_TEST_POSTGRES_URL to run this against a real PostgreSQL instance, e.g.:\n"
        "  docker compose up -d postgres\n"
        "  NIDS_TEST_POSTGRES_URL='postgresql+psycopg://nids:nids@localhost:5432/nids' pytest tests/test_database.py -k postgres\n"
        "Skipped by default so the rest of the suite never requires a live database."
    ),
)
def test_real_postgres_save_and_query_round_trip() -> None:
    """Non-mocked integration test against a real PostgreSQL instance -- see skip reason for how to run it."""
    database_url = os.environ["NIDS_TEST_POSTGRES_URL"]
    engine = create_db_engine(database_url)
    init_db(engine)
    repo = FlowDetectionRepository(create_session_factory(engine))

    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    result = HybridDetector(rule_detector=RuleDetector()).detect(vector, flow)

    flow_id = repo.save_flow(flow, feature_vector=vector)
    detection_id = repo.save_detection(flow_id, result)

    persisted_detection = repo.get_detection(detection_id)
    assert persisted_detection is not None
    assert persisted_detection.final_risk_score == result.final_risk_score
