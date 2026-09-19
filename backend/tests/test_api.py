"""
Unit tests for Phase 10 (FastAPI REST API + WebSocket layer).

Every test overrides `get_repository` (via FastAPI's `app.dependency_overrides`)
with a repository backed by a fresh in-memory SQLite database, so no live
PostgreSQL instance is required and each test starts from a clean slate.
The existing `/api/health` endpoint is verified unchanged from Phase 1.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_repository
from app.capture.models import PacketMetadata, TransportProtocol
from app.database.repository import FlowDetectionRepository
from app.database.session import create_db_engine, create_session_factory, init_db
from app.detection.hybrid_detector import HybridDetector
from app.detection.rule_engine import RuleDetector
from app.features.feature_extractor import FeatureExtractor
from app.flows.models import Flow, FinalizedFlow
from app.main import app
from app.ml.inference import MLInferenceResult
from app.websocket.manager import ConnectionManager


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


def suspicious_flow(dst_port: int = 4444) -> FinalizedFlow:
    packets = [make_packet(timestamp=100.0 + i * 0.005, dst_port=dst_port, tcp_flags="S") for i in range(30)]
    return build_flow(packets)


def normal_flow() -> FinalizedFlow:
    return build_flow(
        [
            make_packet(timestamp=100.0, length=500, tcp_flags="S"),
            make_packet(timestamp=100.1, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=500, tcp_flags="SA"),
            make_packet(timestamp=100.2, length=500, tcp_flags="A"),
        ]
    )


class _StubMLPredictor:
    def __init__(self, probability: float = 0.9) -> None:
        self._probability = probability

    def predict(self, feature_vector):
        return MLInferenceResult(
            model_type="stub_model",
            schema_version="1.0.0",
            predicted_label=1,
            predicted_class="ATTACK",
            probability=self._probability,
            flow_key_repr="stub",
        )


@pytest.fixture
def repository() -> FlowDetectionRepository:
    # StaticPool is essential here: without it, SQLite's ":memory:" database
    # is per-connection, and the TestClient below runs the app's request
    # handlers in a different thread than this fixture -- each thread would
    # otherwise get its OWN separate (empty) in-memory database. StaticPool
    # forces every connection from this engine to share the single
    # underlying SQLite connection, so the app sees the same data this
    # fixture seeds.
    engine = create_db_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return FlowDetectionRepository(create_session_factory(engine))


@pytest.fixture
def client(repository):
    app.dependency_overrides[get_repository] = lambda: repository
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def seed_flow_and_detection(repository: FlowDetectionRepository, *, flow: FinalizedFlow, ml_predictor=None) -> tuple[int, int]:
    vector = FeatureExtractor().extract(flow)
    detector = HybridDetector(rule_detector=RuleDetector(), ml_predictor=ml_predictor)
    result = detector.detect(vector, flow)
    flow_id = repository.save_flow(flow, feature_vector=vector)
    detection_id = repository.save_detection(flow_id, result)
    return flow_id, detection_id


# ---------------------------------------------------------------------------
# Existing /api/health must be preserved exactly
# ---------------------------------------------------------------------------


def test_existing_health_endpoint_is_unchanged(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "environment" in body


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------


def test_status_endpoint_reports_database_reachable_and_counts(client: TestClient, repository) -> None:
    seed_flow_and_detection(repository, flow=normal_flow())

    response = client.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["database_reachable"] is True
    assert body["total_flows"] == 1
    assert body["total_detections"] == 1
    assert "environment" in body
    assert "capture_interface" in body
    assert "ml_active_model_name" in body


def test_status_endpoint_reports_database_unreachable_gracefully(client: TestClient, repository) -> None:
    from app.database.repository import DatabasePersistenceError

    class _BrokenRepository:
        def count_flows(self):
            raise DatabasePersistenceError("simulated outage")

        def count_detections(self):
            raise DatabasePersistenceError("simulated outage")

    app.dependency_overrides[get_repository] = lambda: _BrokenRepository()
    try:
        response = client.get("/api/status")
    finally:
        app.dependency_overrides[get_repository] = lambda: repository

    assert response.status_code == 200  # status endpoint itself still responds
    body = response.json()
    assert body["database_reachable"] is False
    assert body["total_flows"] == 0
    assert body["total_detections"] == 0


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------


def test_list_flows_empty(client: TestClient) -> None:
    response = client.get("/api/flows")
    assert response.status_code == 200
    assert response.json() == []


def test_list_flows_returns_seeded_flows(client: TestClient, repository) -> None:
    seed_flow_and_detection(repository, flow=normal_flow())
    seed_flow_and_detection(repository, flow=suspicious_flow())

    response = client.get("/api/flows")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert {"id", "src_ip", "dst_ip", "protocol", "packet_count"}.issubset(body[0].keys())


def test_list_flows_respects_limit(client: TestClient, repository) -> None:
    for _ in range(5):
        seed_flow_and_detection(repository, flow=normal_flow())

    response = client.get("/api/flows", params={"limit": 2})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_flow_by_id(client: TestClient, repository) -> None:
    flow_id, _ = seed_flow_and_detection(repository, flow=normal_flow())

    response = client.get(f"/api/flows/{flow_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == flow_id
    assert body["feature_snapshot"] is not None  # we passed a feature_vector when seeding


def test_get_flow_not_found_returns_404(client: TestClient) -> None:
    response = client.get("/api/flows/99999")
    assert response.status_code == 404


def test_get_flow_alerts(client: TestClient, repository) -> None:
    flow_id, detection_id = seed_flow_and_detection(repository, flow=suspicious_flow())

    response = client.get(f"/api/flows/{flow_id}/alerts")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == detection_id


def test_get_flow_alerts_not_found_returns_404(client: TestClient) -> None:
    response = client.get("/api/flows/99999/alerts")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


def test_list_alerts_empty(client: TestClient) -> None:
    response = client.get("/api/alerts")
    assert response.status_code == 200
    assert response.json() == []


def test_list_alerts_includes_flow_fields_and_detection_source(client: TestClient, repository) -> None:
    seed_flow_and_detection(repository, flow=suspicious_flow(), ml_predictor=_StubMLPredictor())

    response = client.get("/api/alerts")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    alert = body[0]
    assert alert["src_ip"] == "10.0.2.15"
    assert alert["dst_ip"] == "10.0.2.1"
    assert alert["protocol"] == "TCP"
    assert alert["detection_source"] == "hybrid"  # rules fired AND ML said ATTACK


def test_list_alerts_detection_source_rule_only(client: TestClient, repository) -> None:
    seed_flow_and_detection(repository, flow=suspicious_flow(), ml_predictor=None)

    body = client.get("/api/alerts").json()
    assert body[0]["detection_source"] == "rule"


def test_list_alerts_detection_source_ml_only(client: TestClient, repository) -> None:
    seed_flow_and_detection(repository, flow=normal_flow(), ml_predictor=_StubMLPredictor())

    body = client.get("/api/alerts").json()
    assert body[0]["detection_source"] == "ml"


def test_list_alerts_filters_by_severity(client: TestClient, repository) -> None:
    seed_flow_and_detection(repository, flow=normal_flow())  # NONE severity
    seed_flow_and_detection(repository, flow=suspicious_flow())  # should be HIGH/CRITICAL

    all_alerts = client.get("/api/alerts").json()
    assert len(all_alerts) == 2

    filtered = client.get("/api/alerts", params={"severity": "NONE"}).json()
    assert len(filtered) == 1
    assert filtered[0]["severity"] == "NONE"


def test_list_alerts_rejects_invalid_severity(client: TestClient) -> None:
    response = client.get("/api/alerts", params={"severity": "SUPER_BAD"})
    assert response.status_code == 422


def test_list_alerts_rejects_invalid_limit(client: TestClient) -> None:
    response = client.get("/api/alerts", params={"limit": 0})
    assert response.status_code == 422

    response = client.get("/api/alerts", params={"limit": 10_000})
    assert response.status_code == 422


def test_get_alert_detail_includes_rule_evidence(client: TestClient, repository) -> None:
    _flow_id, detection_id = seed_flow_and_detection(repository, flow=suspicious_flow())

    response = client.get(f"/api/alerts/{detection_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == detection_id
    assert body["triggered_rules"]
    assert body["rule_evidence"]
    assert all(name in body["rule_evidence"] for name in body["triggered_rules"])


def test_get_alert_not_found_returns_404(client: TestClient) -> None:
    response = client.get("/api/alerts/99999")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_detection_stats_endpoint(client: TestClient, repository) -> None:
    seed_flow_and_detection(repository, flow=normal_flow())
    seed_flow_and_detection(repository, flow=suspicious_flow())

    response = client.get("/api/stats/detections")
    assert response.status_code == 200
    body = response.json()
    assert body["total_detections"] == 2
    assert sum(body["severity_counts"].values()) == 2


def test_traffic_stats_endpoint(client: TestClient, repository) -> None:
    seed_flow_and_detection(repository, flow=normal_flow())
    seed_flow_and_detection(repository, flow=suspicious_flow())

    response = client.get("/api/stats/traffic")
    assert response.status_code == 200
    body = response.json()
    assert body["total_flows"] == 2
    assert body["total_packets"] > 0
    assert body["protocol_counts"].get("TCP") == 2


def test_traffic_stats_empty_database(client: TestClient) -> None:
    response = client.get("/api/stats/traffic")
    assert response.status_code == 200
    body = response.json()
    assert body["total_flows"] == 0
    assert body["total_packets"] == 0
    assert body["average_duration_seconds"] == 0.0


# ---------------------------------------------------------------------------
# WebSocket: connection lifecycle, broadcast, multiple clients, disconnects
# ---------------------------------------------------------------------------


def test_websocket_client_can_connect(client: TestClient) -> None:
    with client.websocket_connect("/ws/alerts"):
        pass  # connecting and cleanly closing must not raise


@pytest.mark.anyio
async def test_connection_manager_broadcasts_to_connected_clients() -> None:
    manager = ConnectionManager()
    sent_messages: list[dict] = []

    class _FakeWebSocket:
        async def accept(self):
            pass

        async def send_json(self, message):
            sent_messages.append(message)

    ws = _FakeWebSocket()
    await manager.connect(ws)
    await manager.broadcast({"event": "new_detection", "severity": "HIGH"})

    assert sent_messages == [{"event": "new_detection", "severity": "HIGH"}]


@pytest.mark.anyio
async def test_connection_manager_drops_dead_connections_without_raising() -> None:
    manager = ConnectionManager()

    class _WorkingWebSocket:
        async def accept(self):
            pass

        async def send_json(self, message):
            pass

    class _BrokenWebSocket:
        async def accept(self):
            pass

        async def send_json(self, message):
            raise ConnectionError("client vanished")

    working = _WorkingWebSocket()
    broken = _BrokenWebSocket()
    await manager.connect(working)
    await manager.connect(broken)
    assert manager.connection_count == 2

    await manager.broadcast({"event": "test"})  # must not raise despite `broken` failing

    assert manager.connection_count == 1  # broken connection was dropped


@pytest.mark.anyio
async def test_connection_manager_handles_multiple_clients_and_disconnect() -> None:
    manager = ConnectionManager()

    class _FakeWebSocket:
        def __init__(self):
            self.received = []

        async def accept(self):
            pass

        async def send_json(self, message):
            self.received.append(message)

    clients = [_FakeWebSocket() for _ in range(3)]
    for ws in clients:
        await manager.connect(ws)
    assert manager.connection_count == 3

    await manager.disconnect(clients[0])
    assert manager.connection_count == 2

    await manager.broadcast({"event": "new_detection"})
    assert clients[0].received == []  # disconnected before the broadcast
    assert clients[1].received == [{"event": "new_detection"}]
    assert clients[2].received == [{"event": "new_detection"}]


def test_websocket_disconnect_is_handled_cleanly(client: TestClient) -> None:
    from app.websocket.manager import connection_manager

    initial_count = connection_manager.connection_count
    with client.websocket_connect("/ws/alerts"):
        assert connection_manager.connection_count == initial_count + 1

    import time

    deadline = time.time() + 2.0
    while connection_manager.connection_count > initial_count and time.time() < deadline:
        time.sleep(0.01)
    assert connection_manager.connection_count == initial_count
