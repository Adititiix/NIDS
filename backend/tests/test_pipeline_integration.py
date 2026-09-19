"""
End-to-end integration tests across Phases 2-5.

Two tests here:

1. A synthetic-data integration test (FinalizedFlow -> FeatureExtractor ->
   RuleDetector -> DetectionResult) that always runs, requires no network
   access, and is what CI should rely on.

2. A REAL, non-mocked integration test that exercises actual Scapy packet
   capture on the loopback interface, feeding real captured packets through
   the full pipeline: PacketCapture -> FlowAggregator -> FeatureExtractor ->
   RuleDetector. This mirrors the smoke tests done manually in Phases 2-4.
   It's marked so it can be skipped in environments without capture
   privileges or a loopback interface, without affecting the rest of the
   suite.
"""

from __future__ import annotations

import platform
import subprocess
import time

import pytest

from app.capture.models import PacketMetadata, TransportProtocol
from app.capture.packet_capture import PacketCapture
from app.config import DetectionThresholds, RiskScoringConfig
from app.detection.rule_engine import RuleDetector
from app.features.feature_extractor import FeatureExtractor
from app.features.schema import FEATURE_COUNT
from app.flows.flow_aggregator import FlowAggregator
from app.flows.models import Flow, FinalizedFlow


def make_packet(
    *,
    timestamp: float,
    src_ip: str = "10.0.2.15",
    dst_ip: str = "10.0.2.1",
    src_port: int | None = 51234,
    dst_port: int | None = 4444,
    protocol: TransportProtocol = TransportProtocol.TCP,
    length: int = 100,
    tcp_flags: str | None = None,
) -> PacketMetadata:
    return PacketMetadata(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        length=length,
        tcp_flags=tcp_flags,
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


# ---------------------------------------------------------------------------
# Synthetic-data integration test (always runs)
# ---------------------------------------------------------------------------


def test_finalized_flow_to_feature_extractor_to_rule_detector_to_detection_result() -> None:
    """
    The full Phase 3 -> 4 -> 5 pipeline on a synthetic, clearly-suspicious
    flow: a SYN-heavy burst to a configured suspicious port.
    """
    packets = [make_packet(timestamp=100.0 + i * 0.005, tcp_flags="S") for i in range(30)]
    flow: FinalizedFlow = build_flow(packets)

    feature_vector = FeatureExtractor().extract(flow)
    assert len(feature_vector.values) == FEATURE_COUNT

    thresholds = DetectionThresholds(
        high_packet_rate_pps=50.0,
        syn_heavy_ratio_threshold=0.8,
        syn_heavy_min_packets=5,
        short_flow_max_duration_seconds=1.0,
        short_flow_min_packets=20,
        suspicious_destination_ports=(4444,),
    )
    detector = RuleDetector(thresholds=thresholds, risk_scoring=RiskScoringConfig())
    result = detector.detect(feature_vector, flow)

    assert result.is_suspicious is True
    assert result.risk_score > 0
    assert "syn_heavy" in result.rule_names
    assert "suspicious_destination_port" in result.rule_names
    assert result.flow_key_repr == repr(flow.key)


def test_normal_flow_through_full_pipeline_is_not_suspicious() -> None:
    """The same pipeline on ordinary traffic should not flag anything."""
    packets = [
        make_packet(timestamp=100.0, dst_port=443, length=500, tcp_flags="S"),
        make_packet(timestamp=100.1, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=500, tcp_flags="SA"),
        make_packet(timestamp=100.2, dst_port=443, length=500, tcp_flags="A"),
    ]
    flow = build_flow(packets)
    feature_vector = FeatureExtractor().extract(flow)
    detector = RuleDetector(thresholds=DetectionThresholds(), risk_scoring=RiskScoringConfig())

    result = detector.detect(feature_vector, flow)

    assert result.is_suspicious is False
    assert result.risk_score == 0


# ---------------------------------------------------------------------------
# Real, non-mocked capture integration test
# ---------------------------------------------------------------------------


def _loopback_capture_is_available() -> bool:
    """Best-effort check so this test skips cleanly instead of failing on environments without it."""
    if platform.system() == "Windows":
        return False  # loopback capture semantics differ enough under Npcap to skip this specific smoke test
    try:
        from scapy.all import conf

        return "lo" in [iface for iface in conf.ifaces.data.keys()] if hasattr(conf, "ifaces") else True
    except Exception:
        return False


@pytest.mark.skipif(not _loopback_capture_is_available(), reason="Loopback packet capture not available/testable in this environment.")
def test_real_capture_through_flows_features_and_detection_on_loopback() -> None:
    """
    Non-mocked: real Scapy capture on 'lo', real flow aggregation, real
    feature extraction, real rule evaluation -- no synthetic PacketMetadata
    anywhere in this test. Does not assert about attack detection (real
    background loopback traffic is not attack traffic); it only asserts
    that the full pipeline runs end-to-end without error and produces a
    structurally valid DetectionResult.
    """
    capture = PacketCapture(interface="lo")
    aggregator = FlowAggregator(source_queue=capture.queue, flow_timeout_seconds=2.0)
    extractor = FeatureExtractor()
    detector = RuleDetector(thresholds=DetectionThresholds(), risk_scoring=RiskScoringConfig())

    capture.start()
    aggregator.start()
    try:
        subprocess.run(["curl", "-s", "-o", "/dev/null", "http://127.0.0.1:9/"], timeout=1, check=False)
        time.sleep(4)  # allow the 2s flow timeout to expire the flow
    finally:
        aggregator.stop()
        capture.stop()

    finalized_flows = aggregator.get_expired_flows()
    if not finalized_flows:
        pytest.skip("No loopback traffic was captured in this environment -- cannot exercise the real pipeline.")

    for flow in finalized_flows:
        vector = extractor.extract(flow)
        assert len(vector.values) == FEATURE_COUNT
        result = detector.detect(vector, flow)
        assert result.risk_score >= 0
        assert result.flow_key_repr == repr(flow.key)
