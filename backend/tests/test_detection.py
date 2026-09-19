"""
Unit tests for Phase 5 rule-based detection.

Uses the same synthetic-FinalizedFlow-building pattern as
test_feature_extractor.py (Phase 4) -- no live traffic, no Scapy needed.
Thresholds are passed explicitly to each rule/detector under test rather
than relying on global `settings` defaults, so these tests stay correct
even if the operational defaults in app.config change later.
"""

from __future__ import annotations

from app.capture.models import PacketMetadata, TransportProtocol
from app.config import DetectionThresholds, RiskScoringConfig
from app.detection.models import Severity
from app.detection.rule_engine import RuleDetector
from app.detection.rules.high_byte_rate import HighByteRateRule
from app.detection.rules.high_packet_rate import HighPacketRateRule
from app.detection.rules.short_high_volume import ShortHighVolumeRule
from app.detection.rules.suspicious_port import SuspiciousPortRule
from app.detection.rules.syn_heavy import SynHeavyRule
from app.detection.rules.tcp_flag_anomaly import TcpFlagAnomalyRule
from app.features.feature_extractor import FeatureExtractor
from app.flows.models import Flow, FinalizedFlow


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
        if i == 0:
            flow = Flow.start(pkt)
        else:
            assert flow is not None
            flow.add_packet(pkt)
    assert flow is not None
    return flow.finalize()


def default_thresholds(**overrides: object) -> DetectionThresholds:
    """A DetectionThresholds instance with test-friendly explicit values, minus env/file loading noise."""
    base = dict(
        high_packet_rate_pps=100.0,
        high_byte_rate_bps=1_000_000.0,
        rate_rule_min_packets=5,
        syn_heavy_ratio_threshold=0.8,
        syn_heavy_min_packets=5,
        short_flow_max_duration_seconds=0.5,
        short_flow_min_packets=20,
        flag_anomaly_min_packets=2,
        suspicious_destination_ports=(23, 445, 3389, 4444, 31337),
    )
    base.update(overrides)
    return DetectionThresholds(**base)


extractor = FeatureExtractor()


# ---------------------------------------------------------------------------
# Normal traffic -> no alert
# ---------------------------------------------------------------------------


def test_normal_low_rate_flow_triggers_no_rules() -> None:
    packets = [
        make_packet(timestamp=100.0, length=500, tcp_flags="S"),
        make_packet(timestamp=100.5, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=500, tcp_flags="SA"),
        make_packet(timestamp=101.0, length=500, tcp_flags="A"),
    ]
    flow = build_flow(packets)
    vector = extractor.extract(flow)

    detector = RuleDetector(thresholds=default_thresholds(), risk_scoring=RiskScoringConfig())
    result = detector.detect(vector, flow)

    assert result.is_suspicious is False
    assert result.risk_score == 0
    assert result.severity == Severity.NONE
    assert result.triggered_rules == ()


# ---------------------------------------------------------------------------
# High packet rate
# ---------------------------------------------------------------------------


def test_high_packet_rate_rule_fires_above_threshold() -> None:
    thresholds = default_thresholds(high_packet_rate_pps=50.0)
    rule = HighPacketRateRule(thresholds)

    # 100 packets over 1 second = 100 pkt/s > 50 pkt/s threshold
    packets = [make_packet(timestamp=100.0 + i * 0.01) for i in range(100)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    trigger = rule.evaluate(features, flow)
    assert trigger is not None
    assert trigger.rule_name == "high_packet_rate"
    assert trigger.score == HighPacketRateRule.base_score


def test_high_packet_rate_rule_does_not_fire_below_threshold() -> None:
    thresholds = default_thresholds(high_packet_rate_pps=1000.0)
    rule = HighPacketRateRule(thresholds)

    packets = [make_packet(timestamp=100.0 + i * 0.1) for i in range(5)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None


# ---------------------------------------------------------------------------
# High byte rate
# ---------------------------------------------------------------------------


def test_high_byte_rate_rule_fires_above_threshold() -> None:
    thresholds = default_thresholds(high_byte_rate_bps=1000.0)
    rule = HighByteRateRule(thresholds)

    # 5 packets (>= rate_rule_min_packets) of 1500 bytes over 1 second = 7500 bytes/s > 1000 threshold
    packets = [make_packet(timestamp=100.0 + i * 0.25, length=1500) for i in range(5)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    trigger = rule.evaluate(features, flow)
    assert trigger is not None
    assert trigger.rule_name == "high_byte_rate"


# ---------------------------------------------------------------------------
# rate_rule_min_packets guard (Windows real-traffic false-positive fix):
# a handful of packets separated by microseconds can produce a
# mathematically enormous but statistically meaningless rate. These rules
# must withhold judgment below the configured minimum sample size.
# ---------------------------------------------------------------------------


def test_high_packet_rate_rule_does_not_fire_below_min_packets_even_with_enormous_rate() -> None:
    thresholds = default_thresholds(high_packet_rate_pps=100.0, rate_rule_min_packets=5)
    rule = HighPacketRateRule(thresholds)

    # 2 packets ~60 microseconds apart -> rate ~33,000 pkt/s, but only 2 packets (< 5).
    packets = [make_packet(timestamp=100.0), make_packet(timestamp=100.00006)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()
    assert features["total_packets_per_second"] > 10_000  # confirm the rate really is enormous

    assert rule.evaluate(features, flow) is None


def test_high_byte_rate_rule_does_not_fire_below_min_packets_even_with_enormous_rate() -> None:
    thresholds = default_thresholds(high_byte_rate_bps=1_000_000.0, rate_rule_min_packets=5)
    rule = HighByteRateRule(thresholds)

    # 2 packets ~60 microseconds apart, 1000 bytes each -> rate ~33 MB/s, but only 2 packets (< 5).
    packets = [make_packet(timestamp=100.0, length=1000), make_packet(timestamp=100.00006, length=1000)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()
    assert features["total_bytes_per_second"] > 1_000_000  # confirm the rate really is enormous

    assert rule.evaluate(features, flow) is None


def test_high_packet_rate_rule_fires_at_exactly_min_packets_when_above_threshold() -> None:
    thresholds = default_thresholds(high_packet_rate_pps=100.0, rate_rule_min_packets=5)
    rule = HighPacketRateRule(thresholds)

    # Exactly 5 packets (== rate_rule_min_packets) over 0.01s -> 500 pkt/s > 100 threshold.
    packets = [make_packet(timestamp=100.0 + i * 0.0025) for i in range(5)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()
    assert features["total_packets"] == 5

    trigger = rule.evaluate(features, flow)
    assert trigger is not None
    assert trigger.rule_name == "high_packet_rate"


def test_high_byte_rate_rule_fires_at_exactly_min_packets_when_above_threshold() -> None:
    thresholds = default_thresholds(high_byte_rate_bps=1000.0, rate_rule_min_packets=5)
    rule = HighByteRateRule(thresholds)

    # Exactly 5 packets (== rate_rule_min_packets) of 1500 bytes over 1 second = 7500 bytes/s > 1000 threshold.
    packets = [make_packet(timestamp=100.0 + i * 0.25, length=1500) for i in range(5)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()
    assert features["total_packets"] == 5

    trigger = rule.evaluate(features, flow)
    assert trigger is not None
    assert trigger.rule_name == "high_byte_rate"


def test_high_byte_rate_rule_does_not_fire_below_threshold() -> None:
    thresholds = default_thresholds(high_byte_rate_bps=10_000_000.0)
    rule = HighByteRateRule(thresholds)

    packets = [make_packet(timestamp=100.0, length=100), make_packet(timestamp=101.0, length=100)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None


# ---------------------------------------------------------------------------
# SYN-heavy
# ---------------------------------------------------------------------------


def test_syn_heavy_rule_fires_on_syn_dominant_flow() -> None:
    thresholds = default_thresholds(syn_heavy_ratio_threshold=0.8, syn_heavy_min_packets=5)
    rule = SynHeavyRule(thresholds)

    # 9 SYNs, 1 unrelated ACK-only packet on the SAME 5-tuple -> 90% SYN ratio
    packets = [make_packet(timestamp=100.0 + i, tcp_flags="S") for i in range(9)]
    packets.append(make_packet(timestamp=110.0, tcp_flags="A"))
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    trigger = rule.evaluate(features, flow)
    assert trigger is not None
    assert trigger.rule_name == "syn_heavy"
    assert features["syn_count"] == 9.0
    assert features["ack_count"] == 1.0


def test_syn_heavy_rule_does_not_fire_below_min_packets() -> None:
    thresholds = default_thresholds(syn_heavy_ratio_threshold=0.5, syn_heavy_min_packets=10)
    rule = SynHeavyRule(thresholds)

    # Only 3 packets, all SYN (100% ratio) but below the min_packets floor
    packets = [make_packet(timestamp=100.0 + i, tcp_flags="S") for i in range(3)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None


def test_syn_heavy_rule_does_not_fire_on_normal_ratio() -> None:
    thresholds = default_thresholds(syn_heavy_ratio_threshold=0.8, syn_heavy_min_packets=5)
    rule = SynHeavyRule(thresholds)

    # 2 SYN out of 10 packets = 20% ratio, well under threshold
    packets = [make_packet(timestamp=100.0, tcp_flags="S")]
    packets += [make_packet(timestamp=100.0 + i, tcp_flags="A") for i in range(1, 9)]
    packets.append(make_packet(timestamp=110.0, tcp_flags="S"))
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None


# ---------------------------------------------------------------------------
# TCP flag anomalies
# ---------------------------------------------------------------------------


def test_tcp_flag_anomaly_fires_on_syn_fin_without_ack() -> None:
    thresholds = default_thresholds(flag_anomaly_min_packets=2)
    rule = TcpFlagAnomalyRule(thresholds)

    packets = [
        make_packet(timestamp=100.0, tcp_flags="S"),
        make_packet(timestamp=100.1, tcp_flags="F"),
    ]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    trigger = rule.evaluate(features, flow)
    assert trigger is not None
    assert trigger.rule_name == "tcp_flag_anomaly"


def test_tcp_flag_anomaly_fires_on_null_xmas_pattern() -> None:
    thresholds = default_thresholds(flag_anomaly_min_packets=2)
    rule = TcpFlagAnomalyRule(thresholds)

    packets = [
        make_packet(timestamp=100.0, tcp_flags="FPU"),  # FIN+PSH+URG, no SYN/ACK ("Xmas" pattern)
        make_packet(timestamp=100.1, tcp_flags="FPU"),
    ]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    trigger = rule.evaluate(features, flow)
    assert trigger is not None


def test_tcp_flag_anomaly_does_not_fire_on_normal_handshake() -> None:
    thresholds = default_thresholds(flag_anomaly_min_packets=2)
    rule = TcpFlagAnomalyRule(thresholds)

    packets = [
        make_packet(timestamp=100.0, tcp_flags="S"),
        make_packet(timestamp=100.1, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, tcp_flags="SA"),
        make_packet(timestamp=100.2, tcp_flags="A"),
        make_packet(timestamp=100.3, tcp_flags="FA"),
    ]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None


def test_tcp_flag_anomaly_does_not_fire_on_non_tcp_flow() -> None:
    thresholds = default_thresholds(flag_anomaly_min_packets=1)
    rule = TcpFlagAnomalyRule(thresholds)

    packets = [make_packet(timestamp=100.0, protocol=TransportProtocol.UDP, dst_port=53, tcp_flags=None)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None


# ---------------------------------------------------------------------------
# Short/high-volume burst
# ---------------------------------------------------------------------------


def test_short_high_volume_rule_fires_on_burst() -> None:
    thresholds = default_thresholds(short_flow_max_duration_seconds=0.5, short_flow_min_packets=20)
    rule = ShortHighVolumeRule(thresholds)

    packets = [make_packet(timestamp=100.0 + i * 0.01) for i in range(25)]  # 25 packets in 0.24s
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    trigger = rule.evaluate(features, flow)
    assert trigger is not None
    assert trigger.rule_name == "short_high_volume_burst"


def test_short_high_volume_rule_does_not_fire_when_duration_too_long() -> None:
    thresholds = default_thresholds(short_flow_max_duration_seconds=0.1, short_flow_min_packets=5)
    rule = ShortHighVolumeRule(thresholds)

    packets = [make_packet(timestamp=100.0 + i) for i in range(10)]  # spread over 9 seconds
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None


def test_short_high_volume_rule_does_not_fire_when_packet_count_too_low() -> None:
    thresholds = default_thresholds(short_flow_max_duration_seconds=1.0, short_flow_min_packets=100)
    rule = ShortHighVolumeRule(thresholds)

    packets = [make_packet(timestamp=100.0 + i * 0.01) for i in range(5)]
    flow = build_flow(packets)
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None


# ---------------------------------------------------------------------------
# Suspicious destination port
# ---------------------------------------------------------------------------


def test_suspicious_port_rule_fires_on_listed_port() -> None:
    thresholds = default_thresholds(suspicious_destination_ports=(3389,))
    rule = SuspiciousPortRule(thresholds)

    flow = build_flow([make_packet(timestamp=100.0, dst_port=3389)])
    features = extractor.extract(flow).to_dict()

    trigger = rule.evaluate(features, flow)
    assert trigger is not None
    assert trigger.score == SuspiciousPortRule.base_score


def test_suspicious_port_rule_does_not_fire_on_unlisted_port() -> None:
    thresholds = default_thresholds(suspicious_destination_ports=(3389,))
    rule = SuspiciousPortRule(thresholds)

    flow = build_flow([make_packet(timestamp=100.0, dst_port=443)])
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None


def test_suspicious_port_rule_does_not_fire_on_icmp_with_no_port() -> None:
    thresholds = default_thresholds(suspicious_destination_ports=(3389,))
    rule = SuspiciousPortRule(thresholds)

    flow = build_flow([make_packet(timestamp=100.0, protocol=TransportProtocol.ICMP, src_port=None, dst_port=None)])
    features = extractor.extract(flow).to_dict()

    assert rule.evaluate(features, flow) is None  # dst_port sentinel is -1.0, not in the suspicious set


# ---------------------------------------------------------------------------
# RuleDetector aggregation: multiple rules, risk score, severity
# ---------------------------------------------------------------------------


def test_multiple_rules_triggered_are_all_represented() -> None:
    thresholds = default_thresholds(
        high_packet_rate_pps=10.0,
        syn_heavy_ratio_threshold=0.5,
        syn_heavy_min_packets=5,
        suspicious_destination_ports=(4444,),
    )
    detector = RuleDetector(thresholds=thresholds, risk_scoring=RiskScoringConfig())

    # High packet rate (many packets, short duration) + SYN-heavy + suspicious port, all at once
    packets = [make_packet(timestamp=100.0 + i * 0.01, dst_port=4444, tcp_flags="S") for i in range(10)]
    flow = build_flow(packets)
    vector = extractor.extract(flow)

    result = detector.detect(vector, flow)

    assert result.is_suspicious is True
    assert "high_packet_rate" in result.rule_names
    assert "syn_heavy" in result.rule_names
    assert "suspicious_destination_port" in result.rule_names
    assert len(result.triggered_rules) >= 3
    # Each trigger has its own non-empty reason string -- not a shared/generic one.
    assert len(set(result.reasons)) == len(result.reasons)
    for reason in result.reasons:
        assert isinstance(reason, str) and len(reason) > 0


def test_risk_score_is_sum_of_triggered_rule_scores_capped_at_100() -> None:
    thresholds = default_thresholds(high_packet_rate_pps=1.0, high_byte_rate_bps=1.0)
    detector = RuleDetector(thresholds=thresholds, risk_scoring=RiskScoringConfig())

    packets = [make_packet(timestamp=100.0 + i * 0.01, length=1000) for i in range(10)]
    flow = build_flow(packets)
    vector = extractor.extract(flow)

    result = detector.detect(vector, flow)

    expected_uncapped = sum(t.score for t in result.triggered_rules)
    assert result.risk_score == min(100, expected_uncapped)
    assert result.risk_score <= 100


def test_severity_buckets_match_risk_scoring_config() -> None:
    risk_scoring = RiskScoringConfig(low_max=29, medium_max=59, high_max=79)
    detector = RuleDetector(thresholds=default_thresholds(), risk_scoring=risk_scoring, rules=())

    # No rules registered -> risk_score always 0 -> Severity.NONE, regardless of flow.
    flow = build_flow([make_packet(timestamp=100.0)])
    vector = extractor.extract(flow)
    result = detector.detect(vector, flow)
    assert result.risk_score == 0
    assert result.severity == Severity.NONE


def test_severity_classification_boundaries_directly() -> None:
    risk_scoring = RiskScoringConfig(low_max=29, medium_max=59, high_max=79)
    detector = RuleDetector(thresholds=default_thresholds(), risk_scoring=risk_scoring, rules=())

    assert detector._classify_severity(0) == Severity.NONE
    assert detector._classify_severity(1) == Severity.LOW
    assert detector._classify_severity(29) == Severity.LOW
    assert detector._classify_severity(30) == Severity.MEDIUM
    assert detector._classify_severity(59) == Severity.MEDIUM
    assert detector._classify_severity(60) == Severity.HIGH
    assert detector._classify_severity(79) == Severity.HIGH
    assert detector._classify_severity(80) == Severity.CRITICAL
    assert detector._classify_severity(100) == Severity.CRITICAL


# ---------------------------------------------------------------------------
# Zero/single-packet flows through the full rule set
# ---------------------------------------------------------------------------


def test_single_packet_flow_triggers_no_rules_with_default_thresholds() -> None:
    detector = RuleDetector(thresholds=default_thresholds(), risk_scoring=RiskScoringConfig())

    flow = build_flow([make_packet(timestamp=100.0, tcp_flags="S")])
    vector = extractor.extract(flow)
    result = detector.detect(vector, flow)

    # A lone SYN is common (an ordinary connection attempt) and must not,
    # by itself, be flagged by any default-threshold rule.
    assert result.is_suspicious is False
    assert result.risk_score == 0


def test_rule_detector_handles_icmp_flow_without_error() -> None:
    detector = RuleDetector(thresholds=default_thresholds(), risk_scoring=RiskScoringConfig())

    flow = build_flow([make_packet(timestamp=100.0, protocol=TransportProtocol.ICMP, src_port=None, dst_port=None, tcp_flags=None)])
    vector = extractor.extract(flow)
    result = detector.detect(vector, flow)  # must not raise

    assert result.risk_score >= 0