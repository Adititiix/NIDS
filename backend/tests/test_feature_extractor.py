"""
Unit tests for Phase 4 feature extraction.

These tests build FinalizedFlow objects the same way Phase 3 does (via
Flow.start()/add_packet()/finalize()) with synthetic PacketMetadata, then
verify FeatureExtractor produces a correct, deterministic, fixed-length
vector. No live traffic, no Scapy sniffing required.
"""

from __future__ import annotations

import math

import pytest

from app.capture.models import PacketMetadata, TransportProtocol
from app.features.feature_extractor import FeatureExtractionError, FeatureExtractor
from app.features.models import FeatureVector
from app.features.schema import FEATURE_COUNT, FEATURE_NAMES, FEATURE_SCHEMA, FEATURE_SCHEMA_VERSION
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
    """Build a FinalizedFlow from a sequence of packets, exactly as FlowAggregator would."""
    flow: Flow | None = None
    for pkt in packets:
        if flow is None:
            flow = Flow.start(pkt)
        else:
            flow.add_packet(pkt)
    assert flow is not None
    return flow.finalize()


def idx(name: str) -> int:
    """Look up a feature's index by name, so tests read by name, not magic numbers."""
    return FEATURE_NAMES.index(name)


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


# ---------------------------------------------------------------------------
# Schema self-consistency
# ---------------------------------------------------------------------------


def test_schema_names_are_unique_and_contiguous() -> None:
    assert len(FEATURE_NAMES) == FEATURE_COUNT
    assert len(set(FEATURE_NAMES)) == FEATURE_COUNT
    assert [spec.index for spec in FEATURE_SCHEMA] == list(range(FEATURE_COUNT))


# ---------------------------------------------------------------------------
# Vector shape / determinism
# ---------------------------------------------------------------------------


def test_feature_vector_has_exact_expected_length(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0)])
    vector = extractor.extract(flow)
    assert len(vector.values) == FEATURE_COUNT
    assert len(vector.to_list()) == FEATURE_COUNT
    assert len(vector.to_dict()) == FEATURE_COUNT


def test_feature_vector_ordering_matches_schema(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0, dst_port=8080)])
    vector = extractor.extract(flow)
    as_dict = vector.to_dict()
    assert list(as_dict.keys()) == list(FEATURE_NAMES)
    assert as_dict["dst_port"] == 8080.0


def test_extraction_is_deterministic(extractor: FeatureExtractor) -> None:
    flow = build_flow(
        [
            make_packet(timestamp=100.0, length=60, tcp_flags="S"),
            make_packet(timestamp=100.5, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=1400, tcp_flags="SA"),
        ]
    )
    v1 = extractor.extract(flow)
    v2 = extractor.extract(flow)
    assert v1.values == v2.values
    assert v1.to_json() == v2.to_json()


def test_feature_vector_carries_schema_version(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0)])
    vector = extractor.extract(flow)
    assert vector.schema_version == FEATURE_SCHEMA_VERSION


def test_feature_vector_rejects_wrong_length_construction() -> None:
    with pytest.raises(ValueError):
        FeatureVector(values=(1.0, 2.0), schema_version=FEATURE_SCHEMA_VERSION, flow_key_repr="x")


# ---------------------------------------------------------------------------
# Single-packet flow edge cases
# ---------------------------------------------------------------------------


def test_single_packet_flow_has_zero_duration_and_zero_rates(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0, length=60)])
    vector = extractor.extract(flow)
    d = vector.to_dict()

    assert d["flow_duration_seconds"] == 0.0
    assert d["total_packets"] == 1.0
    assert d["total_forward_packets"] == 1.0
    assert d["total_backward_packets"] == 0.0
    # Rates must be 0.0, never NaN/Inf, when duration is 0.
    assert d["total_packets_per_second"] == 0.0
    assert d["forward_packets_per_second"] == 0.0
    assert d["backward_packets_per_second"] == 0.0
    assert d["total_bytes_per_second"] == 0.0


def test_single_packet_flow_length_stats_equal_the_one_packet(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0, length=77)])
    d = extractor.extract(flow).to_dict()

    assert d["packet_length_mean"] == 77.0
    assert d["packet_length_min"] == 77.0
    assert d["packet_length_max"] == 77.0
    assert d["packet_length_std"] == 0.0  # no variance observable from 1 sample


def test_single_packet_flow_iat_stats_are_zero_not_nan(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0)])
    d = extractor.extract(flow).to_dict()

    for name in ("flow_iat_mean", "flow_iat_std", "flow_iat_min", "flow_iat_max", "forward_iat_mean", "forward_iat_std"):
        assert d[name] == 0.0
        assert not math.isnan(d[name])
        assert not math.isinf(d[name])


def test_single_packet_flow_has_zero_backward_stats(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0, length=90)])
    d = extractor.extract(flow).to_dict()

    assert d["backward_packet_length_mean"] == 0.0
    assert d["backward_packet_length_min"] == 0.0
    assert d["backward_packet_length_max"] == 0.0
    assert d["backward_iat_mean"] == 0.0
    assert d["backward_iat_std"] == 0.0


# ---------------------------------------------------------------------------
# Multi-packet, forward/backward, byte and rate calculations
# ---------------------------------------------------------------------------


def test_multi_packet_bidirectional_flow_feature_values(extractor: FeatureExtractor) -> None:
    packets = [
        make_packet(timestamp=100.0, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443, length=60),
        make_packet(timestamp=100.5, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=1400),
        make_packet(timestamp=101.0, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443, length=60),
        make_packet(timestamp=102.0, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=1400),
    ]
    flow = build_flow(packets)
    d = extractor.extract(flow).to_dict()

    assert d["total_packets"] == 4.0
    assert d["total_bytes"] == 2920.0
    assert d["total_forward_packets"] == 2.0
    assert d["total_backward_packets"] == 2.0
    assert d["total_forward_bytes"] == 120.0
    assert d["total_backward_bytes"] == 2800.0
    assert d["flow_duration_seconds"] == pytest.approx(2.0)

    # Rates: count / duration
    assert d["total_packets_per_second"] == pytest.approx(4.0 / 2.0)
    assert d["forward_packets_per_second"] == pytest.approx(2.0 / 2.0)
    assert d["backward_packets_per_second"] == pytest.approx(2.0 / 2.0)
    assert d["total_bytes_per_second"] == pytest.approx(2920.0 / 2.0)

    # Forward IAT: gaps between forward packets at t=100.0 and t=101.0 -> 1.0s
    assert d["forward_iat_mean"] == pytest.approx(1.0)
    # Backward IAT: gaps between backward packets at t=100.5 and t=102.0 -> 1.5s
    assert d["backward_iat_mean"] == pytest.approx(1.5)


def test_packet_length_mean_matches_bytes_over_packets(extractor: FeatureExtractor) -> None:
    flow = build_flow(
        [
            make_packet(timestamp=100.0, length=50),
            make_packet(timestamp=100.1, length=150),
        ]
    )
    d = extractor.extract(flow).to_dict()
    assert d["packet_length_mean"] == pytest.approx(d["total_bytes"] / d["total_packets"])
    assert d["packet_length_min"] == 50.0
    assert d["packet_length_max"] == 150.0
    assert d["packet_length_std"] == pytest.approx(50.0)  # population std of [50, 150]


# ---------------------------------------------------------------------------
# Protocol one-hot encoding
# ---------------------------------------------------------------------------


def test_tcp_protocol_one_hot(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0, protocol=TransportProtocol.TCP)])
    d = extractor.extract(flow).to_dict()
    assert (d["protocol_is_tcp"], d["protocol_is_udp"], d["protocol_is_icmp"]) == (1.0, 0.0, 0.0)


def test_udp_protocol_one_hot(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0, protocol=TransportProtocol.UDP, dst_port=53)])
    d = extractor.extract(flow).to_dict()
    assert (d["protocol_is_tcp"], d["protocol_is_udp"], d["protocol_is_icmp"]) == (0.0, 1.0, 0.0)


def test_icmp_protocol_one_hot_and_no_ports(extractor: FeatureExtractor) -> None:
    flow = build_flow(
        [make_packet(timestamp=100.0, protocol=TransportProtocol.ICMP, src_port=None, dst_port=None)]
    )
    d = extractor.extract(flow).to_dict()
    assert (d["protocol_is_tcp"], d["protocol_is_udp"], d["protocol_is_icmp"]) == (0.0, 0.0, 1.0)
    assert d["dst_port"] == -1.0  # sentinel for "no port"


# ---------------------------------------------------------------------------
# TCP flag counting
# ---------------------------------------------------------------------------


def test_tcp_flag_counts_across_a_handshake(extractor: FeatureExtractor) -> None:
    packets = [
        make_packet(timestamp=100.0, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443, tcp_flags="S"),
        make_packet(timestamp=100.1, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, tcp_flags="SA"),
        make_packet(timestamp=100.2, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443, tcp_flags="A"),
        make_packet(timestamp=100.3, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443, tcp_flags="FA"),
    ]
    flow = build_flow(packets)
    d = extractor.extract(flow).to_dict()

    assert d["syn_count"] == 2.0  # "S" and "SA"
    assert d["ack_count"] == 3.0  # "SA", "A", "FA"
    assert d["fin_count"] == 1.0  # "FA"
    assert d["rst_count"] == 0.0
    assert d["psh_count"] == 0.0
    assert d["urg_count"] == 0.0


def test_udp_flow_has_zero_tcp_flag_counts(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0, protocol=TransportProtocol.UDP, dst_port=53, tcp_flags=None)])
    d = extractor.extract(flow).to_dict()
    for name in ("syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count"):
        assert d[name] == 0.0


# ---------------------------------------------------------------------------
# Zero-duration flows with many packets (all at the same timestamp)
# ---------------------------------------------------------------------------


def test_zero_duration_multi_packet_flow_has_zero_rate_not_infinite(extractor: FeatureExtractor) -> None:
    # Multiple packets, but all captured at the exact same timestamp
    # (plausible with coarse capture-time resolution) -- duration is 0
    # despite packet_count > 1, and rate must still be 0.0, not Inf.
    flow = build_flow(
        [
            make_packet(timestamp=100.0, length=60),
            make_packet(timestamp=100.0, length=60),
            make_packet(timestamp=100.0, length=60),
        ]
    )
    d = extractor.extract(flow).to_dict()
    assert d["flow_duration_seconds"] == 0.0
    assert d["total_packets_per_second"] == 0.0
    assert not math.isinf(d["total_packets_per_second"])
    assert not math.isnan(d["total_packets_per_second"])


# ---------------------------------------------------------------------------
# No NaN / infinite values anywhere, across a variety of flow shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "packets",
    [
        [make_packet(timestamp=100.0)],
        [make_packet(timestamp=100.0, protocol=TransportProtocol.ICMP, src_port=None, dst_port=None)],
        [make_packet(timestamp=100.0, protocol=TransportProtocol.UDP, dst_port=53)],
        [
            make_packet(timestamp=100.0, length=40, tcp_flags="S"),
            make_packet(timestamp=100.0, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=40, tcp_flags="SA"),
            make_packet(timestamp=105.3, length=1500, tcp_flags="A"),
        ],
    ],
)
def test_no_nan_or_infinite_values_in_any_feature(extractor: FeatureExtractor, packets: list[PacketMetadata]) -> None:
    flow = build_flow(packets)
    vector = extractor.extract(flow)
    for name, value in vector.to_dict().items():
        assert not math.isnan(value), f"{name} was NaN"
        assert not math.isinf(value), f"{name} was infinite"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_extract_rejects_flow_with_inconsistent_packet_counts(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0)])
    # Tamper with an otherwise-valid finalized flow to simulate a corrupted/malformed input.
    broken = FinalizedFlow(
        key=flow.key,
        origin_src_ip=flow.origin_src_ip,
        origin_src_port=flow.origin_src_port,
        origin_dst_ip=flow.origin_dst_ip,
        origin_dst_port=flow.origin_dst_port,
        first_seen=flow.first_seen,
        last_seen=flow.last_seen,
        packet_count=5,  # inconsistent with forward+backward below
        byte_count=flow.byte_count,
        forward_packet_count=1,
        backward_packet_count=1,
        forward_byte_count=flow.forward_byte_count,
        backward_byte_count=flow.backward_byte_count,
    )
    with pytest.raises(FeatureExtractionError):
        extractor.extract(broken)


def test_extract_rejects_flow_with_zero_packets(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0)])
    broken = FinalizedFlow(
        key=flow.key,
        origin_src_ip=flow.origin_src_ip,
        origin_src_port=flow.origin_src_port,
        origin_dst_ip=flow.origin_dst_ip,
        origin_dst_port=flow.origin_dst_port,
        first_seen=flow.first_seen,
        last_seen=flow.last_seen,
        packet_count=0,
        byte_count=0,
        forward_packet_count=0,
        backward_packet_count=0,
        forward_byte_count=0,
        backward_byte_count=0,
    )
    with pytest.raises(FeatureExtractionError):
        extractor.extract(broken)


def test_extract_rejects_flow_with_last_seen_before_first_seen(extractor: FeatureExtractor) -> None:
    flow = build_flow([make_packet(timestamp=100.0)])
    broken = FinalizedFlow(
        key=flow.key,
        origin_src_ip=flow.origin_src_ip,
        origin_src_port=flow.origin_src_port,
        origin_dst_ip=flow.origin_dst_ip,
        origin_dst_port=flow.origin_dst_port,
        first_seen=100.0,
        last_seen=99.0,
        packet_count=1,
        byte_count=60,
        forward_packet_count=1,
        backward_packet_count=0,
        forward_byte_count=60,
        backward_byte_count=0,
    )
    with pytest.raises(FeatureExtractionError):
        extractor.extract(broken)
