"""
Unit tests for the Phase 2 packet capture engine.

Design constraint (per project spec): these tests must NOT require live
traffic or elevated (CAP_NET_RAW/root) permissions, so they can run in any
CI environment. We achieve that by:

- Monkeypatching interface discovery (get_if_list / conf.iface) instead of
  relying on the real host's interfaces.
- Building synthetic in-memory Scapy packets (IP()/TCP()/UDP()/ICMP()) for
  metadata-extraction tests -- this requires no socket access at all.
- Monkeypatching `sniff` itself for the start/stop lifecycle test, so no
  actual packet capture occurs.
"""

from __future__ import annotations

import queue
import time

import pytest
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.l2 import ARP, Ether

from app.capture import packet_capture as pc
from app.capture.models import CaptureStats, TransportProtocol

# ---------------------------------------------------------------------------
# Interface resolution
# ---------------------------------------------------------------------------


def test_resolve_interface_uses_valid_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "list_available_interfaces", lambda: ["eth0", "lo"])
    assert pc.resolve_interface("eth0") == "eth0"


def test_resolve_interface_falls_back_when_override_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "list_available_interfaces", lambda: ["eth0", "lo"])
    monkeypatch.setattr(pc, "detect_default_interface", lambda: "eth0")
    # Requesting a nonexistent interface should NOT raise -- it should fall
    # back to auto-detection rather than hard-failing on a stale config value.
    assert pc.resolve_interface("does-not-exist-99") == "eth0"


def test_resolve_interface_raises_when_nothing_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "list_available_interfaces", lambda: [])

    def _raise() -> str:
        raise pc.InterfaceNotFoundError("no interfaces")

    monkeypatch.setattr(pc, "detect_default_interface", _raise)
    with pytest.raises(pc.InterfaceNotFoundError):
        pc.resolve_interface("eth0")


def test_detect_default_interface_uses_conf_iface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConf:
        iface = "fake-default-iface"

    monkeypatch.setattr(pc, "conf", FakeConf)

    assert pc.detect_default_interface() == "fake-default-iface"


def test_detect_default_interface_falls_back_to_working_ifaces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc.conf, "iface", None, raising=False)

    class _FakeIface:
        name = "lo"

    monkeypatch.setattr(pc, "get_working_ifaces", lambda: [_FakeIface()])
    assert pc.detect_default_interface() == "lo"


# ---------------------------------------------------------------------------
# Packet metadata extraction (synthetic packets, no live traffic needed)
# ---------------------------------------------------------------------------


def test_extract_metadata_from_tcp_syn_packet() -> None:
    pkt = IP(src="10.0.2.15", dst="10.0.2.1") / TCP(sport=51234, dport=443, flags="S")
    metadata = pc.extract_packet_metadata(pkt)

    assert metadata is not None
    assert metadata.src_ip == "10.0.2.15"
    assert metadata.dst_ip == "10.0.2.1"
    assert metadata.src_port == 51234
    assert metadata.dst_port == 443
    assert metadata.protocol == TransportProtocol.TCP
    assert metadata.tcp_flags == "S"
    assert metadata.length > 0


def test_extract_metadata_from_udp_packet() -> None:
    pkt = IP(src="10.0.2.15", dst="8.8.8.8") / UDP(sport=53421, dport=53)
    metadata = pc.extract_packet_metadata(pkt)

    assert metadata is not None
    assert metadata.protocol == TransportProtocol.UDP
    assert metadata.src_port == 53421
    assert metadata.dst_port == 53
    assert metadata.tcp_flags is None


def test_extract_metadata_from_icmp_packet_has_no_ports() -> None:
    pkt = IP(src="10.0.2.15", dst="10.0.2.1") / ICMP()
    metadata = pc.extract_packet_metadata(pkt)

    assert metadata is not None
    assert metadata.protocol == TransportProtocol.ICMP
    assert metadata.src_port is None
    assert metadata.dst_port is None


def test_extract_metadata_returns_none_for_non_ip_packet() -> None:
    pkt = Ether() / ARP()
    assert pc.extract_packet_metadata(pkt) is None


# ---------------------------------------------------------------------------
# CaptureStats
# ---------------------------------------------------------------------------


def test_capture_stats_drop_rate_and_pps() -> None:
    stats = CaptureStats(
        packets_observed=100,
        packets_processed=90,
        packets_dropped=10,
        start_time=1000.0,
        last_packet_time=1010.0,
    )
    assert stats.packet_drop_rate_pct == pytest.approx(10.0)
    assert stats.elapsed_seconds == pytest.approx(10.0)
    assert stats.packets_per_second == pytest.approx(9.0)


def test_capture_stats_handles_zero_observed_without_division_error() -> None:
    stats = CaptureStats()
    assert stats.packet_drop_rate_pct == 0.0
    assert stats.packets_per_second == 0.0


# ---------------------------------------------------------------------------
# PacketCapture: queue behavior and drop counting (direct callback invocation,
# no sniff thread involved)
# ---------------------------------------------------------------------------


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> pc.PacketCapture:
    monkeypatch.setattr(pc, "list_available_interfaces", lambda: ["eth0"])
    return pc.PacketCapture(interface="eth0", queue_max_size=2)


def test_handle_packet_enqueues_valid_ip_packets(capture: pc.PacketCapture) -> None:
    pkt = IP(src="10.0.2.15", dst="10.0.2.1") / TCP(sport=1234, dport=80, flags="S")
    capture._handle_packet(pkt)

    assert capture.stats.packets_observed == 1
    assert capture.stats.packets_processed == 1
    assert capture.stats.packets_dropped == 0
    assert capture.queue.qsize() == 1


def test_handle_packet_counts_non_ip_as_observed_but_not_processed(capture: pc.PacketCapture) -> None:
    capture._handle_packet(Ether() / ARP())

    assert capture.stats.packets_observed == 1
    assert capture.stats.packets_processed == 0
    assert capture.queue.qsize() == 0


def test_handle_packet_drops_when_queue_full(capture: pc.PacketCapture) -> None:
    # queue_max_size=2 from the fixture
    for _ in range(4):
        pkt = IP(src="10.0.2.15", dst="10.0.2.1") / TCP(sport=1234, dport=80, flags="S")
        capture._handle_packet(pkt)

    assert capture.stats.packets_observed == 4
    assert capture.stats.packets_processed == 2
    assert capture.stats.packets_dropped == 2
    assert capture.queue.qsize() == 2


# ---------------------------------------------------------------------------
# Start/stop lifecycle (sniff() itself is mocked -- no real capture occurs)
# ---------------------------------------------------------------------------


def test_start_stop_lifecycle_with_mocked_sniff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "list_available_interfaces", lambda: ["eth0"])

    call_count = {"n": 0}

    def fake_sniff(*, iface: str, prn, store: bool, promisc: bool, timeout: float) -> None:  # noqa: ANN001
        call_count["n"] += 1
        synthetic_packet = IP(src="10.0.2.15", dst="10.0.2.1") / TCP(sport=1111, dport=22, flags="S")
        prn(synthetic_packet)
        time.sleep(0.01)  # simulate the timeout slice without actually waiting 1s per iteration

    monkeypatch.setattr(pc, "sniff", fake_sniff)

    capture = pc.PacketCapture(interface="eth0")
    assert not capture.is_running

    capture.start()
    try:
        assert capture.is_running
        # Give the background thread a few iterations to run.
        deadline = time.time() + 2.0
        while capture.stats.packets_processed == 0 and time.time() < deadline:
            time.sleep(0.02)

        assert capture.stats.packets_observed > 0
        assert capture.stats.packets_processed > 0
        assert call_count["n"] > 0
    finally:
        capture.stop(timeout=2.0)

    assert not capture.is_running


def test_start_is_idempotent_when_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "list_available_interfaces", lambda: ["eth0"])

    def fake_sniff(*, iface: str, prn, store: bool, promisc: bool, timeout: float) -> None:  # noqa: ANN001
        time.sleep(0.01)

    monkeypatch.setattr(pc, "sniff", fake_sniff)

    capture = pc.PacketCapture(interface="eth0")
    capture.start()
    first_thread = capture._thread
    capture.start()  # should be a no-op, not spawn a second thread
    try:
        assert capture._thread is first_thread
    finally:
        capture.stop(timeout=2.0)


def test_stats_snapshot_is_serializable_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "list_available_interfaces", lambda: ["eth0"])
    capture = pc.PacketCapture(interface="eth0")
    snapshot = capture.stats_snapshot()

    expected_keys = {
        "interface",
        "is_running",
        "packets_observed",
        "packets_processed",
        "packets_dropped",
        "packet_drop_rate_pct",
        "packets_per_second",
        "queue_size",
        "queue_max_size",
    }
    assert expected_keys.issubset(snapshot.keys())
    assert snapshot["interface"] == "eth0"
    assert snapshot["is_running"] is False
