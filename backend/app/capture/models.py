"""
Data models for the packet capture layer.

These are intentionally minimal: PacketMetadata is the raw per-packet
material handed to the flow aggregator in Phase 3. It is NOT a feature
vector -- flow-level features (duration, IAT stats, flag counts, etc.) are
computed later from sequences of these records, per the feature
compatibility table in docs/methodology.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TransportProtocol(str, Enum):
    """Transport-layer protocol, as far as this project distinguishes them."""

    TCP = "TCP"
    UDP = "UDP"
    ICMP = "ICMP"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class PacketMetadata:
    """Minimal per-packet metadata extracted at capture time."""

    timestamp: float  # epoch seconds, from Scapy's packet.time (capture time, not processing time)
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    protocol: TransportProtocol
    length: int  # total packet length in bytes, as seen on the wire
    tcp_flags: str | None = None  # raw Scapy flag string (e.g. "S", "SA", "PA", "FA", "R"); parsed into counts in Phase 4


@dataclass
class CaptureStats:
    """
    Running counters for the capture engine.

    Exposed via PacketCapture.stats for the health endpoint (Phase 11) and
    the Phase 15 benchmark harness (packet drop rate, throughput).
    """

    packets_observed: int = 0
    packets_processed: int = 0
    packets_dropped: int = 0
    start_time: float | None = None
    last_packet_time: float | None = None

    @property
    def packet_drop_rate_pct(self) -> float:
        """
        Packet Drop Rate = dropped / observed * 100.

        Note: this only measures drops at OUR queue boundary (i.e. packets
        Scapy delivered to us but we couldn't enqueue fast enough). It does
        NOT capture NIC-ring-buffer or libpcap-level drops, which depend on
        the capture technology and OS -- see the limitation noted in
        docs/threat_model.md.
        """
        if self.packets_observed == 0:
            return 0.0
        return (self.packets_dropped / self.packets_observed) * 100

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.last_packet_time or self.start_time
        return max(end - self.start_time, 0.0)

    @property
    def packets_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.packets_processed / elapsed
