"""
Data models for flow aggregation.

A "flow" here is the standard 5-tuple grouping (src IP, dst IP, src port,
dst port, protocol), bidirectional: packets traveling in either direction
between the same two endpoints belong to the same flow. Direction relative
to the *first* packet that created the flow ("forward") vs. its reverse
("backward") is tracked separately, since forward/backward packet and byte
counts are needed feature-engineering inputs in Phase 4.

This module does not know about Scapy, sockets, or capture at all -- it
only depends on app.capture.models.PacketMetadata, per the Phase 3
requirement that flow aggregation not touch raw packets directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.capture.models import PacketMetadata, TransportProtocol
from app.common.stats import RunningStats


def _endpoint_sort_key(ip: str, port: int | None) -> tuple[str, int]:
    """
    Sort key for canonicalizing endpoint order. Only used for comparison --
    the actual (ip, port) tuple stored in the key preserves a real `None`
    for portless protocols (e.g. ICMP).
    """
    return (ip, port if port is not None else -1)


@dataclass(frozen=True, slots=True)
class FlowKey:
    """
    Canonical, direction-independent identifier for a flow.

    Endpoints are stored in a deterministic (sorted) order so that a packet
    from A->B and its reply from B->A both hash to the same key. This is
    what makes the flow table bidirectional: `FlowAggregator` doesn't need
    to try both orderings when looking up a flow for an incoming packet.
    """

    endpoint1_ip: str
    endpoint1_port: int | None
    endpoint2_ip: str
    endpoint2_port: int | None
    protocol: TransportProtocol

    @classmethod
    def from_packet(cls, pkt: PacketMetadata) -> FlowKey:
        a = (pkt.src_ip, pkt.src_port)
        b = (pkt.dst_ip, pkt.dst_port)
        if _endpoint_sort_key(*a) <= _endpoint_sort_key(*b):
            endpoint1, endpoint2 = a, b
        else:
            endpoint1, endpoint2 = b, a
        return cls(
            endpoint1_ip=endpoint1[0],
            endpoint1_port=endpoint1[1],
            endpoint2_ip=endpoint2[0],
            endpoint2_port=endpoint2[1],
            protocol=pkt.protocol,
        )


@dataclass(frozen=True, slots=True)
class FinalizedFlow:
    """
    Immutable snapshot of a completed (expired) flow, handed off to Phase 4
    feature extraction. Frozen so a downstream consumer can't accidentally
    mutate a flow that the aggregator considers closed.
    """

    key: FlowKey
    origin_src_ip: str
    origin_src_port: int | None
    origin_dst_ip: str
    origin_dst_port: int | None
    first_seen: float
    last_seen: float
    packet_count: int
    byte_count: int
    forward_packet_count: int
    backward_packet_count: int
    forward_byte_count: int
    backward_byte_count: int

    # --- Added for Phase 4 feature extraction. All are additive: nothing
    # above this line changed meaning or position. See docs/methodology.md
    # for why the aggregate counters above aren't sufficient on their own
    # (e.g. packet_length_mean IS derivable from byte_count/packet_count,
    # but std/min/max are not, and IAT/flag data didn't exist here at all).

    # Packet-length distribution, in bytes. "packet_length_*" is computed
    # over ALL packets in the flow; "forward_packet_length_*" and
    # "backward_packet_length_*" are the same distribution split by
    # direction, matching CICFlowMeter's convention of providing both.
    packet_length_mean: float = 0.0
    packet_length_std: float = 0.0
    packet_length_min: float = 0.0
    packet_length_max: float = 0.0
    forward_packet_length_mean: float = 0.0
    forward_packet_length_std: float = 0.0
    forward_packet_length_min: float = 0.0
    forward_packet_length_max: float = 0.0
    backward_packet_length_mean: float = 0.0
    backward_packet_length_std: float = 0.0
    backward_packet_length_min: float = 0.0
    backward_packet_length_max: float = 0.0

    # Inter-arrival time (seconds) between consecutive packets. "flow_iat_*"
    # considers every packet regardless of direction; "forward_iat_*" and
    # "backward_iat_*" only consider gaps between consecutive packets in
    # that same direction.
    flow_iat_mean: float = 0.0
    flow_iat_std: float = 0.0
    flow_iat_min: float = 0.0
    flow_iat_max: float = 0.0
    forward_iat_mean: float = 0.0
    forward_iat_std: float = 0.0
    backward_iat_mean: float = 0.0
    backward_iat_std: float = 0.0

    # TCP flag counts across the whole flow. Always 0 for non-TCP flows.
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    urg_count: int = 0

    @property
    def duration_seconds(self) -> float:
        return max(self.last_seen - self.first_seen, 0.0)


@dataclass(slots=True)
class Flow:
    """
    A mutable, in-progress flow being actively updated as packets arrive.

    "Forward" is defined relative to the packet that created the flow
    (origin_src_ip/port -> origin_dst_ip/port). A subsequent packet is
    forward if it matches that exact direction, and backward if it matches
    the reverse -- both map to the same FlowKey, so this is the only place
    direction is actually decided.
    """

    key: FlowKey
    origin_src_ip: str
    origin_src_port: int | None
    origin_dst_ip: str
    origin_dst_port: int | None
    first_seen: float
    last_seen: float
    packet_count: int = 0
    byte_count: int = 0
    forward_packet_count: int = 0
    backward_packet_count: int = 0
    forward_byte_count: int = 0
    backward_byte_count: int = 0

    # --- Added for Phase 4. Streaming accumulators only (see RunningStats
    # docstring for why raw samples aren't retained) plus small bookkeeping
    # fields to compute inter-arrival times without storing timestamps.
    length_stats: RunningStats = field(default_factory=RunningStats)
    forward_length_stats: RunningStats = field(default_factory=RunningStats)
    backward_length_stats: RunningStats = field(default_factory=RunningStats)
    flow_iat_stats: RunningStats = field(default_factory=RunningStats)
    forward_iat_stats: RunningStats = field(default_factory=RunningStats)
    backward_iat_stats: RunningStats = field(default_factory=RunningStats)
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    urg_count: int = 0
    _prev_timestamp: float | None = field(default=None, repr=False)
    _prev_forward_timestamp: float | None = field(default=None, repr=False)
    _prev_backward_timestamp: float | None = field(default=None, repr=False)

    @classmethod
    def start(cls, pkt: PacketMetadata) -> Flow:
        """Create a new flow from the packet that first establishes it."""
        flow = cls(
            key=FlowKey.from_packet(pkt),
            origin_src_ip=pkt.src_ip,
            origin_src_port=pkt.src_port,
            origin_dst_ip=pkt.dst_ip,
            origin_dst_port=pkt.dst_port,
            first_seen=pkt.timestamp,
            last_seen=pkt.timestamp,
        )
        flow.add_packet(pkt)
        return flow

    def is_forward(self, pkt: PacketMetadata) -> bool:
        """True if `pkt` travels in the same direction as the flow's originating packet."""
        return (
            pkt.src_ip == self.origin_src_ip
            and pkt.src_port == self.origin_src_port
            and pkt.dst_ip == self.origin_dst_ip
            and pkt.dst_port == self.origin_dst_port
        )

    def add_packet(self, pkt: PacketMetadata) -> None:
        """Fold one more packet into this flow's running counters."""
        self.packet_count += 1
        self.byte_count += pkt.length
        self.last_seen = max(self.last_seen, pkt.timestamp)

        self.length_stats.add(float(pkt.length))

        # Inter-arrival time, overall. Clamped to 0.0 rather than allowed to
        # go negative on an out-of-order arrival (see the Phase 3 note in
        # docs/methodology.md on last_seen using max() for the same reason) --
        # this is a documented limitation, not silently ignored.
        if self._prev_timestamp is not None:
            flow_iat = pkt.timestamp - self._prev_timestamp
            self.flow_iat_stats.add(max(flow_iat, 0.0))
        self._prev_timestamp = pkt.timestamp

        if self.is_forward(pkt):
            self.forward_packet_count += 1
            self.forward_byte_count += pkt.length
            self.forward_length_stats.add(float(pkt.length))
            if self._prev_forward_timestamp is not None:
                fwd_iat = pkt.timestamp - self._prev_forward_timestamp
                self.forward_iat_stats.add(max(fwd_iat, 0.0))
            self._prev_forward_timestamp = pkt.timestamp
        else:
            self.backward_packet_count += 1
            self.backward_byte_count += pkt.length
            self.backward_length_stats.add(float(pkt.length))
            if self._prev_backward_timestamp is not None:
                bwd_iat = pkt.timestamp - self._prev_backward_timestamp
                self.backward_iat_stats.add(max(bwd_iat, 0.0))
            self._prev_backward_timestamp = pkt.timestamp

        if pkt.protocol == TransportProtocol.TCP and pkt.tcp_flags:
            flags = pkt.tcp_flags
            if "S" in flags:
                self.syn_count += 1
            if "A" in flags:
                self.ack_count += 1
            if "F" in flags:
                self.fin_count += 1
            if "R" in flags:
                self.rst_count += 1
            if "P" in flags:
                self.psh_count += 1
            if "U" in flags:
                self.urg_count += 1

    def is_idle(self, now: float, timeout_seconds: float) -> bool:
        return (now - self.last_seen) >= timeout_seconds

    def finalize(self) -> FinalizedFlow:
        """Produce an immutable snapshot for handoff to Phase 4."""
        return FinalizedFlow(
            key=self.key,
            origin_src_ip=self.origin_src_ip,
            origin_src_port=self.origin_src_port,
            origin_dst_ip=self.origin_dst_ip,
            origin_dst_port=self.origin_dst_port,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            packet_count=self.packet_count,
            byte_count=self.byte_count,
            forward_packet_count=self.forward_packet_count,
            backward_packet_count=self.backward_packet_count,
            forward_byte_count=self.forward_byte_count,
            backward_byte_count=self.backward_byte_count,
            packet_length_mean=self.length_stats.mean,
            packet_length_std=self.length_stats.std,
            packet_length_min=self.length_stats.min,
            packet_length_max=self.length_stats.max,
            forward_packet_length_mean=self.forward_length_stats.mean,
            forward_packet_length_std=self.forward_length_stats.std,
            forward_packet_length_min=self.forward_length_stats.min,
            forward_packet_length_max=self.forward_length_stats.max,
            backward_packet_length_mean=self.backward_length_stats.mean,
            backward_packet_length_std=self.backward_length_stats.std,
            backward_packet_length_min=self.backward_length_stats.min,
            backward_packet_length_max=self.backward_length_stats.max,
            flow_iat_mean=self.flow_iat_stats.mean,
            flow_iat_std=self.flow_iat_stats.std,
            flow_iat_min=self.flow_iat_stats.min,
            flow_iat_max=self.flow_iat_stats.max,
            forward_iat_mean=self.forward_iat_stats.mean,
            forward_iat_std=self.forward_iat_stats.std,
            backward_iat_mean=self.backward_iat_stats.mean,
            backward_iat_std=self.backward_iat_stats.std,
            syn_count=self.syn_count,
            ack_count=self.ack_count,
            fin_count=self.fin_count,
            rst_count=self.rst_count,
            psh_count=self.psh_count,
            urg_count=self.urg_count,
        )


@dataclass
class FlowAggregatorStats:
    """Running counters for the flow aggregation layer."""

    total_flows_created: int = 0
    total_flows_expired: int = 0
    total_packets_ingested: int = 0
    queue_timeouts: int = field(default=0)  # number of times the consumer loop hit an empty-queue timeout
