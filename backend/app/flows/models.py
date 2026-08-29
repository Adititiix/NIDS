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

        if self.is_forward(pkt):
            self.forward_packet_count += 1
            self.forward_byte_count += pkt.length
        else:
            self.backward_packet_count += 1
            self.backward_byte_count += pkt.length

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
        )


@dataclass
class FlowAggregatorStats:
    """Running counters for the flow aggregation layer."""

    total_flows_created: int = 0
    total_flows_expired: int = 0
    total_packets_ingested: int = 0
    queue_timeouts: int = field(default=0)  # number of times the consumer loop hit an empty-queue timeout
