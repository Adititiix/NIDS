"""
Unit tests for the Phase 3 flow aggregation engine.

All tests use synthetic PacketMetadata objects constructed directly --
no live packet capture, no Scapy, no PacketCapture instance required. This
matches Phase 2's testing pattern and the Phase 3 requirement that flow
aggregation be testable independent of real traffic.
"""

from __future__ import annotations

import queue
import time

import pytest

from app.capture.models import PacketMetadata, TransportProtocol
from app.flows.flow_aggregator import FlowAggregator
from app.flows.models import Flow, FlowKey


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
    """Build a synthetic PacketMetadata without needing Scapy or live capture."""
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


@pytest.fixture
def aggregator() -> FlowAggregator:
    # A real (but never-consumed-from-by-a-thread) queue is fine here --
    # these tests call ingest_packet()/expire_inactive_flows() directly.
    return FlowAggregator(source_queue=queue.Queue(), flow_timeout_seconds=30.0, flow_table_max_size=1000)


# ---------------------------------------------------------------------------
# FlowKey canonicalization
# ---------------------------------------------------------------------------


def test_flow_key_is_identical_for_both_directions() -> None:
    forward = make_packet(timestamp=1.0, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443)
    reverse = make_packet(timestamp=1.1, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234)

    assert FlowKey.from_packet(forward) == FlowKey.from_packet(reverse)


def test_flow_key_differs_for_different_ports() -> None:
    a = make_packet(timestamp=1.0, dst_port=443)
    b = make_packet(timestamp=1.0, dst_port=80)
    assert FlowKey.from_packet(a) != FlowKey.from_packet(b)


def test_flow_key_differs_for_different_protocol() -> None:
    a = make_packet(timestamp=1.0, protocol=TransportProtocol.TCP)
    b = make_packet(timestamp=1.0, protocol=TransportProtocol.UDP)
    assert FlowKey.from_packet(a) != FlowKey.from_packet(b)


# ---------------------------------------------------------------------------
# New flow creation
# ---------------------------------------------------------------------------


def test_ingest_first_packet_creates_new_flow(aggregator: FlowAggregator) -> None:
    pkt = make_packet(timestamp=100.0, length=60)
    flow = aggregator.ingest_packet(pkt)

    assert aggregator.active_flow_count == 1
    assert aggregator.stats.total_flows_created == 1
    assert flow.packet_count == 1
    assert flow.byte_count == 60
    assert flow.first_seen == 100.0
    assert flow.last_seen == 100.0


def test_ingest_does_not_create_duplicate_flow_for_same_5_tuple(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0))
    aggregator.ingest_packet(make_packet(timestamp=101.0))

    assert aggregator.active_flow_count == 1
    assert aggregator.stats.total_flows_created == 1


# ---------------------------------------------------------------------------
# Same-flow aggregation: packet count, byte count, first/last seen
# ---------------------------------------------------------------------------


def test_same_flow_packets_aggregate_counts(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0, length=60))
    aggregator.ingest_packet(make_packet(timestamp=101.0, length=40))
    aggregator.ingest_packet(make_packet(timestamp=102.5, length=80))

    key = FlowKey.from_packet(make_packet(timestamp=0))
    flow = aggregator._flows[key]  # internal access is fine for a white-box unit test

    assert flow.packet_count == 3
    assert flow.byte_count == 180
    assert flow.first_seen == 100.0
    assert flow.last_seen == 102.5


def test_last_seen_uses_max_timestamp_even_if_packets_arrive_out_of_order(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0))
    aggregator.ingest_packet(make_packet(timestamp=95.0))  # arrives "late" relative to timestamp

    key = FlowKey.from_packet(make_packet(timestamp=0))
    flow = aggregator._flows[key]
    assert flow.first_seen == 100.0  # first_seen is fixed at flow creation
    assert flow.last_seen == 100.0  # should not regress backward


# ---------------------------------------------------------------------------
# Forward vs backward direction
# ---------------------------------------------------------------------------


def test_forward_and_backward_packet_counts(aggregator: FlowAggregator) -> None:
    # Flow initiated 10.0.2.15:51234 -> 10.0.2.1:443 (forward direction)
    aggregator.ingest_packet(
        make_packet(timestamp=100.0, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443, length=60)
    )
    aggregator.ingest_packet(
        make_packet(timestamp=100.1, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443, length=60)
    )
    # Reply traffic: 10.0.2.1:443 -> 10.0.2.15:51234 (backward direction)
    aggregator.ingest_packet(
        make_packet(timestamp=100.2, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=1400)
    )

    key = FlowKey.from_packet(make_packet(timestamp=0, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443))
    flow = aggregator._flows[key]

    assert flow.packet_count == 3
    assert flow.forward_packet_count == 2
    assert flow.backward_packet_count == 1
    assert flow.forward_byte_count == 120
    assert flow.backward_byte_count == 1400


def test_reverse_direction_first_packet_still_creates_correct_forward_baseline(aggregator: FlowAggregator) -> None:
    # If the *first* packet we happen to observe is what would conventionally
    # be "the response" (e.g. we started capturing mid-flow), forward is
    # defined relative to whichever packet we saw first -- this is a known,
    # documented limitation for mid-flow capture starts.
    first_pkt = make_packet(timestamp=100.0, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234)
    second_pkt = make_packet(timestamp=100.1, src_ip="10.0.2.15", src_port=51234, dst_ip="10.0.2.1", dst_port=443)

    flow = aggregator.ingest_packet(first_pkt)
    aggregator.ingest_packet(second_pkt)

    assert flow.forward_packet_count == 1  # the first_pkt direction
    assert flow.backward_packet_count == 1  # second_pkt is the reverse of that


# ---------------------------------------------------------------------------
# Multiple simultaneous flows
# ---------------------------------------------------------------------------


def test_multiple_simultaneous_flows_are_tracked_independently(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0, dst_port=443))
    aggregator.ingest_packet(make_packet(timestamp=100.0, dst_port=80))
    aggregator.ingest_packet(make_packet(timestamp=100.0, dst_ip="8.8.8.8", dst_port=53, protocol=TransportProtocol.UDP))

    assert aggregator.active_flow_count == 3
    assert aggregator.stats.total_flows_created == 3


# ---------------------------------------------------------------------------
# Flow expiration (configurable timeout, deterministic via explicit `now`)
# ---------------------------------------------------------------------------


def test_flow_does_not_expire_before_timeout(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0))
    expired = aggregator.expire_inactive_flows(now=100.0 + aggregator.flow_timeout_seconds - 1)

    assert expired == []
    assert aggregator.active_flow_count == 1


def test_flow_expires_after_timeout(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0))
    expired = aggregator.expire_inactive_flows(now=100.0 + aggregator.flow_timeout_seconds)

    assert len(expired) == 1
    assert aggregator.active_flow_count == 0
    assert aggregator.stats.total_flows_expired == 1


def test_configurable_timeout_is_respected() -> None:
    short_timeout_aggregator = FlowAggregator(source_queue=queue.Queue(), flow_timeout_seconds=5.0)
    short_timeout_aggregator.ingest_packet(make_packet(timestamp=100.0))

    # Not yet expired at 4s idle
    assert short_timeout_aggregator.expire_inactive_flows(now=104.0) == []
    # Expired at exactly 5s idle
    assert len(short_timeout_aggregator.expire_inactive_flows(now=105.0)) == 1


def test_active_flow_is_not_expired_by_new_traffic_resetting_last_seen(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0))
    aggregator.ingest_packet(make_packet(timestamp=100.0 + aggregator.flow_timeout_seconds - 1))  # keeps it alive

    expired = aggregator.expire_inactive_flows(now=100.0 + aggregator.flow_timeout_seconds)
    assert expired == []
    assert aggregator.active_flow_count == 1


# ---------------------------------------------------------------------------
# Expired-flow retrieval / finalization
# ---------------------------------------------------------------------------


def test_expired_flows_are_retrievable_and_drained_exactly_once(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0, length=500))
    aggregator.expire_inactive_flows(now=100.0 + aggregator.flow_timeout_seconds)

    first_drain = aggregator.get_expired_flows()
    assert len(first_drain) == 1
    finalized = first_drain[0]
    assert finalized.byte_count == 500
    assert finalized.duration_seconds == 0.0

    second_drain = aggregator.get_expired_flows()
    assert second_drain == []  # already drained -- no duplicates


def test_get_expired_flows_respects_limit(aggregator: FlowAggregator) -> None:
    for i in range(5):
        aggregator.ingest_packet(make_packet(timestamp=100.0 + i, dst_port=1000 + i))
    aggregator.expire_inactive_flows(now=200.0 + aggregator.flow_timeout_seconds)

    first_batch = aggregator.get_expired_flows(limit=2)
    assert len(first_batch) == 2

    remaining = aggregator.get_expired_flows()
    assert len(remaining) == 3


def test_force_expire_all_finalizes_every_active_flow_regardless_of_idle_time(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0, dst_port=443))
    aggregator.ingest_packet(make_packet(timestamp=100.0, dst_port=80))

    finalized = aggregator.force_expire_all()

    assert len(finalized) == 2
    assert aggregator.active_flow_count == 0
    assert aggregator.stats.total_flows_expired == 2


# ---------------------------------------------------------------------------
# Flow-table statistics
# ---------------------------------------------------------------------------


def test_stats_snapshot_reports_active_created_and_expired_counts(aggregator: FlowAggregator) -> None:
    aggregator.ingest_packet(make_packet(timestamp=100.0, dst_port=443))
    aggregator.ingest_packet(make_packet(timestamp=100.0, dst_port=80))
    aggregator.expire_inactive_flows(now=100.0 + aggregator.flow_timeout_seconds)  # expires both

    snapshot = aggregator.stats_snapshot()
    assert snapshot["active_flow_count"] == 0
    assert snapshot["total_flows_created"] == 2
    assert snapshot["total_flows_expired"] == 2
    assert snapshot["total_packets_ingested"] == 2


def test_flow_table_max_size_evicts_oldest_flow_instead_of_growing_unboundedly() -> None:
    aggregator = FlowAggregator(source_queue=queue.Queue(), flow_timeout_seconds=9999.0, flow_table_max_size=2)

    aggregator.ingest_packet(make_packet(timestamp=100.0, dst_port=1))
    aggregator.ingest_packet(make_packet(timestamp=101.0, dst_port=2))
    # Table is now at max size (2). A third distinct flow must evict the oldest.
    aggregator.ingest_packet(make_packet(timestamp=102.0, dst_port=3))

    assert aggregator.active_flow_count == 2
    assert aggregator.stats.total_flows_created == 3
    assert aggregator.stats.total_flows_expired == 1  # the evicted one


# ---------------------------------------------------------------------------
# Empty queue behavior / background loop / graceful shutdown
# (these exercise the real threaded consumer loop, with a real queue.Queue)
# ---------------------------------------------------------------------------


def test_empty_queue_does_not_block_or_crash_the_consumer_loop() -> None:
    aggregator = FlowAggregator(
        source_queue=queue.Queue(),
        flow_timeout_seconds=30.0,
        queue_get_timeout=0.05,  # fast timeout so the test doesn't wait long
    )
    aggregator.start()
    try:
        time.sleep(0.3)  # several empty-queue timeout cycles with nothing to consume
        assert aggregator.is_running
        assert aggregator.stats.queue_timeouts > 0
        assert aggregator.active_flow_count == 0
    finally:
        aggregator.stop(timeout=2.0)


def test_background_loop_ingests_packets_pushed_onto_the_source_queue() -> None:
    source_queue: "queue.Queue" = queue.Queue()
    aggregator = FlowAggregator(source_queue=source_queue, flow_timeout_seconds=30.0, queue_get_timeout=0.05)

    aggregator.start()
    try:
        source_queue.put(make_packet(timestamp=time.time(), dst_port=443))
        source_queue.put(make_packet(timestamp=time.time(), dst_port=443))

        deadline = time.time() + 2.0
        while aggregator.stats.total_packets_ingested < 2 and time.time() < deadline:
            time.sleep(0.02)

        assert aggregator.stats.total_packets_ingested == 2
        assert aggregator.active_flow_count == 1
    finally:
        aggregator.stop(timeout=2.0)


def test_stop_is_graceful_and_idempotent_when_not_running() -> None:
    aggregator = FlowAggregator(source_queue=queue.Queue())
    assert aggregator.stop() == []  # no-op, must not raise
    assert not aggregator.is_running


def test_stop_finalizes_remaining_active_flows_by_default() -> None:
    source_queue: "queue.Queue" = queue.Queue()
    aggregator = FlowAggregator(source_queue=source_queue, flow_timeout_seconds=9999.0, queue_get_timeout=0.05)

    aggregator.start()
    try:
        source_queue.put(make_packet(timestamp=time.time(), dst_port=443))
        deadline = time.time() + 2.0
        while aggregator.active_flow_count == 0 and time.time() < deadline:
            time.sleep(0.02)
        assert aggregator.active_flow_count == 1
    finally:
        remaining = aggregator.stop(timeout=2.0)

    # The flow was nowhere near its 9999s timeout, but stop() should still
    # finalize it rather than silently dropping in-progress flow data.
    assert len(remaining) == 1
    assert aggregator.active_flow_count == 0


def test_start_is_idempotent_when_already_running() -> None:
    aggregator = FlowAggregator(source_queue=queue.Queue(), queue_get_timeout=0.05)
    aggregator.start()
    first_thread = aggregator._thread
    aggregator.start()  # should be a no-op, not spawn a second thread
    try:
        assert aggregator._thread is first_thread
    finally:
        aggregator.stop(timeout=2.0)
