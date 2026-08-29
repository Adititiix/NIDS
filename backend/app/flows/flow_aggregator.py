"""
Flow aggregation engine.

Responsibilities (Phase 3 scope):
- Consume PacketMetadata objects (from PacketCapture.queue, or any
  queue.Queue[PacketMetadata] -- this module has no dependency on Scapy or
  raw packets at all, only on the Phase 2 PacketMetadata abstraction).
- Group packets into bidirectional 5-tuple flows in an in-memory table.
- Track first_seen/last_seen, packet/byte counts, and forward/backward
  counts per flow.
- Expire flows that have been idle longer than a configurable timeout, so
  the table can't grow unboundedly, and hand finalized flows off for
  Phase 4 feature extraction.
- Expose flow-table statistics (active count, total created, total expired).

Explicitly OUT of scope for Phase 3: feature extraction, detection, risk
scoring, and anything that writes to the database.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from app.capture.models import PacketMetadata
from app.config import settings
from app.flows.models import Flow, FlowAggregatorStats, FlowKey, FinalizedFlow

logger = logging.getLogger("nids.flows")

# How often (in seconds) the background consumer loop re-checks the flow
# table for idle flows to expire. Independent of flow_timeout_seconds itself
# -- this just bounds how "stale" an expired flow can be before it's
# actually moved out of the active table, without scanning the whole table
# on every single packet.
_DEFAULT_EXPIRY_CHECK_INTERVAL_SECONDS = 5.0

# How long a single queue.get() call blocks before giving the consumer loop
# a chance to check the stop event and run an expiry sweep, mirroring the
# pattern used in PacketCapture._run_sniff_loop.
_DEFAULT_QUEUE_GET_TIMEOUT_SECONDS = 1.0


class FlowAggregator:
    """
    Consumes PacketMetadata from a source queue and maintains a bidirectional
    flow table with inactivity-based expiration.

    Usage:
        aggregator = FlowAggregator(source_queue=capture.queue)
        aggregator.start()
        ...
        finalized = aggregator.get_expired_flows()   # feed to Phase 4
        ...
        aggregator.stop()

    Also fully usable without the background thread, for direct/synchronous
    testing:
        aggregator = FlowAggregator(source_queue=queue.Queue())
        aggregator.ingest_packet(some_packet_metadata)
        aggregator.expire_inactive_flows(now=...)
    """

    def __init__(
        self,
        source_queue: "queue.Queue[PacketMetadata]",
        flow_timeout_seconds: float | None = None,
        flow_table_max_size: int | None = None,
        queue_get_timeout: float = _DEFAULT_QUEUE_GET_TIMEOUT_SECONDS,
        expiry_check_interval_seconds: float = _DEFAULT_EXPIRY_CHECK_INTERVAL_SECONDS,
    ) -> None:
        self.source_queue = source_queue
        self.flow_timeout_seconds = flow_timeout_seconds or settings.flow_timeout_seconds
        self.flow_table_max_size = flow_table_max_size or settings.flow_table_max_size
        self.queue_get_timeout = queue_get_timeout
        self.expiry_check_interval_seconds = expiry_check_interval_seconds

        self._flows: dict[FlowKey, Flow] = {}
        self._expired_flows: list[FinalizedFlow] = []
        self._lock = threading.RLock()
        self.stats = FlowAggregatorStats()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._last_expiry_check = 0.0

    # ------------------------------------------------------------------
    # Core ingestion / expiration (thread-safe; usable directly or from
    # the background loop)
    # ------------------------------------------------------------------

    def ingest_packet(self, pkt: PacketMetadata) -> Flow:
        """
        Fold one packet into the flow table: creates a new flow if this is
        the first packet for its 5-tuple, otherwise updates the existing
        flow. Thread-safe.
        """
        key = FlowKey.from_packet(pkt)
        with self._lock:
            flow = self._flows.get(key)
            if flow is None:
                if len(self._flows) >= self.flow_table_max_size:
                    # Safety valve: rather than growing unboundedly, force-expire
                    # the single oldest (least-recently-active) flow to make room.
                    self._evict_oldest_flow_locked()
                flow = Flow.start(pkt)
                self._flows[key] = flow
                self.stats.total_flows_created += 1
            else:
                flow.add_packet(pkt)
            self.stats.total_packets_ingested += 1
            return flow

    def _evict_oldest_flow_locked(self) -> None:
        """Must be called with self._lock held. Finalizes the least-recently-active flow."""
        if not self._flows:
            return
        oldest_key = min(self._flows, key=lambda k: self._flows[k].last_seen)
        oldest_flow = self._flows.pop(oldest_key)
        logger.warning(
            "Flow table reached max size (%d) -- force-expiring oldest flow (last_seen=%.2f) to make room.",
            self.flow_table_max_size,
            oldest_flow.last_seen,
        )
        self._expired_flows.append(oldest_flow.finalize())
        self.stats.total_flows_expired += 1

    def expire_inactive_flows(self, now: float | None = None) -> list[FinalizedFlow]:
        """
        Scan the flow table and finalize any flow idle longer than
        `flow_timeout_seconds`. Returns the newly expired flows (they are
        also appended to the internal buffer retrievable via
        `get_expired_flows()`). Thread-safe.

        `now` can be supplied explicitly (tests do this) to make expiration
        deterministic without real sleeping; defaults to time.time().
        """
        current_time = now if now is not None else time.time()
        newly_expired: list[FinalizedFlow] = []
        with self._lock:
            idle_keys = [k for k, f in self._flows.items() if f.is_idle(current_time, self.flow_timeout_seconds)]
            for k in idle_keys:
                flow = self._flows.pop(k)
                finalized = flow.finalize()
                newly_expired.append(finalized)
            if newly_expired:
                self.stats.total_flows_expired += len(newly_expired)
                self._expired_flows.extend(newly_expired)
        return newly_expired

    def force_expire_all(self) -> list[FinalizedFlow]:
        """
        Finalize every currently active flow regardless of idle time. Used
        on shutdown so in-progress flow data isn't silently discarded.
        Thread-safe.
        """
        with self._lock:
            finalized = [flow.finalize() for flow in self._flows.values()]
            self._flows.clear()
            if finalized:
                self.stats.total_flows_expired += len(finalized)
                self._expired_flows.extend(finalized)
            return finalized

    def get_expired_flows(self, limit: int | None = None) -> list[FinalizedFlow]:
        """
        Drain and return finalized flows ready for Phase 4 feature
        extraction. Flows are removed from the internal buffer once
        returned -- calling this repeatedly does not return duplicates.
        """
        with self._lock:
            if limit is None:
                drained, self._expired_flows = self._expired_flows, []
            else:
                drained = self._expired_flows[:limit]
                self._expired_flows = self._expired_flows[limit:]
            return drained

    @property
    def active_flow_count(self) -> int:
        with self._lock:
            return len(self._flows)

    def stats_snapshot(self) -> dict[str, int | float | bool]:
        """Point-in-time snapshot, safe to serialize for an API/dashboard response."""
        with self._lock:
            return {
                "is_running": self.is_running,
                "active_flow_count": len(self._flows),
                "pending_expired_flow_count": len(self._expired_flows),
                "total_flows_created": self.stats.total_flows_created,
                "total_flows_expired": self.stats.total_flows_expired,
                "total_packets_ingested": self.stats.total_packets_ingested,
                "queue_timeouts": self.stats.queue_timeouts,
                "flow_timeout_seconds": self.flow_timeout_seconds,
                "flow_table_max_size": self.flow_table_max_size,
            }

    # ------------------------------------------------------------------
    # Background consumer thread
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background consumer thread. Safe to call again once already running (no-op)."""
        with self._lifecycle_lock:
            if self.is_running:
                logger.warning("FlowAggregator.start() called while already running -- ignoring.")
                return
            self._stop_event.clear()
            self._last_expiry_check = time.time()
            self._thread = threading.Thread(
                target=self._run_consumer_loop,
                name="nids-flow-aggregator",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "Flow aggregator started (flow_timeout_seconds=%.1f, flow_table_max_size=%d)",
                self.flow_timeout_seconds,
                self.flow_table_max_size,
            )

    def stop(self, timeout: float = 5.0, finalize_remaining: bool = True) -> list[FinalizedFlow]:
        """
        Signal the consumer loop to stop and wait (up to `timeout` seconds)
        for it to exit. If `finalize_remaining` is True (default), every
        still-active flow is force-expired on shutdown so no in-progress
        flow data is silently lost -- callers that care can retrieve it via
        `get_expired_flows()` afterward.
        """
        with self._lifecycle_lock:
            if not self.is_running:
                return []
            self._stop_event.set()
            assert self._thread is not None
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Flow aggregator thread did not stop within %.1fs timeout.", timeout)
            self._thread = None

        remaining: list[FinalizedFlow] = []
        if finalize_remaining:
            remaining = self.force_expire_all()

        logger.info(
            "Flow aggregator stopped. Active=%d Created=%d Expired=%d",
            self.active_flow_count,
            self.stats.total_flows_created,
            self.stats.total_flows_expired,
        )
        return remaining

    def _run_consumer_loop(self) -> None:
        """
        Repeatedly pull packets off the source queue with a short timeout so
        the stop event is checked frequently (mirrors PacketCapture's sniff
        loop pattern), running periodic idle-flow expiration sweeps in
        between rather than on every single packet.
        """
        while not self._stop_event.is_set():
            try:
                pkt = self.source_queue.get(timeout=self.queue_get_timeout)
            except queue.Empty:
                self.stats.queue_timeouts += 1
                self._maybe_expire(time.time())
                continue

            try:
                self.ingest_packet(pkt)
            except Exception:  # pragma: no cover - defensive; malformed metadata shouldn't kill the loop
                logger.exception("Failed to ingest a packet into the flow table -- skipping it.")
            finally:
                self.source_queue.task_done()

            self._maybe_expire(time.time())

    def _maybe_expire(self, now: float) -> None:
        if (now - self._last_expiry_check) >= self.expiry_check_interval_seconds:
            self.expire_inactive_flows(now=now)
            self._last_expiry_check = now
