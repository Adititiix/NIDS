"""
Live packet capture engine.

Responsibilities (Phase 2 scope):
- Detect/validate the capture interface: accept an explicit override
  (CAPTURE_INTERFACE in .env, e.g. "eth0") and validate it actually exists,
  or auto-detect a sensible default and validate that too.
- Run a Scapy sniff loop on a background thread in short timed slices, so
  it (a) never blocks the rest of the backend and (b) can be stopped
  promptly even on a quiet interface, rather than blocking indefinitely
  inside a single sniff() call waiting for the next packet.
- Push extracted PacketMetadata onto a bounded queue.Queue for downstream
  consumers -- flow aggregation is added in Phase 3 and will be the thing
  actually calling `.queue.get()`.
- Count observed/processed/dropped packets and expose packet-rate stats.
- Start/stop cleanly.

Explicitly OUT of scope for Phase 2: flow aggregation, feature extraction,
detection, and anything that writes to the database.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from scapy.config import conf
from scapy.interfaces import get_if_list, get_working_ifaces
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.packet import Packet
from scapy.sendrecv import sniff

from app.capture.models import CaptureStats, PacketMetadata, TransportProtocol
from app.config import settings

logger = logging.getLogger("nids.capture")

# How long each individual sniff() call is allowed to block before returning
# control to the loop, which then re-checks the stop event. Small enough for
# prompt shutdown, large enough not to thrash on a quiet interface.
_SNIFF_SLICE_TIMEOUT_SECONDS = 1.0


class InterfaceNotFoundError(RuntimeError):
    """Raised when the configured or auto-detected interface doesn't exist / isn't usable."""


def list_available_interfaces() -> list[str]:
    """Return interface names Scapy can see on this host."""
    return get_if_list()


def detect_default_interface() -> str:
    """
    Auto-detect a reasonable capture interface.

    Uses Scapy's notion of the default-route interface (conf.iface), falling
    back to the first "working" interface Scapy reports if that lookup
    fails or returns something unusable. This is a convenience default for
    lab use -- CAPTURE_INTERFACE in .env always takes priority when it's set
    to a valid interface (see `resolve_interface`).
    """
    try:
        default_iface = conf.iface
        if default_iface:
            return str(default_iface)
    except Exception as exc:  # pragma: no cover - defensive, platform dependent
        logger.warning("Default-route interface lookup failed: %s", exc)

    working = get_working_ifaces()
    if working:
        return str(working[0].name)

    raise InterfaceNotFoundError("Could not auto-detect any usable network interface.")


def resolve_interface(requested: str | None = None) -> str:
    """
    Resolve and validate the interface to capture on.

    Priority:
    1. Explicit `requested` argument (lets callers/tests override directly).
    2. CAPTURE_INTERFACE from settings (.env), e.g. "eth0" -- validated
       against the interfaces Scapy can actually see on this host.
    3. Auto-detected default-route interface, also validated.

    Raises InterfaceNotFoundError if nothing resolvable is found.
    """
    available = list_available_interfaces()
    candidate = requested or settings.capture_interface

    if candidate:
        if candidate in available:
            logger.info("Using configured capture interface: %s", candidate)
            return candidate
        logger.warning(
            "Configured/requested interface '%s' not found among available interfaces %s -- falling back to auto-detection.",
            candidate,
            available,
        )

    detected = detect_default_interface()
    if detected not in available:
        raise InterfaceNotFoundError(
            f"Auto-detected interface '{detected}' is not in the available interface list {available}."
        )
    logger.info("Auto-detected capture interface: %s", detected)
    return detected


def _protocol_and_ports(pkt: Packet) -> tuple[TransportProtocol, int | None, int | None, str | None]:
    """Extract transport protocol, ports, and TCP flags from a Scapy packet."""
    if pkt.haslayer(TCP):
        tcp_layer = pkt[TCP]
        return TransportProtocol.TCP, int(tcp_layer.sport), int(tcp_layer.dport), str(tcp_layer.flags)
    if pkt.haslayer(UDP):
        udp_layer = pkt[UDP]
        return TransportProtocol.UDP, int(udp_layer.sport), int(udp_layer.dport), None
    if pkt.haslayer(ICMP):
        return TransportProtocol.ICMP, None, None, None
    return TransportProtocol.OTHER, None, None, None


def extract_packet_metadata(pkt: Packet) -> PacketMetadata | None:
    """
    Convert a raw Scapy packet into PacketMetadata.

    Returns None for packets without an IP layer (e.g. bare ARP). This
    project's detection is IP-flow based (see the feature-compatibility
    table in docs/methodology.md), so non-IP traffic is counted as
    "observed" in CaptureStats but not enqueued for further processing.
    """
    if not pkt.haslayer(IP):
        return None

    ip_layer = pkt[IP]
    protocol, src_port, dst_port, tcp_flags = _protocol_and_ports(pkt)

    return PacketMetadata(
        timestamp=float(pkt.time),
        src_ip=str(ip_layer.src),
        dst_ip=str(ip_layer.dst),
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        length=int(len(pkt)),
        tcp_flags=tcp_flags,
    )


class PacketCapture:
    """
    Runs a Scapy sniff loop on a background thread and feeds extracted
    PacketMetadata into a bounded queue for downstream consumers.

    Usage:
        capture = PacketCapture()   # interface resolved from .env / auto-detect
        capture.start()
        ...
        metadata = capture.queue.get()
        ...
        capture.stop()
    """

    def __init__(self, interface: str | None = None, queue_max_size: int | None = None) -> None:
        self.interface = resolve_interface(interface)
        self.queue: queue.Queue[PacketMetadata] = queue.Queue(
            maxsize=queue_max_size or settings.capture_queue_max_size
        )
        self.stats = CaptureStats()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the sniff loop on a background thread. Safe to call again once already running (no-op)."""
        with self._lifecycle_lock:
            if self.is_running:
                logger.warning("PacketCapture.start() called while already running -- ignoring.")
                return

            self._stop_event.clear()
            self.stats = CaptureStats(start_time=time.time())
            self._thread = threading.Thread(
                target=self._run_sniff_loop,
                name="nids-packet-capture",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "Packet capture started on interface '%s' (promiscuous=%s, queue_max_size=%d)",
                self.interface,
                settings.capture_promiscuous,
                self.queue.maxsize,
            )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the sniff loop to stop and wait (up to `timeout` seconds) for the thread to exit."""
        with self._lifecycle_lock:
            if not self.is_running:
                return
            self._stop_event.set()
            assert self._thread is not None
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Packet capture thread did not stop within %.1fs timeout.", timeout)
            else:
                logger.info(
                    "Packet capture stopped. Observed=%d Processed=%d Dropped=%d (%.2f%% drop rate)",
                    self.stats.packets_observed,
                    self.stats.packets_processed,
                    self.stats.packets_dropped,
                    self.stats.packet_drop_rate_pct,
                )
            self._thread = None

    def stats_snapshot(self) -> dict[str, float | int]:
        """Point-in-time snapshot of capture stats, safe to serialize for an API/dashboard response."""
        return {
            "interface": self.interface,
            "is_running": self.is_running,
            "packets_observed": self.stats.packets_observed,
            "packets_processed": self.stats.packets_processed,
            "packets_dropped": self.stats.packets_dropped,
            "packet_drop_rate_pct": round(self.stats.packet_drop_rate_pct, 4),
            "packets_per_second": round(self.stats.packets_per_second, 2),
            "queue_size": self.queue.qsize(),
            "queue_max_size": self.queue.maxsize,
        }

    def _handle_packet(self, pkt: Packet) -> None:
        """Scapy sniff() callback: extract metadata and enqueue it. Must never block or raise."""
        self.stats.packets_observed += 1
        self.stats.last_packet_time = time.time()

        try:
            metadata = extract_packet_metadata(pkt)
        except Exception:  # pragma: no cover - defensive; malformed packets shouldn't crash the sniff loop
            logger.exception("Failed to extract metadata from a captured packet -- skipping it.")
            return

        if metadata is None:
            return

        try:
            self.queue.put_nowait(metadata)
            self.stats.packets_processed += 1
        except queue.Full:
            self.stats.packets_dropped += 1
            logger.debug(
                "Capture queue full (maxsize=%d) -- dropping packet %s:%s -> %s:%s.",
                self.queue.maxsize,
                metadata.src_ip,
                metadata.src_port,
                metadata.dst_ip,
                metadata.dst_port,
            )

    def _run_sniff_loop(self) -> None:
        """
        Repeatedly call sniff() in short timed slices so the stop event is
        checked frequently, instead of one long-blocking sniff() call that
        might not return until the next packet arrives on a quiet interface.
        """
        try:
            while not self._stop_event.is_set():
                sniff(
                    iface=self.interface,
                    prn=self._handle_packet,
                    store=False,
                    promisc=settings.capture_promiscuous,
                    timeout=_SNIFF_SLICE_TIMEOUT_SECONDS,
                )
        except PermissionError:
            logger.error(
                "Permission denied capturing on '%s'. Packet capture requires elevated privileges "
                "(run as root in the lab VM, or grant the capability: "
                "`sudo setcap cap_net_raw,cap_net_admin=eip $(readlink -f $(which python3))`).",
                self.interface,
            )
        except OSError as exc:
            logger.error("Capture on interface '%s' failed: %s", self.interface, exc)
