"""
The Phase 4 feature schema: a fixed, versioned, ordered contract between
real-time inference and offline training.

THIS FILE IS THE SOURCE OF TRUTH for feature ordering. Nothing else in the
codebase should hardcode a feature index or count -- import FEATURE_SCHEMA,
FEATURE_NAMES, or FEATURE_COUNT from here.

Full field-by-field documentation (definition, unit, directionality,
edge-case handling, live vs. dataset availability) lives in
docs/methodology.md; this file carries a short human-readable description
per feature plus the machine-checkable metadata needed to keep the vector
length and dtype expectations honest.

Design principle (per project requirement): this is not a "collect anything
that sounds useful" feature list. Every feature here is either directly
requested in the project's original feature list (Section 7) or is a
CICFlowMeter-convention split (e.g. forward/backward packet length, in
addition to the combined figure) needed for compatibility with the eventual
CIC-IDS2017 training data. Three previously-planned features are explicitly
EXCLUDED here with a documented reason -- see the "Excluded" section below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FeatureDtype(str, Enum):
    FLOAT = "float"
    INT = "int"


class Directionality(str, Enum):
    """Whether a feature considers all packets, only forward, or only backward packets."""

    OVERALL = "overall"
    FORWARD = "forward"
    BACKWARD = "backward"
    NOT_APPLICABLE = "n/a"  # e.g. dst_port, protocol one-hot


@dataclass(frozen=True)
class FeatureSpec:
    """One row of the feature contract."""

    index: int
    name: str
    dtype: FeatureDtype
    unit: str
    directionality: Directionality
    description: str
    zero_case_handling: str
    live_available: bool
    dataset_available: bool


FEATURE_SCHEMA_VERSION = "1.0.0"

# fmt: off
FEATURE_SCHEMA: tuple[FeatureSpec, ...] = (
    FeatureSpec(0, "dst_port", FeatureDtype.INT, "port number", Directionality.NOT_APPLICABLE,
        "Destination port of the flow's originating packet. Source port is deliberately excluded: "
        "it's ephemeral/high-cardinality and not predictive on its own -- CICFlowMeter drops it too.",
        "-1 for protocols without ports (e.g. ICMP).", True, True),
    FeatureSpec(1, "protocol_is_tcp", FeatureDtype.INT, "boolean (0/1)", Directionality.NOT_APPLICABLE,
        "One-hot: 1 if the flow's transport protocol is TCP, else 0.",
        "n/a -- always defined.", True, True),
    FeatureSpec(2, "protocol_is_udp", FeatureDtype.INT, "boolean (0/1)", Directionality.NOT_APPLICABLE,
        "One-hot: 1 if the flow's transport protocol is UDP, else 0.",
        "n/a -- always defined.", True, True),
    FeatureSpec(3, "protocol_is_icmp", FeatureDtype.INT, "boolean (0/1)", Directionality.NOT_APPLICABLE,
        "One-hot: 1 if the flow's transport protocol is ICMP, else 0. Any other protocol "
        "(TransportProtocol.OTHER) is represented as all three one-hot columns being 0.",
        "n/a -- always defined.", True, True),
    FeatureSpec(4, "flow_duration_seconds", FeatureDtype.FLOAT, "seconds", Directionality.OVERALL,
        "last_seen - first_seen for the flow.",
        "0.0 for a single-packet flow (first_seen == last_seen).", True, True),
    FeatureSpec(5, "total_packets", FeatureDtype.INT, "count", Directionality.OVERALL,
        "Total number of packets observed in the flow (both directions).",
        "Always >= 1 -- a FinalizedFlow cannot exist with zero packets.", True, True),
    FeatureSpec(6, "total_bytes", FeatureDtype.INT, "bytes", Directionality.OVERALL,
        "Total bytes observed in the flow (both directions), on-the-wire packet length.",
        "Always >= 1.", True, True),
    FeatureSpec(7, "total_forward_packets", FeatureDtype.INT, "count", Directionality.FORWARD,
        "Packets traveling in the same direction as the flow's originating packet.",
        "Always >= 1 (the originating packet itself is forward).", True, True),
    FeatureSpec(8, "total_backward_packets", FeatureDtype.INT, "count", Directionality.BACKWARD,
        "Packets traveling in the reverse direction of the flow's originating packet.",
        "0 for a one-directional flow (e.g. an unanswered probe).", True, True),
    FeatureSpec(9, "total_forward_bytes", FeatureDtype.INT, "bytes", Directionality.FORWARD,
        "Sum of on-the-wire lengths of forward packets.",
        "Always >= 1.", True, True),
    FeatureSpec(10, "total_backward_bytes", FeatureDtype.INT, "bytes", Directionality.BACKWARD,
        "Sum of on-the-wire lengths of backward packets.",
        "0 for a one-directional flow.", True, True),
    FeatureSpec(11, "packet_length_mean", FeatureDtype.FLOAT, "bytes", Directionality.OVERALL,
        "Mean packet length across all packets in the flow. Note: mathematically equal to "
        "total_bytes / total_packets, kept as its own feature for readability and dataset-column parity.",
        "Equal to the single packet's length for a 1-packet flow.", True, True),
    FeatureSpec(12, "packet_length_std", FeatureDtype.FLOAT, "bytes", Directionality.OVERALL,
        "Population standard deviation of packet length across all packets.",
        "0.0 for a flow with fewer than 2 packets.", True, True),
    FeatureSpec(13, "packet_length_min", FeatureDtype.FLOAT, "bytes", Directionality.OVERALL,
        "Minimum packet length observed in the flow.",
        "Equal to the single packet's length for a 1-packet flow.", True, True),
    FeatureSpec(14, "packet_length_max", FeatureDtype.FLOAT, "bytes", Directionality.OVERALL,
        "Maximum packet length observed in the flow.",
        "Equal to the single packet's length for a 1-packet flow.", True, True),
    FeatureSpec(15, "forward_packet_length_mean", FeatureDtype.FLOAT, "bytes", Directionality.FORWARD,
        "Mean packet length, forward packets only.",
        "0.0 if there are no forward packets (cannot occur -- see total_forward_packets).", True, True),
    FeatureSpec(16, "forward_packet_length_std", FeatureDtype.FLOAT, "bytes", Directionality.FORWARD,
        "Population standard deviation of packet length, forward packets only.",
        "0.0 for fewer than 2 forward packets.", True, True),
    FeatureSpec(17, "forward_packet_length_min", FeatureDtype.FLOAT, "bytes", Directionality.FORWARD,
        "Minimum packet length, forward packets only.",
        "Equal to the single forward packet's length if there's only one.", True, True),
    FeatureSpec(18, "forward_packet_length_max", FeatureDtype.FLOAT, "bytes", Directionality.FORWARD,
        "Maximum packet length, forward packets only.",
        "Equal to the single forward packet's length if there's only one.", True, True),
    FeatureSpec(19, "backward_packet_length_mean", FeatureDtype.FLOAT, "bytes", Directionality.BACKWARD,
        "Mean packet length, backward packets only.",
        "0.0 if there are no backward packets (one-directional flow).", True, True),
    FeatureSpec(20, "backward_packet_length_std", FeatureDtype.FLOAT, "bytes", Directionality.BACKWARD,
        "Population standard deviation of packet length, backward packets only.",
        "0.0 for fewer than 2 backward packets, including zero.", True, True),
    FeatureSpec(21, "backward_packet_length_min", FeatureDtype.FLOAT, "bytes", Directionality.BACKWARD,
        "Minimum packet length, backward packets only.",
        "0.0 if there are no backward packets.", True, True),
    FeatureSpec(22, "backward_packet_length_max", FeatureDtype.FLOAT, "bytes", Directionality.BACKWARD,
        "Maximum packet length, backward packets only.",
        "0.0 if there are no backward packets.", True, True),
    FeatureSpec(23, "flow_iat_mean", FeatureDtype.FLOAT, "seconds", Directionality.OVERALL,
        "Mean inter-arrival time between consecutive packets, regardless of direction.",
        "0.0 for a flow with fewer than 2 packets (no gap to measure).", True, True),
    FeatureSpec(24, "flow_iat_std", FeatureDtype.FLOAT, "seconds", Directionality.OVERALL,
        "Population standard deviation of inter-arrival time, regardless of direction.",
        "0.0 for fewer than 3 packets (need >=2 gaps for a variance to exist).", True, True),
    FeatureSpec(25, "flow_iat_min", FeatureDtype.FLOAT, "seconds", Directionality.OVERALL,
        "Minimum inter-arrival time observed.",
        "0.0 for a flow with fewer than 2 packets.", True, True),
    FeatureSpec(26, "flow_iat_max", FeatureDtype.FLOAT, "seconds", Directionality.OVERALL,
        "Maximum inter-arrival time observed.",
        "0.0 for a flow with fewer than 2 packets.", True, True),
    FeatureSpec(27, "forward_iat_mean", FeatureDtype.FLOAT, "seconds", Directionality.FORWARD,
        "Mean inter-arrival time between consecutive FORWARD packets only.",
        "0.0 for fewer than 2 forward packets.", True, True),
    FeatureSpec(28, "forward_iat_std", FeatureDtype.FLOAT, "seconds", Directionality.FORWARD,
        "Population standard deviation of inter-arrival time, forward packets only.",
        "0.0 for fewer than 3 forward packets.", True, True),
    FeatureSpec(29, "backward_iat_mean", FeatureDtype.FLOAT, "seconds", Directionality.BACKWARD,
        "Mean inter-arrival time between consecutive BACKWARD packets only.",
        "0.0 for fewer than 2 backward packets, including zero backward packets.", True, True),
    FeatureSpec(30, "backward_iat_std", FeatureDtype.FLOAT, "seconds", Directionality.BACKWARD,
        "Population standard deviation of inter-arrival time, backward packets only.",
        "0.0 for fewer than 3 backward packets.", True, True),
    FeatureSpec(31, "syn_count", FeatureDtype.INT, "count", Directionality.OVERALL,
        "Number of packets in the flow with the TCP SYN flag set.",
        "0 for non-TCP flows.", True, True),
    FeatureSpec(32, "ack_count", FeatureDtype.INT, "count", Directionality.OVERALL,
        "Number of packets in the flow with the TCP ACK flag set.",
        "0 for non-TCP flows.", True, True),
    FeatureSpec(33, "fin_count", FeatureDtype.INT, "count", Directionality.OVERALL,
        "Number of packets in the flow with the TCP FIN flag set.",
        "0 for non-TCP flows.", True, True),
    FeatureSpec(34, "rst_count", FeatureDtype.INT, "count", Directionality.OVERALL,
        "Number of packets in the flow with the TCP RST flag set.",
        "0 for non-TCP flows.", True, True),
    FeatureSpec(35, "psh_count", FeatureDtype.INT, "count", Directionality.OVERALL,
        "Number of packets in the flow with the TCP PSH flag set.",
        "0 for non-TCP flows.", True, True),
    FeatureSpec(36, "urg_count", FeatureDtype.INT, "count", Directionality.OVERALL,
        "Number of packets in the flow with the TCP URG flag set.",
        "0 for non-TCP flows.", True, True),
    FeatureSpec(37, "forward_packets_per_second", FeatureDtype.FLOAT, "packets/sec", Directionality.FORWARD,
        "total_forward_packets / flow_duration_seconds.",
        "Defined as 0.0 (not NaN/Inf) when flow_duration_seconds == 0 -- i.e. a single-timestamp "
        "flow's rate is undefined and we define it as 0 by convention. See docs/methodology.md "
        "for why this needs validating against how the training dataset handles the same case.",
        True, True),
    FeatureSpec(38, "backward_packets_per_second", FeatureDtype.FLOAT, "packets/sec", Directionality.BACKWARD,
        "total_backward_packets / flow_duration_seconds.",
        "0.0 when flow_duration_seconds == 0, same convention as forward_packets_per_second.",
        True, True),
    FeatureSpec(39, "total_packets_per_second", FeatureDtype.FLOAT, "packets/sec", Directionality.OVERALL,
        "total_packets / flow_duration_seconds.",
        "0.0 when flow_duration_seconds == 0.", True, True),
    FeatureSpec(40, "total_bytes_per_second", FeatureDtype.FLOAT, "bytes/sec", Directionality.OVERALL,
        "total_bytes / flow_duration_seconds.",
        "0.0 when flow_duration_seconds == 0.", True, True),
)
# fmt: on

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SCHEMA)
FEATURE_COUNT: int = len(FEATURE_SCHEMA)

# Sanity-check the schema is internally consistent at import time -- indices
# must be contiguous starting at 0, and names must be unique. A bug here
# would silently corrupt every downstream feature vector, so fail loudly
# and immediately instead.
assert [spec.index for spec in FEATURE_SCHEMA] == list(range(FEATURE_COUNT)), "FEATURE_SCHEMA indices must be contiguous from 0"
assert len(set(FEATURE_NAMES)) == FEATURE_COUNT, "FEATURE_SCHEMA names must be unique"


# ---------------------------------------------------------------------------
# Explicitly excluded features (documented, not silently dropped)
# ---------------------------------------------------------------------------
#
# init_win_bytes_forward / init_win_bytes_backward
#   Planned in Phase 1 and incorrectly marked "Live Available" in that
#   planning table. On inspection, PacketMetadata (Phase 2) does not
#   capture TCP window size at all. Adding it would require extending
#   PacketMetadata and extract_packet_metadata() in Phase 2 -- deferred
#   pending a decision on whether the feature is worth that change.
#
# active_mean / idle_mean
#   CICFlowMeter defines these by splitting a flow into active bursts
#   separated by idle gaps above a threshold, then averaging burst/gap
#   durations. This requires meaningfully more per-flow state than a
#   streaming accumulator (tracking burst boundaries as they happen) and
#   risks a subtly wrong reimplementation of CICFlowMeter's exact algorithm.
#   Deferred rather than guessed at.
#
# subflow_forward_bytes / subflow_backward_bytes
#   Already excluded in the Phase 1 feature-compatibility table: not
#   faithfully reproducible from live capture without re-implementing
#   CICFlowMeter's subflow segmentation logic.
