"""
Deterministic feature extraction: FinalizedFlow -> FeatureVector.

This is the ONLY place that should read FinalizedFlow fields to build a
feature vector -- both the (future) real-time inference path and the
(future) offline training pipeline should route through the same
FeatureExtractor so that "the same feature vector" from the project's own
requirement is enforced by code, not by convention.
"""

from __future__ import annotations

from app.capture.models import TransportProtocol
from app.features.models import FeatureVector
from app.features.schema import FEATURE_SCHEMA_VERSION
from app.flows.models import FinalizedFlow


class FeatureExtractionError(ValueError):
    """Raised when a FinalizedFlow is structurally invalid for feature extraction."""


class FeatureExtractor:
    """
    Stateless, deterministic converter from a single FinalizedFlow to a
    single FeatureVector. Stateless by design: calling extract() twice on
    the same (immutable) FinalizedFlow must produce bit-identical output.
    """

    def extract(self, flow: FinalizedFlow) -> FeatureVector:
        self._validate(flow)

        duration = flow.duration_seconds
        rate_denominator = duration if duration > 0 else None  # None signals "use the 0.0 convention"

        values: list[float] = [
            float(flow.origin_dst_port) if flow.origin_dst_port is not None else -1.0,
            1.0 if flow.key.protocol == TransportProtocol.TCP else 0.0,
            1.0 if flow.key.protocol == TransportProtocol.UDP else 0.0,
            1.0 if flow.key.protocol == TransportProtocol.ICMP else 0.0,
            duration,
            float(flow.packet_count),
            float(flow.byte_count),
            float(flow.forward_packet_count),
            float(flow.backward_packet_count),
            float(flow.forward_byte_count),
            float(flow.backward_byte_count),
            flow.packet_length_mean,
            flow.packet_length_std,
            flow.packet_length_min,
            flow.packet_length_max,
            flow.forward_packet_length_mean,
            flow.forward_packet_length_std,
            flow.forward_packet_length_min,
            flow.forward_packet_length_max,
            flow.backward_packet_length_mean,
            flow.backward_packet_length_std,
            flow.backward_packet_length_min,
            flow.backward_packet_length_max,
            flow.flow_iat_mean,
            flow.flow_iat_std,
            flow.flow_iat_min,
            flow.flow_iat_max,
            flow.forward_iat_mean,
            flow.forward_iat_std,
            flow.backward_iat_mean,
            flow.backward_iat_std,
            float(flow.syn_count),
            float(flow.ack_count),
            float(flow.fin_count),
            float(flow.rst_count),
            float(flow.psh_count),
            float(flow.urg_count),
            self._safe_rate(flow.forward_packet_count, rate_denominator),
            self._safe_rate(flow.backward_packet_count, rate_denominator),
            self._safe_rate(flow.packet_count, rate_denominator),
            self._safe_rate(flow.byte_count, rate_denominator),
        ]

        return FeatureVector(
            values=tuple(values),
            schema_version=FEATURE_SCHEMA_VERSION,
            flow_key_repr=repr(flow.key),
        )

    @staticmethod
    def _safe_rate(numerator: float, denominator: float | None) -> float:
        """
        numerator / denominator, defined as 0.0 (never NaN/Inf) when the
        denominator is zero/None. See FeatureSpec entries for
        *_packets_per_second / *_bytes_per_second in schema.py for the
        rationale and the open question this raises for dataset comparison.
        """
        if denominator is None or denominator <= 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _validate(flow: FinalizedFlow) -> None:
        if flow.packet_count <= 0:
            raise FeatureExtractionError(
                "Cannot extract features from a flow with packet_count <= 0 -- "
                "this should be structurally impossible (Flow.start() always ingests "
                "at least one packet), so seeing it indicates a bug upstream."
            )
        if flow.last_seen < flow.first_seen:
            raise FeatureExtractionError(
                f"Invalid flow: last_seen ({flow.last_seen}) precedes first_seen ({flow.first_seen})."
            )
        if flow.packet_count != flow.forward_packet_count + flow.backward_packet_count:
            raise FeatureExtractionError(
                "Invalid flow: total_packets does not equal forward_packets + backward_packets "
                f"({flow.packet_count} != {flow.forward_packet_count} + {flow.backward_packet_count})."
            )
