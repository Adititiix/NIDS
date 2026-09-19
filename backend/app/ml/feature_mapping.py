"""
Dataset-column -> live-feature alignment.

This module is the answer to Phase 6's central requirement: the live
41-feature schema (app.features.schema.FEATURE_NAMES) is the single source
of truth for feature ordering, and this module explicitly maps a
CICFlowMeter-style CSV's columns onto that schema -- it does not assume
column names blindly, and it does not silently substitute or rename an
unrelated column to paper over a mismatch.

IMPORTANT DISTINCTION (added after real CIC-IDS2017 dataset validation):
this module now separately tracks LIVE feature availability (all 41, used
by real-time detection) from ML-TRAINABLE feature availability (38, see
app.ml.ml_feature_schema) -- because the real, publicly-released
CIC-IDS2017 MachineLearningCSV files have no Protocol column at all.
A dataset missing only protocol-derived features is NOT declared
incompatible: those three features are excluded from ML training
entirely, so their absence from a dataset doesn't block anything.
`FeatureAlignmentReport.is_ml_compatible` is the actual gate `align_features()`
uses; `missing_ml_features` (blocking) is kept distinct from
`unavailable_live_only_features` (informational, non-blocking).

Two known CICFlowMeter-specific quirks this module handles explicitly:

1. Column headers commonly have leading/trailing whitespace (e.g.
   " Destination Port") in the public CIC-IDS2017 CSV release --
   `normalize_columns()` strips this before any matching happens.
2. `Flow Duration`, `Flow IAT *`, `Fwd IAT *`, and `Bwd IAT *` are reported
   in MICROSECONDS in the raw CSVs, but the live feature schema (built from
   PacketMetadata.timestamp, a float seconds value) uses SECONDS. The scale
   factors in DIRECT_COLUMN_MAP below convert at load time so live and
   dataset values are in the same unit before a model ever sees them.

Everything in DIRECT_COLUMN_MAP, PROTOCOL_NUMBER_MAP, and the computed-
feature logic in `align_features()` is a documented, explicit assertion
about how a specific dataset column relates to a specific live feature --
NOT a guess. If a required ML-trainable feature can't be obtained this
way, it is reported as missing, not silently invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.capture.models import TransportProtocol
from app.features.schema import FEATURE_NAMES
from app.ml.ml_feature_schema import ML_FEATURE_COUNT, ML_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Column mapping tables (source of the actual mapping assertions)
# ---------------------------------------------------------------------------

#: CICFlowMeter CSV column name (AFTER whitespace-stripping) -> (live feature
#: name, multiplicative scale factor to convert the dataset's unit into the
#: live feature's unit). A scale of 1.0 means the units already match.
DIRECT_COLUMN_MAP: dict[str, tuple[str, float]] = {
    "Destination Port": ("dst_port", 1.0),
    "Flow Duration": ("flow_duration_seconds", 1e-6),  # dataset: microseconds -> live: seconds
    "Total Fwd Packets": ("total_forward_packets", 1.0),
    "Total Backward Packets": ("total_backward_packets", 1.0),
    "Total Length of Fwd Packets": ("total_forward_bytes", 1.0),
    "Total Length of Bwd Packets": ("total_backward_bytes", 1.0),
    "Fwd Packet Length Max": ("forward_packet_length_max", 1.0),
    "Fwd Packet Length Min": ("forward_packet_length_min", 1.0),
    "Fwd Packet Length Mean": ("forward_packet_length_mean", 1.0),
    "Fwd Packet Length Std": ("forward_packet_length_std", 1.0),
    "Bwd Packet Length Max": ("backward_packet_length_max", 1.0),
    "Bwd Packet Length Min": ("backward_packet_length_min", 1.0),
    "Bwd Packet Length Mean": ("backward_packet_length_mean", 1.0),
    "Bwd Packet Length Std": ("backward_packet_length_std", 1.0),
    "Flow Bytes/s": ("total_bytes_per_second", 1.0),
    "Flow Packets/s": ("total_packets_per_second", 1.0),
    "Flow IAT Mean": ("flow_iat_mean", 1e-6),
    "Flow IAT Std": ("flow_iat_std", 1e-6),
    "Flow IAT Max": ("flow_iat_max", 1e-6),
    "Flow IAT Min": ("flow_iat_min", 1e-6),
    "Fwd IAT Mean": ("forward_iat_mean", 1e-6),
    "Fwd IAT Std": ("forward_iat_std", 1e-6),
    "Bwd IAT Mean": ("backward_iat_mean", 1e-6),
    "Bwd IAT Std": ("backward_iat_std", 1e-6),
    "Fwd Packets/s": ("forward_packets_per_second", 1.0),
    "Bwd Packets/s": ("backward_packets_per_second", 1.0),
    "Min Packet Length": ("packet_length_min", 1.0),
    "Max Packet Length": ("packet_length_max", 1.0),
    "Packet Length Mean": ("packet_length_mean", 1.0),
    "Packet Length Std": ("packet_length_std", 1.0),
    "SYN Flag Count": ("syn_count", 1.0),
    "ACK Flag Count": ("ack_count", 1.0),
    "FIN Flag Count": ("fin_count", 1.0),
    "RST Flag Count": ("rst_count", 1.0),
    "PSH Flag Count": ("psh_count", 1.0),
    "URG Flag Count": ("urg_count", 1.0),
}

#: Live features that are DERIVED from other already-mapped live features
#: rather than copied from a single dataset column. Documented explicitly
#: (not hidden inside a generic "compute everything else" branch).
COMPUTED_FROM_OTHER_LIVE_FEATURES: dict[str, tuple[str, ...]] = {
    "total_packets": ("total_forward_packets", "total_backward_packets"),
    "total_bytes": ("total_forward_bytes", "total_backward_bytes"),
}

#: The CICFlowMeter "Protocol" column uses IANA protocol numbers.
#: 6 = TCP, 17 = UDP, 1 = ICMP. Anything else maps to all-zero one-hot,
#: matching the live TransportProtocol.OTHER convention (see
#: app.capture.models.TransportProtocol and app.features.schema).
PROTOCOL_COLUMN = "Protocol"
PROTOCOL_NUMBER_MAP: dict[int, TransportProtocol] = {
    6: TransportProtocol.TCP,
    17: TransportProtocol.UDP,
    1: TransportProtocol.ICMP,
}
PROTOCOL_ONE_HOT_FEATURES = ("protocol_is_tcp", "protocol_is_udp", "protocol_is_icmp")

#: Candidate label-column names (checked case-insensitively, after stripping).
LABEL_COLUMN_CANDIDATES = ("Label",)

#: Features intentionally excluded from the live schema (Phase 4 decision) --
#: listed here so the compatibility report can explain a dataset column
#: like "Init_Win_bytes_forward" as "known and deliberately unused", not an
#: unexplained gap.
KNOWN_UNUSED_DATASET_CONCEPTS = (
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "Active Mean",
    "Idle Mean",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from column names -- a well-known
    quirk of the public CIC-IDS2017 CSV release (e.g. " Destination Port").
    Returns a copy; does not mutate the input.
    """
    result = df.copy()
    result.columns = [str(c).strip() for c in result.columns]
    return result


@dataclass(frozen=True)
class FeatureAlignmentReport:
    """
    The result of comparing a dataset's columns against the live 41-feature
    schema, with the ML-trainable 38-feature subset (app.ml.ml_feature_schema)
    tracked separately.

    `is_ml_compatible` -- not full 41-feature coverage -- is the actual gate
    `align_features()` uses to decide whether to proceed. A dataset (like
    the real CIC-IDS2017 MachineLearningCSV release) that's missing ONLY
    features already excluded from ML training (protocol_is_tcp/udp/icmp,
    because that dataset has no Protocol column at all) is fully
    ML-compatible -- `missing_ml_features` will be empty and
    `unavailable_live_only_features` will report the excluded features
    informationally, not as a blocker.
    """

    matched_direct_features: tuple[str, ...]
    matched_computed_features: tuple[str, ...]
    matched_protocol_features: tuple[str, ...]
    missing_ml_features: tuple[str, ...]
    unavailable_live_only_features: tuple[str, ...]
    unmapped_dataset_columns: tuple[str, ...]
    label_column_found: str | None

    @property
    def missing_live_features(self) -> tuple[str, ...]:
        """All live (41-schema) features not obtainable from this dataset -- both blocking and informational."""
        return self.missing_ml_features + self.unavailable_live_only_features

    @property
    def matched_live_features(self) -> tuple[str, ...]:
        return self.matched_direct_features + self.matched_computed_features + self.matched_protocol_features

    @property
    def is_ml_compatible(self) -> bool:
        """True if every ML-trainable feature (see app.ml.ml_feature_schema) can be obtained -- the real training gate."""
        return len(self.missing_ml_features) == 0 and self.label_column_found is not None

    def summary(self) -> str:
        ml_obtained = ML_FEATURE_COUNT - len(self.missing_ml_features)
        lines = [
            f"Feature compatibility: {len(self.matched_live_features)}/{len(FEATURE_NAMES)} live features obtainable "
            f"({ml_obtained}/{ML_FEATURE_COUNT} ML-trainable features obtainable).",
            f"  Direct column matches: {len(self.matched_direct_features)}",
            f"  Computed from other matched features: {len(self.matched_computed_features)}",
            f"  Protocol one-hot derived: {len(self.matched_protocol_features)} "
            f"(informational only -- protocol features are excluded from ML training regardless of "
            f"whether a dataset provides them; see app.ml.ml_feature_schema).",
        ]
        if self.missing_ml_features:
            lines.append(f"  MISSING ML FEATURES (blocks training): {', '.join(self.missing_ml_features)}")
        if self.unavailable_live_only_features:
            lines.append(
                "  Live-only features unavailable from this dataset, excluded from ML training anyway "
                f"(NOT blocking): {', '.join(self.unavailable_live_only_features)}"
            )
        if self.label_column_found is None:
            lines.append("  MISSING: no recognizable label column found.")
        else:
            lines.append(f"  Label column: '{self.label_column_found}'")
        lines.append(f"  ML-trainable: {self.is_ml_compatible}")
        return "\n".join(lines)


def build_compatibility_report(df: pd.DataFrame) -> FeatureAlignmentReport:
    """
    Inspect a (column-normalized) dataset and report exactly which of the
    41 live features -- and, separately, which of the 38 ML-trainable
    features -- can be obtained from it, without modifying the dataframe.
    Safe to call before deciding whether to proceed with alignment/training
    at all.
    """
    columns = set(df.columns)

    matched_direct = tuple(
        live_name for col, (live_name, _scale) in DIRECT_COLUMN_MAP.items() if col in columns
    )
    matched_direct_set = set(matched_direct)

    matched_computed = tuple(
        live_name
        for live_name, deps in COMPUTED_FROM_OTHER_LIVE_FEATURES.items()
        if all(dep in matched_direct_set for dep in deps)
    )

    matched_protocol = PROTOCOL_ONE_HOT_FEATURES if PROTOCOL_COLUMN in columns else ()

    obtained = matched_direct_set | set(matched_computed) | set(matched_protocol)
    missing_live = tuple(name for name in FEATURE_NAMES if name not in obtained)

    ml_feature_set = set(ML_FEATURE_NAMES)
    missing_ml = tuple(name for name in missing_live if name in ml_feature_set)
    unavailable_live_only = tuple(name for name in missing_live if name not in ml_feature_set)

    used_columns = {col for col in DIRECT_COLUMN_MAP if col in columns}
    if PROTOCOL_COLUMN in columns:
        used_columns.add(PROTOCOL_COLUMN)
    label_column = next((c for c in LABEL_COLUMN_CANDIDATES if c in columns), None)
    if label_column is not None:
        used_columns.add(label_column)
    unmapped = tuple(sorted(columns - used_columns))

    return FeatureAlignmentReport(
        matched_direct_features=matched_direct,
        matched_computed_features=matched_computed,
        matched_protocol_features=matched_protocol,
        missing_ml_features=missing_ml,
        unavailable_live_only_features=unavailable_live_only,
        unmapped_dataset_columns=unmapped,
        label_column_found=label_column,
    )


class FeatureAlignmentError(ValueError):
    """Raised when a dataset cannot be reliably aligned to the ML-trainable feature subset."""


def align_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, FeatureAlignmentReport]:
    """
    Produce a DataFrame with EXACTLY the columns in ML_FEATURE_NAMES (the
    38-feature ML-trainable subset of the live 41-feature schema -- see
    app.ml.ml_feature_schema), in that exact order, plus the raw label
    Series, from a raw (not yet column-normalized) dataset.

    Raises FeatureAlignmentError if the dataset cannot supply every
    ML-trainable feature or a recognizable label column. This function
    never silently substitutes an unrelated column or invents a missing
    feature -- in particular, protocol_is_tcp/udp/icmp are NEVER fabricated
    here even when a dataset happens to provide a Protocol column: the
    ML-trainable feature set is fixed (ML_FEATURE_NAMES), so a model always
    trains on exactly the same 38 features regardless of what extra
    columns a particular dataset variant might offer. Call
    `build_compatibility_report()` first if you want to inspect
    compatibility without risking the exception.
    """
    normalized = normalize_columns(df)
    report = build_compatibility_report(normalized)

    if not report.is_ml_compatible:
        raise FeatureAlignmentError(
            "Dataset is not compatible with the ML-trainable feature subset:\n" + report.summary()
        )

    aligned = pd.DataFrame(index=normalized.index)

    for col, (live_name, scale) in DIRECT_COLUMN_MAP.items():
        if col in normalized.columns and live_name in ML_FEATURE_NAMES:
            aligned[live_name] = pd.to_numeric(normalized[col], errors="coerce") * scale

    for live_name, deps in COMPUTED_FROM_OTHER_LIVE_FEATURES.items():
        if live_name in report.matched_computed_features and live_name in ML_FEATURE_NAMES:
            aligned[live_name] = sum(aligned[dep] for dep in deps)

    # protocol_is_tcp/udp/icmp are deliberately NOT computed here, even if
    # this particular dataset happens to have a Protocol column -- see the
    # docstring above.

    # Reorder to match the ML-trainable schema EXACTLY -- this is the whole point.
    aligned = aligned[list(ML_FEATURE_NAMES)]

    assert report.label_column_found is not None  # guaranteed by is_ml_compatible check above
    labels = normalized[report.label_column_found]

    return aligned, labels, report