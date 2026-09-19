"""
The ML-trainable feature subset.

Real-dataset validation against the actual CIC-IDS2017 MachineLearningCSV
release (see docs/methodology.md, "CIC-IDS2017 real-dataset validation")
found that the public CSV files do NOT include a Protocol column. That
means `protocol_is_tcp` / `protocol_is_udp` / `protocol_is_icmp` -- three
of the live 41 features -- cannot be genuinely derived from that dataset.

Per this project's explicit rule (app.ml.feature_mapping module docstring):
if a dataset feature cannot be mapped reliably, do NOT silently substitute
or fabricate it. Guessing a protocol from destination port, or assuming
TCP by default, would be exactly that kind of fabrication. So these three
features are EXCLUDED from ML training, not faked.

THIS DOES NOT CHANGE THE LIVE 41-FEATURE SCHEMA. `app.features.schema`
remains the single source of truth for real-time detection/rules, and
protocol_is_tcp/udp/icmp remain fully available there -- for a live
captured packet, the transport protocol is always known with certainty
(app.capture.models.TransportProtocol), unlike a CIC-IDS2017 CSV row that
simply has no such column at all. The distinction this module encodes is:

    live feature schema (app.features.schema)   = 41 features, real-time only
    ML-trainable feature schema (this module)   = 38 features, offline training + live inference's model input

This module is the single source of truth for the ML-trainable subset --
nothing else should hardcode "38" or re-derive this list independently.
"""

from __future__ import annotations

from app.features.schema import FEATURE_NAMES

#: Bump this whenever EXCLUDED_ML_FEATURES (or the underlying live schema)
#: changes in a way that affects what a previously-trained model expects.
#: Distinct from app.features.schema.FEATURE_SCHEMA_VERSION -- that version
#: tracks the live 41-feature contract; this one tracks the ML-trainable
#: subset specifically, since the two can now change independently (e.g. a
#: future dataset gap could exclude a different feature without the live
#: schema itself changing at all).
ML_FEATURE_SCHEMA_VERSION = "1.0.0-ml38"

#: Live features excluded from ML training, with the reason why. Keyed so
#: the compatibility report and persisted model metadata can explain
#: EXACTLY which features are missing and why, rather than presenting an
#: unexplained gap.
EXCLUDED_ML_FEATURES: dict[str, str] = {
    "protocol_is_tcp": (
        "CIC-IDS2017's MachineLearningCSV release does not include a Protocol column. "
        "Fabricating a protocol value (e.g. guessing from destination port) would not be a "
        "genuine dataset-derived feature, so this is excluded from ML training rather than faked."
    ),
    "protocol_is_udp": "See protocol_is_tcp -- same dataset gap, same reason.",
    "protocol_is_icmp": "See protocol_is_tcp -- same dataset gap, same reason.",
}

#: The exact, fixed, ordered list of features a CIC-IDS2017-trained model is
#: trained on and expects at inference time -- a strict subset of
#: FEATURE_NAMES, in the same relative order. Derived automatically from
#: the live schema minus the excluded set, so it can never silently drift
#: out of sync with app.features.schema.
ML_FEATURE_NAMES: tuple[str, ...] = tuple(name for name in FEATURE_NAMES if name not in EXCLUDED_ML_FEATURES)

ML_FEATURE_COUNT: int = len(ML_FEATURE_NAMES)

assert ML_FEATURE_COUNT == len(FEATURE_NAMES) - len(EXCLUDED_ML_FEATURES), (
    "ML_FEATURE_NAMES must be exactly FEATURE_NAMES minus EXCLUDED_ML_FEATURES -- "
    "a mismatch here means an excluded feature name doesn't actually exist in the live schema."
)