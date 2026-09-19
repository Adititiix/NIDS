"""
The FeatureVector model: the fixed-length, ordered numeric representation
of one flow, ready to hand to a model at either training or inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.features.schema import FEATURE_COUNT, FEATURE_NAMES, FEATURE_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """
    An immutable, fixed-length feature vector plus enough metadata to trace
    it back to the flow it came from and the schema version that produced
    it. `schema_version` matters because a model trained against one schema
    version must never silently be fed vectors from a different one --
    Phase 7/8 wiring should check this before running inference.
    """

    values: tuple[float, ...]
    schema_version: str
    flow_key_repr: str  # str(FlowKey) -- for traceability/debugging, not itself a feature

    def __post_init__(self) -> None:
        if len(self.values) != FEATURE_COUNT:
            raise ValueError(f"FeatureVector expected {FEATURE_COUNT} values, got {len(self.values)}")

    def to_list(self) -> list[float]:
        """Plain list, in schema order -- what a scikit-learn model's .predict() expects."""
        return list(self.values)

    def to_dict(self) -> dict[str, float]:
        """Name -> value mapping, in schema order. Useful for logging, CSV export, and debugging."""
        return dict(zip(FEATURE_NAMES, self.values, strict=True))

    def to_json(self) -> str:
        """
        Serialize for storage or transport (e.g. a future database JSONB
        column, or a message to a model-serving process). Includes the
        schema version so a consumer can detect a mismatch rather than
        silently misinterpreting the vector.
        """
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "flow_key": self.flow_key_repr,
                "features": self.to_dict(),
            }
        )
