"""
Base interface for a single-flow, stateless detection rule.

Every rule takes the same two inputs -- the flow's extracted features as a
name -> value dict (in `app.features.schema.FEATURE_NAMES` order, from
`FeatureVector.to_dict()`) and the underlying `FinalizedFlow` for anything
not itself a feature (e.g. the flow key, for reporting) -- and returns
either `None` (the rule did not fire) or a `RuleTrigger` describing what it
found.

Rules are intentionally STATELESS and operate on a SINGLE flow at a time.
Cross-flow correlation (e.g. "how many distinct ports has this source IP
touched in the last 10 seconds") needs a stateful, multi-flow tracking
layer that doesn't exist yet -- see docs/methodology.md. The pre-existing
`DetectionThresholds.port_scan_*`, `.high_conn_rate_per_sec`, and
`.packet_rate_zscore_threshold` config fields are reserved for that future
layer and are deliberately NOT read by any rule in this package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.detection.models import RuleTrigger
from app.flows.models import FinalizedFlow


class DetectionRule(ABC):
    """A single, named, stateless heuristic evaluated against one flow's features."""

    #: Short, stable identifier used in DetectionResult.rule_names and logs.
    name: str

    #: This rule's contribution (0-100 scale) to the aggregate risk score
    #: when it fires. An operational weighting choice, not a statistically
    #: derived value -- see docs/methodology.md.
    base_score: int

    @abstractmethod
    def evaluate(self, features: dict[str, float], flow: FinalizedFlow) -> RuleTrigger | None:
        """Return a RuleTrigger if this rule's condition is met on this flow, else None."""
        raise NotImplementedError
