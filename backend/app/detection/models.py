"""
Phase 5 detection result types.

A DetectionResult is the structured output of running the rule engine
against ONE flow. It is deliberately a plain, serializable dataclass (no
behavior beyond a couple of read-only properties) so it can later be
handed to an API response, a database row, or a WebSocket message without
translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.ml.inference import MLInferenceResult


class Severity(str, Enum):
    """
    Operational severity bucket, derived from `risk_score` via the existing
    `RiskScoringConfig` boundaries in app.config (0-29 LOW, 30-59 MEDIUM,
    60-79 HIGH, 80-100 CRITICAL). NONE means no rule triggered at all --
    NOT the same as "confirmed benign", since we only have a fixed set of
    heuristics and no rule firing just means none of them recognized
    anything, not that the traffic was inspected and cleared.
    """

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class RuleTrigger:
    """One rule's contribution to a DetectionResult."""

    rule_name: str
    reason: str
    score: int  # this rule's contribution to the aggregate risk score (0-100 scale)
    evidence: dict[str, float]  # the specific feature values that justified the trigger, for explainability


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """
    The structured output of running all registered rules against one flow.

    `is_suspicious` is simply "at least one rule fired" -- it does not by
    itself distinguish a confirmed attack from a heuristic false positive.
    `risk_score` and `severity` exist to let a consumer (dashboard, API,
    future ML fusion layer) prioritize results, not to assert ground truth.
    """

    flow_key_repr: str
    is_suspicious: bool
    risk_score: int
    severity: Severity
    triggered_rules: tuple[RuleTrigger, ...]

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(trigger.rule_name for trigger in self.triggered_rules)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(trigger.reason for trigger in self.triggered_rules)


@dataclass(frozen=True, slots=True)
class HybridDetectionResult:
    """
    Phase 8: the combined output of running both the rule engine (Phase 5)
    and ML inference (Phase 7) against one flow, fused into a single risk
    score and severity.

    Deliberately keeps `rule_result` and `ml_result` intact and inspectable
    rather than collapsing them into just the final number -- a consumer
    (dashboard, API, analyst) can always see exactly what each detector
    said, not just the blended outcome. `ml_result` is `None` when no ML
    predictor was configured, or when ML inference failed safely (see
    `ml_error` for why) -- in either case the final score falls back to
    rules-only rather than silently treating "no ML signal" the same as
    "ML said benign".
    """

    flow_key_repr: str
    rule_result: DetectionResult
    ml_result: MLInferenceResult | None
    ml_error: str | None
    final_risk_score: int
    final_severity: Severity
    rule_weight: float
    ml_weight: float

    @property
    def is_suspicious(self) -> bool:
        return self.final_severity != Severity.NONE

    @property
    def rule_names(self) -> tuple[str, ...]:
        return self.rule_result.rule_names

    @property
    def ml_available(self) -> bool:
        return self.ml_result is not None
