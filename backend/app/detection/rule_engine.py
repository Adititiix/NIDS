"""
RuleDetector: evaluates a fixed set of stateless, single-flow rules against
one flow's FeatureVector + FinalizedFlow, and aggregates the results into a
single DetectionResult.

This is Phase 5's entry point. The intended pipeline is:

    FinalizedFlow -> FeatureExtractor.extract() -> FeatureVector
                                                        |
                                                        v
                                          RuleDetector.detect(vector, flow)
                                                        |
                                                        v
                                               DetectionResult

Adding a new rule later means writing a new DetectionRule subclass and
adding it to `_build_default_rules()` -- nothing else in this file changes.
This is also the seam where Phase 7's ML detector will plug in alongside
the rule engine (fusing rule + ML output into a shared risk score), without
needing to change how rules themselves work.
"""

from __future__ import annotations

from app.config import DetectionThresholds, RiskScoringConfig, settings as default_settings
from app.detection.models import DetectionResult, Severity
from app.detection.rules.base import DetectionRule
from app.detection.rules.high_byte_rate import HighByteRateRule
from app.detection.rules.high_packet_rate import HighPacketRateRule
from app.detection.rules.short_high_volume import ShortHighVolumeRule
from app.detection.rules.suspicious_port import SuspiciousPortRule
from app.detection.rules.syn_heavy import SynHeavyRule
from app.detection.rules.tcp_flag_anomaly import TcpFlagAnomalyRule
from app.detection.severity import classify_severity
from app.features.models import FeatureVector
from app.flows.models import FinalizedFlow


def _build_default_rules(thresholds: DetectionThresholds) -> tuple[DetectionRule, ...]:
    """The full set of Phase 5 single-flow rules, in a fixed (but order-independent) sequence."""
    return (
        HighPacketRateRule(thresholds),
        HighByteRateRule(thresholds),
        SynHeavyRule(thresholds),
        TcpFlagAnomalyRule(thresholds),
        ShortHighVolumeRule(thresholds),
        SuspiciousPortRule(thresholds),
    )


class RuleDetector:
    """
    Runs every registered rule against one flow and combines their results.

    Aggregate risk score is the sum of each triggered rule's `base_score`,
    capped at 100. This is a simple, explainable, and deliberately
    non-"scientifically calibrated" scheme -- multiple independent weak
    signals add up to a stronger one, which is a reasonable operational
    choice but not a statistically fitted model. Severity is then derived
    from that score via the existing `RiskScoringConfig` bucket boundaries
    (the same 0-29/30-59/60-79/80-100 scheme documented in app.config and
    docs/methodology.md).
    """

    def __init__(
        self,
        thresholds: DetectionThresholds | None = None,
        risk_scoring: RiskScoringConfig | None = None,
        rules: tuple[DetectionRule, ...] | None = None,
    ) -> None:
        self._thresholds = thresholds or default_settings.thresholds
        self._risk_scoring = risk_scoring or default_settings.risk_scoring
        self._rules = rules if rules is not None else _build_default_rules(self._thresholds)

    @property
    def rules(self) -> tuple[DetectionRule, ...]:
        return self._rules

    def detect(self, feature_vector: FeatureVector, flow: FinalizedFlow) -> DetectionResult:
        """Evaluate every rule against `flow`'s features and return the combined result."""
        features = feature_vector.to_dict()

        triggered = []
        for rule in self._rules:
            trigger = rule.evaluate(features, flow)
            if trigger is not None:
                triggered.append(trigger)

        risk_score = min(100, sum(trigger.score for trigger in triggered))
        severity = self._classify_severity(risk_score)

        return DetectionResult(
            flow_key_repr=repr(flow.key),
            is_suspicious=bool(triggered),
            risk_score=risk_score,
            severity=severity,
            triggered_rules=tuple(triggered),
        )

    def _classify_severity(self, risk_score: int) -> Severity:
        """Delegates to app.detection.severity.classify_severity -- kept as a method for backward compatibility."""
        return classify_severity(risk_score, self._risk_scoring)
