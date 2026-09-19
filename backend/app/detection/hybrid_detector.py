"""
Phase 8: hybrid rule + ML detection.

    FinalizedFlow -> FeatureExtractor.extract() -> FeatureVector
                                                        |
                              +-------------------------+-------------------------+
                              |                                                   |
                              v                                                   v
                   RuleDetector.detect(vector, flow)                 MLPredictor.predict(vector)
                              |                                                   |
                              v                                                   v
                        DetectionResult                                  MLInferenceResult (or None)
                              |                                                   |
                              +-------------------------+-------------------------+
                                                        |
                                                        v
                                          HybridDetector._fuse(...)
                                                        |
                                                        v
                                            HybridDetectionResult

This module does NOT reimplement rule evaluation or ML inference -- it
orchestrates the existing `RuleDetector` (Phase 5) and `MLPredictor`
(Phase 7) unchanged, and only adds the fusion/scoring logic on top.
`RuleDetector` remains fully usable and unmodified on its own; this is an
additional consumer of it, not a replacement.
"""

from __future__ import annotations

import logging

from app.config import HybridScoringConfig, RiskScoringConfig, settings as default_settings
from app.detection.models import HybridDetectionResult
from app.detection.rule_engine import RuleDetector
from app.detection.severity import classify_severity
from app.features.models import FeatureVector
from app.flows.models import FinalizedFlow
from app.ml.inference import MLInferenceError, MLInferenceResult, MLPredictor

logger = logging.getLogger("nids.detection.hybrid")


class HybridDetector:
    """
    Runs the existing RuleDetector and (optionally) an MLPredictor against
    one flow and fuses their outputs into a single HybridDetectionResult.

    ML is optional by design: `ml_predictor=None` (the default) means
    "rules-only mode" -- useful when no trained model has been deployed
    yet (see Phase 6's open item: no model has been trained on real
    CIC-IDS2017 data). This lets a caller construct a HybridDetector today
    without an ML artifact and add one later without changing any calling
    code.

    Fusion formula (see `HybridScoringConfig` in app.config for the full
    rationale): when both a rule score and an ML score are available,

        final_risk_score = round(rule_weight * rule_score + ml_weight * ml_score)

    where `ml_score` is the ATTACK-class probability scaled to 0-100 (or
    `ml_no_probability_fallback_score` if the model can't report a
    probability, or 0.0 if the model predicts BENIGN). When ML has no
    signal at all -- no predictor configured, or inference failed -- the
    result falls back to the rule score alone rather than treating a
    missing ML opinion the same as ML saying "benign" (which would
    incorrectly drag down a rules-flagged flow's score by the configured
    ml_weight).
    """

    def __init__(
        self,
        rule_detector: RuleDetector | None = None,
        ml_predictor: MLPredictor | None = None,
        hybrid_scoring: HybridScoringConfig | None = None,
        risk_scoring: RiskScoringConfig | None = None,
    ) -> None:
        self._rule_detector = rule_detector or RuleDetector()
        self._ml_predictor = ml_predictor
        self._hybrid_scoring = hybrid_scoring or default_settings.hybrid_scoring
        self._risk_scoring = risk_scoring or default_settings.risk_scoring

        weight_sum = self._hybrid_scoring.rule_weight + self._hybrid_scoring.ml_weight
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"HybridScoringConfig.rule_weight ({self._hybrid_scoring.rule_weight}) + ml_weight "
                f"({self._hybrid_scoring.ml_weight}) must sum to 1.0, got {weight_sum}."
            )

    @property
    def ml_enabled(self) -> bool:
        return self._ml_predictor is not None

    def detect(self, feature_vector: FeatureVector, flow: FinalizedFlow) -> HybridDetectionResult:
        """Run rules and (if configured) ML against `flow`'s features, and return the fused result."""
        rule_result = self._rule_detector.detect(feature_vector, flow)
        ml_result, ml_error = self._run_ml_safely(feature_vector, flow)

        final_score = self._fuse_scores(rule_result.risk_score, ml_result)
        final_severity = classify_severity(final_score, self._risk_scoring)

        return HybridDetectionResult(
            flow_key_repr=repr(flow.key),
            rule_result=rule_result,
            ml_result=ml_result,
            ml_error=ml_error,
            final_risk_score=final_score,
            final_severity=final_severity,
            rule_weight=self._hybrid_scoring.rule_weight,
            ml_weight=self._hybrid_scoring.ml_weight,
        )

    def _run_ml_safely(self, feature_vector: FeatureVector, flow: FinalizedFlow) -> tuple[MLInferenceResult | None, str | None]:
        """
        Run ML inference if a predictor is configured, catching any
        MLInferenceError so a model/schema problem degrades to rules-only
        for this flow rather than crashing the whole detection pipeline.
        """
        if self._ml_predictor is None:
            return None, "No ML predictor configured -- running in rules-only mode."

        try:
            return self._ml_predictor.predict(feature_vector), None
        except MLInferenceError as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "ML inference failed for flow %s -- falling back to rules-only scoring. %s",
                repr(flow.key), error_message,
            )
            return None, error_message

    def _fuse_scores(self, rule_score: int, ml_result: MLInferenceResult | None) -> int:
        if ml_result is None:
            # No ML signal at all -- fall back to rules-only rather than
            # implicitly scoring ML's absence as "benign" (which would
            # multiply a valid rule score by rule_weight < 1.0 for no reason).
            return max(0, min(100, rule_score))

        ml_score = self._ml_score(ml_result)
        combined = self._hybrid_scoring.rule_weight * rule_score + self._hybrid_scoring.ml_weight * ml_score
        return max(0, min(100, round(combined)))

    def _ml_score(self, ml_result: MLInferenceResult) -> float:
        """Scale one MLInferenceResult to a 0-100 contribution, per the documented fallback rules."""
        if not ml_result.is_attack:
            return 0.0
        if ml_result.probability is not None:
            return ml_result.probability * 100.0
        return self._hybrid_scoring.ml_no_probability_fallback_score
