"""
Unit tests for Phase 8 (hybrid rule + ML detection).

Covers the four required rule/ML combination cases, the weight-sum
validation, graceful degradation when ML is unavailable or fails, and a
full FinalizedFlow -> FeatureExtractor -> HybridDetector integration test.
Uses stub MLPredictor-shaped objects (matching MLPredictor's public
interface: a `.predict()` method returning MLInferenceResult) rather than
real trained models, since these tests are about FUSION logic, not model
accuracy -- Phase 7's tests already cover real model loading/inference.
"""

from __future__ import annotations

import pytest

from app.capture.models import PacketMetadata, TransportProtocol
from app.config import DetectionThresholds, HybridScoringConfig, RiskScoringConfig
from app.detection.hybrid_detector import HybridDetector
from app.detection.models import Severity
from app.detection.rule_engine import RuleDetector
from app.features.feature_extractor import FeatureExtractor
from app.flows.models import Flow, FinalizedFlow
from app.ml.inference import MLInferenceError, MLInferenceResult


def make_packet(
    *,
    timestamp: float,
    src_ip: str = "10.0.2.15",
    dst_ip: str = "10.0.2.1",
    src_port: int | None = 51234,
    dst_port: int | None = 443,
    protocol: TransportProtocol = TransportProtocol.TCP,
    length: int = 100,
    tcp_flags: str | None = None,
) -> PacketMetadata:
    return PacketMetadata(
        timestamp=timestamp, src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port,
        protocol=protocol, length=length, tcp_flags=tcp_flags,
    )


def build_flow(packets: list[PacketMetadata]) -> FinalizedFlow:
    flow: Flow | None = None
    for i, pkt in enumerate(packets):
        flow = Flow.start(pkt) if i == 0 else flow
        if i > 0:
            assert flow is not None
            flow.add_packet(pkt)
    assert flow is not None
    return flow.finalize()


def normal_flow() -> FinalizedFlow:
    return build_flow(
        [
            make_packet(timestamp=100.0, length=500, tcp_flags="S"),
            make_packet(timestamp=100.1, src_ip="10.0.2.1", src_port=443, dst_ip="10.0.2.15", dst_port=51234, length=500, tcp_flags="SA"),
            make_packet(timestamp=100.2, length=500, tcp_flags="A"),
        ]
    )


def suspicious_flow() -> FinalizedFlow:
    # SYN-heavy burst to a suspicious port -- trips multiple Phase 5 rules.
    packets = [make_packet(timestamp=100.0 + i * 0.005, dst_port=4444, tcp_flags="S") for i in range(30)]
    return build_flow(packets)


class _StubPredictor:
    """A minimal MLPredictor-shaped stub: only implements .predict(), returning a fixed result."""

    def __init__(self, result: MLInferenceResult | None = None, raises: Exception | None = None) -> None:
        self._result = result
        self._raises = raises

    def predict(self, feature_vector) -> MLInferenceResult:
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result


def make_ml_result(*, is_attack: bool, probability: float | None, flow_key_repr: str = "x") -> MLInferenceResult:
    return MLInferenceResult(
        model_type="stub_model",
        schema_version="1.0.0",
        predicted_label=1 if is_attack else 0,
        predicted_class="ATTACK" if is_attack else "BENIGN",
        probability=probability,
        flow_key_repr=flow_key_repr,
    )


def rules_only_detector() -> RuleDetector:
    """A RuleDetector using default (production) thresholds -- normal_flow() should not trip it, suspicious_flow() should."""
    return RuleDetector(thresholds=DetectionThresholds(), risk_scoring=RiskScoringConfig())


def even_weights() -> HybridScoringConfig:
    return HybridScoringConfig(rule_weight=0.5, ml_weight=0.5, ml_no_probability_fallback_score=60.0)


# ---------------------------------------------------------------------------
# The four required combination cases
# ---------------------------------------------------------------------------


def test_case_rules_benign_ml_benign() -> None:
    flow = normal_flow()
    vector = FeatureExtractor().extract(flow)
    ml = _StubPredictor(result=make_ml_result(is_attack=False, probability=0.05))
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)

    assert result.rule_result.is_suspicious is False
    assert result.ml_result is not None and result.ml_result.is_attack is False
    assert result.final_risk_score == 0
    assert result.final_severity == Severity.NONE
    assert result.is_suspicious is False


def test_case_rules_suspicious_ml_benign() -> None:
    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    ml = _StubPredictor(result=make_ml_result(is_attack=False, probability=0.02))
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)

    assert result.rule_result.is_suspicious is True
    assert result.ml_result is not None and result.ml_result.is_attack is False
    # ML pulls the score down from pure-rules, but a strongly rule-flagged
    # flow should still register as at least somewhat suspicious.
    assert result.final_risk_score == round(0.5 * result.rule_result.risk_score + 0.5 * 0.0)
    assert result.final_risk_score < result.rule_result.risk_score  # ML did pull it down
    assert result.final_risk_score > 0


def test_case_rules_benign_ml_suspicious() -> None:
    flow = normal_flow()
    vector = FeatureExtractor().extract(flow)
    ml = _StubPredictor(result=make_ml_result(is_attack=True, probability=0.9))
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)

    assert result.rule_result.is_suspicious is False
    assert result.ml_result is not None and result.ml_result.is_attack is True
    assert result.final_risk_score == round(0.5 * 0 + 0.5 * 90.0)
    assert result.final_risk_score == 45
    assert result.final_severity == Severity.MEDIUM
    assert result.is_suspicious is True  # ML alone can surface a flow rules missed


def test_case_rules_suspicious_ml_suspicious() -> None:
    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    ml = _StubPredictor(result=make_ml_result(is_attack=True, probability=0.95))
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)

    assert result.rule_result.is_suspicious is True
    assert result.ml_result is not None and result.ml_result.is_attack is True
    expected = round(0.5 * result.rule_result.risk_score + 0.5 * 95.0)
    assert result.final_risk_score == min(100, expected)
    assert result.final_severity in (Severity.HIGH, Severity.CRITICAL)


# ---------------------------------------------------------------------------
# Graceful degradation: no predictor configured, or ML raises
# ---------------------------------------------------------------------------


def test_no_ml_predictor_falls_back_to_rules_only() -> None:
    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=None, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    assert detector.ml_enabled is False
    result = detector.detect(vector, flow)

    assert result.ml_result is None
    assert result.ml_error is not None and "rules-only" in result.ml_error
    assert result.final_risk_score == result.rule_result.risk_score  # exact fallback, not scaled by rule_weight


def test_ml_inference_error_falls_back_to_rules_only_without_crashing() -> None:
    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    ml = _StubPredictor(raises=MLInferenceError("simulated schema mismatch"))
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)  # must not raise

    assert result.ml_result is None
    assert result.ml_error is not None and "simulated schema mismatch" in result.ml_error
    assert result.final_risk_score == result.rule_result.risk_score


def test_unexpected_non_ml_exception_is_not_swallowed() -> None:
    """Only MLInferenceError (and subclasses) are caught -- a genuine bug elsewhere must still surface."""
    flow = normal_flow()
    vector = FeatureExtractor().extract(flow)
    ml = _StubPredictor(raises=RuntimeError("not an MLInferenceError"))
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    with pytest.raises(RuntimeError):
        detector.detect(vector, flow)


# ---------------------------------------------------------------------------
# ML probability fallback (no predict_proba / no probability reported)
# ---------------------------------------------------------------------------


def test_ml_attack_without_probability_uses_configured_fallback_score() -> None:
    flow = normal_flow()
    vector = FeatureExtractor().extract(flow)
    ml = _StubPredictor(result=make_ml_result(is_attack=True, probability=None))
    scoring = HybridScoringConfig(rule_weight=0.5, ml_weight=0.5, ml_no_probability_fallback_score=60.0)
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=scoring, risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)

    assert result.final_risk_score == round(0.5 * 0 + 0.5 * 60.0)
    assert result.final_risk_score == 30


# ---------------------------------------------------------------------------
# Weight configuration validation
# ---------------------------------------------------------------------------


def test_weights_must_sum_to_one() -> None:
    bad_scoring = HybridScoringConfig(rule_weight=0.7, ml_weight=0.7)  # sums to 1.4
    with pytest.raises(ValueError):
        HybridDetector(hybrid_scoring=bad_scoring)


def test_uneven_weights_are_respected() -> None:
    flow = normal_flow()
    vector = FeatureExtractor().extract(flow)
    ml = _StubPredictor(result=make_ml_result(is_attack=True, probability=1.0))
    scoring = HybridScoringConfig(rule_weight=0.2, ml_weight=0.8, ml_no_probability_fallback_score=60.0)
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=scoring, risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)

    assert result.final_risk_score == round(0.2 * 0 + 0.8 * 100.0)
    assert result.final_risk_score == 80


# ---------------------------------------------------------------------------
# Score is always clamped to [0, 100]
# ---------------------------------------------------------------------------


def test_final_score_is_capped_at_100() -> None:
    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    ml = _StubPredictor(result=make_ml_result(is_attack=True, probability=1.0))
    scoring = HybridScoringConfig(rule_weight=0.9, ml_weight=0.1, ml_no_probability_fallback_score=60.0)
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=scoring, risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)

    assert 0 <= result.final_risk_score <= 100


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_hybrid_result_preserves_underlying_rule_and_ml_results() -> None:
    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    ml_result = make_ml_result(is_attack=True, probability=0.75, flow_key_repr=repr(flow.key))
    ml = _StubPredictor(result=ml_result)
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)

    assert result.rule_result.rule_names  # the underlying triggered rules are still visible
    assert result.ml_result is ml_result  # not copied/mutated
    assert result.ml_available is True
    assert result.flow_key_repr == repr(flow.key)
    assert result.rule_weight == 0.5
    assert result.ml_weight == 0.5


def test_rule_detector_is_not_modified_or_replaced_by_hybrid_use() -> None:
    """Using a RuleDetector inside HybridDetector must not change its standalone behavior."""
    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    standalone_detector = rules_only_detector()
    standalone_result = standalone_detector.detect(vector, flow)

    hybrid = HybridDetector(rule_detector=standalone_detector, ml_predictor=None)
    hybrid_result = hybrid.detect(vector, flow)

    assert hybrid_result.rule_result.risk_score == standalone_result.risk_score
    assert hybrid_result.rule_result.rule_names == standalone_result.rule_names


# ---------------------------------------------------------------------------
# Full pipeline integration: FinalizedFlow -> FeatureExtractor -> HybridDetector
# ---------------------------------------------------------------------------


def test_full_pipeline_integration_with_stub_ml() -> None:
    flow = suspicious_flow()
    extractor = FeatureExtractor()
    vector = extractor.extract(flow)

    ml = _StubPredictor(result=make_ml_result(is_attack=True, probability=0.8, flow_key_repr=repr(flow.key)))
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=ml, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)

    assert result.is_suspicious is True
    assert result.rule_result.is_suspicious is True
    assert result.ml_result is not None and result.ml_result.is_attack is True
    assert result.final_severity in (Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)


def test_full_pipeline_integration_rules_only_mode() -> None:
    """No ML predictor at all -- HybridDetector must still function end-to-end."""
    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    detector = HybridDetector(rule_detector=rules_only_detector())  # default ml_predictor=None

    result = detector.detect(vector, flow)

    assert result.ml_result is None
    assert result.final_risk_score == result.rule_result.risk_score


def test_full_pipeline_integration_with_real_ml_predictor(tmp_path) -> None:
    """
    End-to-end with a REAL (if trivially trained) MLPredictor, not a stub --
    verifies Phase 7 and Phase 8 actually wire together correctly, not just
    that HybridDetector's fusion math is right against a fake. Per Phase
    6/7 caveats, the tiny synthetic model here is a plumbing check only,
    not evidence of real detection accuracy.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from app.ml.inference import MLPredictor
    from app.ml.ml_feature_schema import ML_FEATURE_COUNT
    from app.ml.persistence import save_model_artifact
    from app.ml.training import build_baseline_models, evaluate_fitted_model

    rng = np.random.default_rng(1)
    X = rng.random((40, ML_FEATURE_COUNT))
    y = np.array([0] * 20 + [1] * 20)
    X[20:, 0] += 10.0
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    model = build_baseline_models(random_seed=42)["logistic_regression"]
    model.fit(X_scaled, y)
    eval_result = evaluate_fitted_model(model, "logistic_regression", X_scaled, y)
    model_path, scaler_path, metadata_path = save_model_artifact(
        model=model, scaler=scaler, model_name="logistic_regression",
        evaluation=eval_result, training_config={"note": "hybrid integration test fixture"},
        dataset_info={"dataset_path": "synthetic-test-fixture", "raw_row_count": 40},
        output_dir=str(tmp_path),
    )
    real_predictor = MLPredictor.load(str(model_path), str(scaler_path), str(metadata_path))

    flow = suspicious_flow()
    vector = FeatureExtractor().extract(flow)
    detector = HybridDetector(rule_detector=rules_only_detector(), ml_predictor=real_predictor, hybrid_scoring=even_weights(), risk_scoring=RiskScoringConfig())

    result = detector.detect(vector, flow)  # must not raise

    assert result.ml_result is not None
    assert result.ml_error is None
    assert 0 <= result.final_risk_score <= 100
    assert result.rule_result.is_suspicious is True  # suspicious_flow() should still trip Phase 5 rules regardless of ML