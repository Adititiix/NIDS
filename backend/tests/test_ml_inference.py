"""
Unit tests for Phase 7 (live ML inference).

Uses small, REAL trained model artifacts (a genuine LogisticRegression fit
on a tiny synthetic array via the existing app.ml.training functions, saved
via the existing app.ml.persistence.save_model_artifact) so that MLPredictor
is exercised against actual joblib/JSON files, not mocks -- while staying
fast and self-contained. The synthetic data here is deliberately trivial
and separable purely to exercise the inference PLUMBING; per Phase 6/7
requirements, this is NOT evidence of real detection accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from app.capture.models import PacketMetadata, TransportProtocol
from app.features.feature_extractor import FeatureExtractor
from app.features.schema import FEATURE_COUNT, FEATURE_SCHEMA_VERSION
from app.flows.models import Flow, FinalizedFlow
from app.ml.inference import (
    InvalidFeatureVectorError,
    MLInferenceResult,
    MLPredictor,
    ModelArtifactCorruptedError,
    ModelArtifactNotFoundError,
    SchemaVersionMismatchError,
)
from app.ml.ml_feature_schema import ML_FEATURE_COUNT, ML_FEATURE_NAMES, ML_FEATURE_SCHEMA_VERSION
from app.ml.persistence import save_model_artifact
from app.ml.training import build_baseline_models, evaluate_fitted_model


# ---------------------------------------------------------------------------
# Test fixture: a real, tiny trained model artifact
# ---------------------------------------------------------------------------


def _train_and_save_tiny_model(tmp_path, model_name: str = "logistic_regression") -> tuple[str, str, str]:
    """
    Train a real (not mocked) model directly on a small synthetic array and
    save it via the existing persistence functions, purely to produce a
    genuine artifact triple for inference tests. Not a claim of real
    detection accuracy -- see module docstring.

    Trains on exactly ML_FEATURE_COUNT (38) features, matching what a real
    CIC-IDS2017-trained model actually expects -- NOT the full 41-feature
    live schema, since protocol_is_tcp/udp/icmp are excluded from ML
    training (see app.ml.ml_feature_schema).
    """
    rng = np.random.default_rng(0)
    X = rng.random((40, ML_FEATURE_COUNT))
    y = np.array([0] * 20 + [1] * 20)
    X[20:, 0] += 10.0  # make the second half trivially separable on one feature

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    models = build_baseline_models(random_seed=42)
    model = models[model_name]
    model.fit(X_scaled, y)
    eval_result = evaluate_fitted_model(model, model_name, X_scaled, y)

    model_path, scaler_path, metadata_path = save_model_artifact(
        model=model,
        scaler=scaler,
        model_name=model_name,
        evaluation=eval_result,
        training_config={"note": "Phase 7 test fixture -- not a real evaluation"},
        dataset_info={"dataset_path": "synthetic-test-fixture", "raw_row_count": 40},
        output_dir=str(tmp_path),
    )
    return str(model_path), str(scaler_path), str(metadata_path)


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


@pytest.fixture
def tiny_model_paths(tmp_path):
    return _train_and_save_tiny_model(tmp_path)


@pytest.fixture
def sample_feature_vector():
    flow = build_flow([make_packet(timestamp=100.0, tcp_flags="S"), make_packet(timestamp=100.1, tcp_flags="A")])
    return FeatureExtractor().extract(flow)


# ---------------------------------------------------------------------------
# Valid inference
# ---------------------------------------------------------------------------


def test_predictor_loads_real_artifact_and_runs_inference(tiny_model_paths, sample_feature_vector) -> None:
    model_path, scaler_path, metadata_path = tiny_model_paths
    predictor = MLPredictor.load(model_path, scaler_path, metadata_path)

    result = predictor.predict(sample_feature_vector)

    assert isinstance(result, MLInferenceResult)
    assert result.predicted_label in (0, 1)
    assert result.predicted_class in ("BENIGN", "ATTACK")
    assert result.model_type == "logistic_regression"
    assert result.schema_version == ML_FEATURE_SCHEMA_VERSION
    assert result.flow_key_repr == sample_feature_vector.flow_key_repr


def test_predictor_probability_is_valid_when_present(tiny_model_paths, sample_feature_vector) -> None:
    model_path, scaler_path, metadata_path = tiny_model_paths
    predictor = MLPredictor.load(model_path, scaler_path, metadata_path)

    result = predictor.predict(sample_feature_vector)

    assert result.probability is None or 0.0 <= result.probability <= 1.0


def test_predictor_is_attack_property_matches_predicted_label(tiny_model_paths, sample_feature_vector) -> None:
    model_path, scaler_path, metadata_path = tiny_model_paths
    predictor = MLPredictor.load(model_path, scaler_path, metadata_path)

    result = predictor.predict(sample_feature_vector)
    assert result.is_attack == (result.predicted_label == 1)


def test_predictor_exposes_model_type_and_schema_version_properties(tiny_model_paths) -> None:
    model_path, scaler_path, metadata_path = tiny_model_paths
    predictor = MLPredictor.load(model_path, scaler_path, metadata_path)

    assert predictor.model_type == "logistic_regression"
    assert predictor.schema_version == ML_FEATURE_SCHEMA_VERSION


def test_predictor_selects_exactly_the_38_ml_features_from_the_41_value_live_vector(tiny_model_paths, sample_feature_vector) -> None:
    """
    The core Phase 7 correction: the live FeatureVector carries all 41
    features (including protocol_is_tcp/udp/icmp), but the model was
    trained on exactly ML_FEATURE_NAMES (38). predict() must select those
    38, by name, in the persisted training order -- not assume the raw
    41-value order lines up with the model's 38-value training order.
    """
    model_path, scaler_path, metadata_path = tiny_model_paths
    predictor = MLPredictor.load(model_path, scaler_path, metadata_path)

    assert len(sample_feature_vector.values) == 41  # the live vector genuinely has all 41
    assert predictor.ml_feature_names == ML_FEATURE_NAMES
    assert len(predictor.ml_feature_names) == 38
    assert "protocol_is_tcp" not in predictor.ml_feature_names

    # Spy on the scaler to capture exactly what predict() actually sends it.
    captured = {}
    original_transform = predictor._scaler.transform

    def _spy_transform(X):
        captured["X"] = X
        return original_transform(X)

    predictor._scaler.transform = _spy_transform

    result = predictor.predict(sample_feature_vector)  # must not raise

    assert captured["X"].shape == (1, 38)
    expected_values = [sample_feature_vector.to_dict()[name] for name in ML_FEATURE_NAMES]
    assert list(captured["X"][0]) == expected_values
    assert isinstance(result, MLInferenceResult)


# ---------------------------------------------------------------------------
# Missing artifacts
# ---------------------------------------------------------------------------


def test_load_raises_not_found_for_missing_model_file(tmp_path) -> None:
    _model_path, scaler_path, metadata_path = _train_and_save_tiny_model(tmp_path)
    with pytest.raises(ModelArtifactNotFoundError):
        MLPredictor.load(str(tmp_path / "does_not_exist.joblib"), scaler_path, metadata_path)


def test_load_raises_not_found_for_missing_metadata_file(tmp_path) -> None:
    model_path, scaler_path, _metadata_path = _train_and_save_tiny_model(tmp_path)
    with pytest.raises(ModelArtifactNotFoundError):
        MLPredictor.load(model_path, scaler_path, str(tmp_path / "missing.metadata.json"))


def test_load_raises_not_found_for_missing_scaler_file(tmp_path) -> None:
    model_path, _scaler_path, metadata_path = _train_and_save_tiny_model(tmp_path)
    with pytest.raises(ModelArtifactNotFoundError):
        MLPredictor.load(model_path, str(tmp_path / "missing.scaler.joblib"), metadata_path)


# ---------------------------------------------------------------------------
# Malformed / corrupted metadata
# ---------------------------------------------------------------------------


def test_load_raises_corrupted_for_malformed_json(tmp_path) -> None:
    model_path, scaler_path, metadata_path = _train_and_save_tiny_model(tmp_path)
    with open(metadata_path, "w") as f:
        f.write("{not valid json::")

    with pytest.raises(ModelArtifactCorruptedError):
        MLPredictor.load(model_path, scaler_path, metadata_path)


def test_load_raises_corrupted_for_metadata_missing_required_key(tmp_path) -> None:
    model_path, scaler_path, metadata_path = _train_and_save_tiny_model(tmp_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    del metadata["label_mapping"]
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    with pytest.raises(ModelArtifactCorruptedError):
        MLPredictor.load(model_path, scaler_path, metadata_path)


# ---------------------------------------------------------------------------
# Schema mismatch (load-time, via the existing persistence check) and
# wrong feature count (predict-time, MLPredictor's own defense-in-depth check)
# ---------------------------------------------------------------------------


def test_load_raises_schema_mismatch_for_old_schema_version(tmp_path) -> None:
    model_path, scaler_path, metadata_path = _train_and_save_tiny_model(tmp_path)
    with open(metadata_path) as f:
        metadata = json.load(f)
    metadata["ml_feature_schema_version"] = "0.0.1-old-ml38"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    with pytest.raises(SchemaVersionMismatchError):
        MLPredictor.load(model_path, scaler_path, metadata_path)


def test_predict_raises_on_feature_vector_schema_version_mismatch(tiny_model_paths, sample_feature_vector) -> None:
    """
    Construct a valid predictor, then feed it a FeatureVector whose
    schema_version has been tampered with -- simulating a FeatureVector
    produced by a different schema than the one this specific model
    metadata claims. FeatureVector is frozen, so we build a fresh one via
    its constructor rather than mutating the fixture.
    """
    model_path, scaler_path, metadata_path = tiny_model_paths
    predictor = MLPredictor.load(model_path, scaler_path, metadata_path)

    from app.features.models import FeatureVector

    tampered = FeatureVector(
        values=sample_feature_vector.values,
        schema_version="9.9.9-different",
        flow_key_repr=sample_feature_vector.flow_key_repr,
    )

    with pytest.raises(InvalidFeatureVectorError):
        predictor.predict(tampered)


def test_predict_raises_when_model_requires_a_feature_not_in_the_vector() -> None:
    """
    Directly construct an MLPredictor with fabricated metadata whose
    ml_feature_names includes a feature name that doesn't exist in a real
    FeatureVector -- exercises MLPredictor's own by-name validation,
    independent of FeatureVector's internal invariant (which only guards
    the CURRENTLY IMPORTED live schema's length, not a specific model's
    named feature expectations).
    """
    fake_metadata = {
        "model_type": "fake",
        "live_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "ml_feature_names": list(ML_FEATURE_NAMES) + ["totally_nonexistent_feature"],
        "label_mapping": {"BENIGN": 0, "ATTACK": 1},
    }

    class _StubModel:
        def predict(self, X):
            return [0]

    predictor = MLPredictor(model=_StubModel(), scaler=None, metadata=fake_metadata)

    flow = build_flow([make_packet(timestamp=100.0)])
    vector = FeatureExtractor().extract(flow)

    with pytest.raises(InvalidFeatureVectorError):
        predictor.predict(vector)


def test_predictor_construction_fails_fast_when_ml_feature_names_missing() -> None:
    """predict() cannot function without knowing which named features to select -- this must fail at construction, not at predict() time."""
    fake_metadata = {
        "model_type": "fake",
        "live_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "label_mapping": {"BENIGN": 0, "ATTACK": 1},
        # "ml_feature_names" deliberately omitted
    }

    class _StubModel:
        def predict(self, X):
            return [0]

    with pytest.raises(ModelArtifactCorruptedError):
        MLPredictor(model=_StubModel(), scaler=None, metadata=fake_metadata)


# ---------------------------------------------------------------------------
# Probability handling edge cases
# ---------------------------------------------------------------------------


def test_probability_is_none_when_model_has_no_predict_proba(sample_feature_vector) -> None:
    class _NoProbaModel:
        def predict(self, X):
            return np.array([1])

    metadata = {
        "model_type": "no_proba_stub",
        "live_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "ml_feature_names": list(ML_FEATURE_NAMES),
        "label_mapping": {"BENIGN": 0, "ATTACK": 1},
    }

    class _IdentityScaler:
        def transform(self, X):
            return X

    predictor = MLPredictor(model=_NoProbaModel(), scaler=_IdentityScaler(), metadata=metadata)
    result = predictor.predict(sample_feature_vector)

    assert result.probability is None
    assert result.predicted_label == 1


def test_probability_is_none_when_classes_do_not_include_attack_label(sample_feature_vector) -> None:
    class _SingleClassModel:
        classes_ = np.array([0])  # degenerate: never saw the ATTACK class during training

        def predict(self, X):
            return np.array([0])

        def predict_proba(self, X):
            return np.array([[1.0]])

    metadata = {
        "model_type": "single_class_stub",
        "live_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "ml_feature_names": list(ML_FEATURE_NAMES),
        "label_mapping": {"BENIGN": 0, "ATTACK": 1},
    }

    class _IdentityScaler:
        def transform(self, X):
            return X

    predictor = MLPredictor(model=_SingleClassModel(), scaler=_IdentityScaler(), metadata=metadata)
    result = predictor.predict(sample_feature_vector)

    assert result.probability is None
    assert result.predicted_label == 0


def test_probability_looks_up_attack_column_by_classes_not_fixed_index(sample_feature_vector) -> None:
    """classes_ = [1, 0] (reversed order) -- probability must still come from the column for label 1."""

    class _ReversedClassesModel:
        classes_ = np.array([1, 0])

        def predict(self, X):
            return np.array([1])

        def predict_proba(self, X):
            return np.array([[0.9, 0.1]])  # column 0 -> class 1 (ATTACK), column 1 -> class 0 (BENIGN)

    metadata = {
        "model_type": "reversed_classes_stub",
        "live_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "ml_feature_schema_version": ML_FEATURE_SCHEMA_VERSION,
        "ml_feature_names": list(ML_FEATURE_NAMES),
        "label_mapping": {"BENIGN": 0, "ATTACK": 1},
    }

    class _IdentityScaler:
        def transform(self, X):
            return X

    predictor = MLPredictor(model=_ReversedClassesModel(), scaler=_IdentityScaler(), metadata=metadata)
    result = predictor.predict(sample_feature_vector)

    assert result.probability == pytest.approx(0.9)  # NOT 0.1 -- proves it's not assuming column order


# ---------------------------------------------------------------------------
# Prediction result structure
# ---------------------------------------------------------------------------


def test_ml_inference_result_is_frozen(tiny_model_paths, sample_feature_vector) -> None:
    model_path, scaler_path, metadata_path = tiny_model_paths
    predictor = MLPredictor.load(model_path, scaler_path, metadata_path)
    result = predictor.predict(sample_feature_vector)

    with pytest.raises(Exception):
        result.predicted_label = 999  # frozen dataclass -- must raise


# ---------------------------------------------------------------------------
# Integration: FinalizedFlow -> FeatureExtractor -> MLPredictor
# ---------------------------------------------------------------------------


def test_finalized_flow_to_feature_extractor_to_ml_inference_integration(tmp_path) -> None:
    """
    Full Phase 3 -> 4 -> 7 pipeline on a synthetic flow, using a real
    (if trivially trained) model artifact. This tests PLUMBING correctness
    (a FinalizedFlow can be turned into a prediction without error, with a
    structurally valid result) -- it is explicitly NOT a claim about real
    detection accuracy, since the "model" here was fit on 40 random points.
    """
    model_path, scaler_path, metadata_path = _train_and_save_tiny_model(tmp_path)
    predictor = MLPredictor.load(model_path, scaler_path, metadata_path)

    packets = [make_packet(timestamp=100.0 + i * 0.01, tcp_flags="S") for i in range(20)]
    flow = build_flow(packets)

    feature_vector = FeatureExtractor().extract(flow)
    result = predictor.predict(feature_vector)

    assert isinstance(result, MLInferenceResult)
    assert result.predicted_label in (0, 1)
    assert result.flow_key_repr == repr(flow.key)