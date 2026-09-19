"""
Phase 7: live ML inference.

    FinalizedFlow -> FeatureExtractor.extract() -> FeatureVector (41 live features)
                                                        |
                                                        v
                                          MLPredictor.predict(vector)
                                                        |
                                          (selects the 38 ML-trainable
                                           features, by name, in the
                                           persisted training order)
                                                        |
                                                        v
                                              MLInferenceResult

`MLPredictor` deliberately does NOT extract features itself -- it only ever
consumes a `FeatureVector` produced by the existing
`app.features.feature_extractor.FeatureExtractor`, per the project's "do
not duplicate FeatureExtractor" constraint. It also does not implement its
own scaling/preprocessing: it applies exactly the `StandardScaler` that was
fit during Phase 6 training and saved alongside the model, loaded via the
existing `app.ml.persistence.load_model_artifact()` (which already
enforces ML feature-schema-version and feature-name/order compatibility at
load time -- that check is not duplicated here).

LIVE vs. ML-TRAINABLE feature selection: the live `FeatureVector` always
carries all 41 features (app.features.schema). A trained model, however,
expects exactly the 38-feature ML-trainable subset (app.ml.ml_feature_schema)
-- protocol_is_tcp/udp/icmp are excluded from training because the real
CIC-IDS2017 dataset has no Protocol column. `predict()` therefore selects
exactly the persisted `ml_feature_names`, BY NAME (via
`FeatureVector.to_dict()`), in the exact order recorded in the model's
metadata, before scaling -- it never assumes the live vector's raw
41-value order lines up with the model's 38-value training order.

Performance note: sklearn inference on a single row is a fast, synchronous,
CPU-bound computation (microseconds to low milliseconds for these baseline
models) -- there is no I/O to make async. No continuous capture-processing
orchestrator/main-loop exists yet in this project (Phases 2-5 are
reusable components exercised via tests/manual scripts, not a running
service), so "non-blocking integration" for Phase 7 means: `predict()`
does not raise anything except the typed `MLInferenceError` subclasses
below, so a future orchestrator can catch-and-skip a flow's ML result
without the whole pipeline crashing. Genuine async decoupling (e.g. a
thread pool) is deferred until such an orchestrator exists to decouple.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np

from app.features.models import FeatureVector
from app.ml.persistence import SchemaVersionMismatchError, load_model_artifact

logger = logging.getLogger("nids.ml.inference")


# ---------------------------------------------------------------------------
# Errors -- explicit and actionable, per Phase 7 requirement E
# ---------------------------------------------------------------------------


class MLInferenceError(Exception):
    """Base class for all Phase 7 inference errors."""


class ModelArtifactNotFoundError(MLInferenceError):
    """Raised when the model, scaler, or metadata file does not exist at the given path."""


class ModelArtifactCorruptedError(MLInferenceError):
    """Raised when an artifact file exists but cannot be parsed/deserialized (malformed JSON, corrupted joblib, missing expected keys)."""


class InvalidFeatureVectorError(MLInferenceError):
    """Raised when a FeatureVector's schema version or available feature names don't match what the loaded model expects."""


# Re-exported so callers of this module can catch schema mismatches without
# also importing from app.ml.persistence directly.
__all__ = [
    "MLInferenceError",
    "ModelArtifactNotFoundError",
    "ModelArtifactCorruptedError",
    "InvalidFeatureVectorError",
    "SchemaVersionMismatchError",
    "MLInferenceResult",
    "MLPredictor",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MLInferenceResult:
    """
    The structured output of running the loaded model against one flow's
    FeatureVector.

    `probability` is the model's estimated probability of the ATTACK
    (positive) class, when the underlying model supports `predict_proba`
    and reports both classes -- `None` otherwise (see
    `MLPredictor._compute_probability` for exactly when that happens).
    A `None` probability does not mean "not suspicious"; it means
    "confidence unavailable" -- callers should not treat it as a value.

    `schema_version` reports the ML-trainable feature schema version
    (app.ml.ml_feature_schema.ML_FEATURE_SCHEMA_VERSION) that produced this
    prediction -- i.e. which 38-feature contract the model actually
    consumed -- not the live 41-feature schema version.

    This result reflects ONLY the trained model's output; it does not
    encode any real-world validation of that model's accuracy. Per the
    project's Phase 6 caveats, `model_type`/`schema_version` are provided
    so a consumer can trace exactly which artifact produced a given
    prediction, but this class makes no claim about how trustworthy that
    prediction actually is.
    """

    model_type: str
    schema_version: str
    predicted_label: int
    predicted_class: str
    probability: float | None
    flow_key_repr: str

    @property
    def is_attack(self) -> bool:
        return self.predicted_label == 1


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class MLPredictor:
    """
    Wraps an already-loaded model + scaler + metadata triple and runs
    inference on FeatureVectors. Use `MLPredictor.load(...)` to construct
    one from artifact file paths (the common case); the plain constructor
    is available for tests that want to inject a fake/stub model directly.
    """

    def __init__(self, model: object, scaler: object, metadata: dict) -> None:
        self._model = model
        self._scaler = scaler
        self._metadata = metadata
        try:
            self._label_mapping: dict[str, int] = metadata["label_mapping"]
            # Required at construction time (not just at predict()) because
            # predict() cannot function at all without knowing which named
            # features to select from the live vector -- failing fast here
            # surfaces a corrupted/incompatible artifact at load time.
            self._ml_feature_names: tuple[str, ...] = tuple(metadata["ml_feature_names"])
        except KeyError as exc:
            raise ModelArtifactCorruptedError(f"Metadata is missing required key: {exc}") from exc
        self._inverse_label_mapping: dict[int, str] = {v: k for k, v in self._label_mapping.items()}

    @classmethod
    def load(cls, model_path: str, scaler_path: str, metadata_path: str) -> MLPredictor:
        """
        Load a Phase 6 model artifact triple. Raises:

        - ModelArtifactNotFoundError if any of the three files is missing.
        - ModelArtifactCorruptedError if metadata JSON is malformed, or a
          required key is missing, or the joblib files can't be deserialized.
        - SchemaVersionMismatchError (from app.ml.persistence, re-exported
          here) if the saved ML feature schema version or exact ML feature
          name/order doesn't match the CURRENT ML-trainable feature subset
          -- this check is performed by load_model_artifact() itself and is
          not duplicated.
        """
        try:
            model, scaler, metadata = load_model_artifact(model_path, scaler_path, metadata_path)
        except SchemaVersionMismatchError:
            raise  # already a clear, actionable, typed error -- propagate as-is
        except FileNotFoundError as exc:
            raise ModelArtifactNotFoundError(
                f"Model artifact file not found: {exc}. Expected model={model_path}, "
                f"scaler={scaler_path}, metadata={metadata_path}. Have you run `python -m app.ml.train_cli`?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ModelArtifactCorruptedError(f"Metadata file at {metadata_path} is not valid JSON: {exc}") from exc
        except KeyError as exc:
            raise ModelArtifactCorruptedError(f"Metadata file at {metadata_path} is missing required key: {exc}") from exc
        except Exception as exc:  # joblib corruption, unpickling errors, etc.
            raise ModelArtifactCorruptedError(
                f"Failed to load model artifact (model={model_path}, scaler={scaler_path}): {exc}"
            ) from exc

        return cls(model, scaler, metadata)

    @property
    def model_type(self) -> str:
        return self._metadata["model_type"]

    @property
    def schema_version(self) -> str:
        """The ML-trainable feature schema version (see MLInferenceResult docstring) -- NOT the live 41-feature schema version."""
        return self._metadata["ml_feature_schema_version"]

    @property
    def ml_feature_names(self) -> tuple[str, ...]:
        """The exact, ordered feature names this model was trained on and expects at inference time."""
        return self._ml_feature_names

    def predict(self, feature_vector: FeatureVector) -> MLInferenceResult:
        """
        Run inference on a single flow's FeatureVector.

        Selects exactly this model's `ml_feature_names` (the 38-feature
        ML-trainable subset -- see app.ml.ml_feature_schema), BY NAME, from
        the live vector's full 41-feature set, in the exact order recorded
        at training time, before scaling. This is what makes training and
        inference dimensionally and order-consistent even though the live
        vector carries more features (protocol_is_tcp/udp/icmp) than the
        model was ever trained on.

        Raises InvalidFeatureVectorError if the vector's schema version
        doesn't match what this model's ML feature subset was derived from,
        or if the vector is missing any feature this model requires.
        """
        self._validate_feature_vector(feature_vector)

        feature_dict = feature_vector.to_dict()
        ordered_values = [feature_dict[name] for name in self._ml_feature_names]

        X = np.asarray([ordered_values], dtype=float)
        X_scaled = self._scaler.transform(X)

        predicted_label = int(self._model.predict(X_scaled)[0])
        probability = self._compute_probability(X_scaled)
        predicted_class = self._inverse_label_mapping.get(predicted_label, str(predicted_label))

        return MLInferenceResult(
            model_type=self.model_type,
            schema_version=self.schema_version,
            predicted_label=predicted_label,
            predicted_class=predicted_class,
            probability=probability,
            flow_key_repr=feature_vector.flow_key_repr,
        )

    def _validate_feature_vector(self, feature_vector: FeatureVector) -> None:
        expected_live_version = self._metadata.get("live_feature_schema_version")
        if expected_live_version is not None and feature_vector.schema_version != expected_live_version:
            raise InvalidFeatureVectorError(
                f"FeatureVector schema version '{feature_vector.schema_version}' does not match "
                f"the live feature schema version this model's ML feature subset was derived from "
                f"('{expected_live_version}')."
            )

        feature_dict = feature_vector.to_dict()
        missing = [name for name in self._ml_feature_names if name not in feature_dict]
        if missing:
            raise InvalidFeatureVectorError(
                f"FeatureVector is missing feature(s) {missing} that this model requires. "
                f"Expected exactly this model's ml_feature_names: {list(self._ml_feature_names)}."
            )

    def _compute_probability(self, X_scaled: np.ndarray) -> float | None:
        """
        Return the model's estimated probability of the ATTACK (label 1)
        class, or None if unavailable. Handles two "unsupported" cases
        explicitly rather than letting them raise:

        1. The model has no `predict_proba` at all (not all sklearn
           classifiers support it).
        2. The model DOES support `predict_proba` but its `classes_` never
           includes label 1 -- possible if it was (mis)trained on
           single-class data. Rather than assume a fixed column index
           (predict_proba's column order follows `model.classes_`, which
           is not guaranteed to be [0, 1]), the ATTACK column is looked up
           explicitly; if it's absent, we log and return None instead of
           guessing.
        """
        if not hasattr(self._model, "predict_proba"):
            return None

        try:
            proba_row = self._model.predict_proba(X_scaled)[0]
            classes = list(self._model.classes_)
            attack_label = self._label_mapping.get("ATTACK", 1)
            if attack_label not in classes:
                logger.warning(
                    "Model %s supports predict_proba but its classes_ (%s) do not include the "
                    "ATTACK label (%s) -- returning probability=None rather than guessing.",
                    self.model_type, classes, attack_label,
                )
                return None
            return float(proba_row[classes.index(attack_label)])
        except Exception:
            logger.exception("predict_proba failed unexpectedly for model %s -- returning probability=None.", self.model_type)
            return None