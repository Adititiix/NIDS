"""
Model and metadata persistence for Phase 6/7.

Saves everything Phase 7 needs to run real-time inference correctly: the
trained model, the fitted scaler, and a metadata JSON file recording the
model type, the EXACT ML-trainable feature schema and ordering used at
training time, which live features were excluded from training and why,
dataset provenance, scaler information, the label mapping, training
configuration, and the evaluation metrics of the saved model. This module
only writes/reads artifacts -- it does not connect them to live capture.

LIVE vs. ML-TRAINABLE feature schema: this module distinguishes the two
explicitly (see app.ml.ml_feature_schema for the full rationale). A model
is trained on -- and, at inference time, expects -- exactly the 38-feature
ML-trainable subset, NOT the full 41-feature live schema, because the
real CIC-IDS2017 MachineLearningCSV release has no Protocol column and
protocol_is_tcp/udp/icmp are therefore excluded from ML training rather
than fabricated. The live schema version is still recorded (as
`live_feature_schema_version`) purely for traceability -- it is NOT what
`load_model_artifact()` validates a live FeatureVector against for
inference dimensionality; `ml_feature_schema_version` and
`ml_feature_names` are.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import StandardScaler

from app.features.schema import FEATURE_SCHEMA_VERSION
from app.ml.ml_feature_schema import EXCLUDED_ML_FEATURES, ML_FEATURE_COUNT, ML_FEATURE_NAMES, ML_FEATURE_SCHEMA_VERSION
from app.ml.preprocessing import BINARY_LABEL_MAP
from app.ml.training import ModelEvalResult

logger = logging.getLogger("nids.ml.persistence")


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """
    Everything needed to verify a saved model artifact is safe to load for
    inference, and to reproduce its exact input feature transformation.

    Distinguishes the LIVE 41-feature schema (used by real-time detection,
    recorded here only for traceability) from the ML-trainable 38-feature
    subset (used by this specific saved model, and what inference actually
    validates against) -- see app.ml.ml_feature_schema.
    """

    model_type: str
    live_feature_schema_version: str
    ml_feature_schema_version: str
    ml_feature_names: tuple[str, ...]
    ml_feature_count: int
    excluded_features: dict[str, str]
    dataset_info: dict[str, Any]
    scaler_info: dict[str, Any]
    label_mapping: dict[str, int]
    training_config: dict[str, Any]
    evaluation: dict[str, Any]
    trained_at: str  # ISO 8601 UTC timestamp


class SchemaVersionMismatchError(ValueError):
    """Raised when loading a model artifact whose ML feature schema doesn't match the current ML-trainable feature subset."""


def save_model_artifact(
    model: object,
    scaler: StandardScaler,
    model_name: str,
    evaluation: ModelEvalResult,
    training_config: dict[str, Any],
    dataset_info: dict[str, Any],
    output_dir: str,
) -> tuple[Path, Path, Path]:
    """
    Save the model, its scaler, and a metadata JSON file into `output_dir`.
    Returns (model_path, scaler_path, metadata_path).

    `dataset_info` should capture provenance for the dataset actually used
    to train this model (e.g. path, row counts, compatibility/cleaning
    summaries) -- kept separate from `training_config` (hyperparameters,
    split sizes, random seed) so the two concerns don't get conflated.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / f"{model_name}.joblib"
    scaler_path = out / f"{model_name}.scaler.joblib"
    metadata_path = out / f"{model_name}.metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    scaler_info = {
        "type": type(scaler).__name__,
        "n_features_in": int(getattr(scaler, "n_features_in_", ML_FEATURE_COUNT)),
    }

    metadata = ModelArtifactMetadata(
        model_type=model_name,
        live_feature_schema_version=FEATURE_SCHEMA_VERSION,
        ml_feature_schema_version=ML_FEATURE_SCHEMA_VERSION,
        ml_feature_names=ML_FEATURE_NAMES,
        ml_feature_count=ML_FEATURE_COUNT,
        excluded_features=dict(EXCLUDED_ML_FEATURES),
        dataset_info=dataset_info,
        scaler_info=scaler_info,
        label_mapping=dict(BINARY_LABEL_MAP),
        training_config=training_config,
        evaluation=evaluation.to_dict(),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )
    with open(metadata_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)

    logger.info("Saved model artifact: %s, %s, %s", model_path, scaler_path, metadata_path)
    return model_path, scaler_path, metadata_path


def load_model_artifact(model_path: str, scaler_path: str, metadata_path: str) -> tuple[object, StandardScaler, dict[str, Any]]:
    """
    Load a saved model + scaler + metadata, and verify the metadata's ML
    feature schema version AND exact ML feature name/order match the
    CURRENT ML-trainable feature subset before handing it back. Raises
    SchemaVersionMismatchError rather than silently returning a model that
    would misinterpret a live feature vector -- this check is what makes
    it safe for Phase 7 to load artifacts saved by an earlier ML feature
    schema version without corrupting inference.

    Deliberately validates against `ml_feature_schema_version`/
    `ml_feature_names`, NOT `live_feature_schema_version` -- the live
    schema can gain unrelated features over time without invalidating an
    already-trained model, as long as the ML-trainable subset it was
    actually trained on hasn't changed.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    if metadata["ml_feature_schema_version"] != ML_FEATURE_SCHEMA_VERSION:
        raise SchemaVersionMismatchError(
            f"Model artifact was trained against ML feature schema version "
            f"'{metadata['ml_feature_schema_version']}', but the current ML-trainable feature schema is "
            f"'{ML_FEATURE_SCHEMA_VERSION}'. Refusing to load -- retrain against the current schema."
        )

    if tuple(metadata["ml_feature_names"]) != ML_FEATURE_NAMES:
        raise SchemaVersionMismatchError(
            "Model artifact's ML feature name/order does not match the current ML-trainable feature "
            "subset, despite a matching version string. Refusing to load."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, metadata