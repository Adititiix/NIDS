"""
Phase 6 end-to-end offline training pipeline orchestration.

    Dataset CSV
      -> load_dataset_csv
      -> normalize_columns + build_compatibility_report   (STOP here if not ML-compatible)
      -> align_features                                    (dataset -> 38 ML-trainable features, exact schema order)
      -> normalize_labels_binary
      -> clean_aligned_dataset                              (duplicates / inf / NaN, all counted)
      -> split_and_scale                                    (stratified, scaler fit on train only)
      -> train_and_evaluate for each baseline model
      -> select_best_model
      -> save_model_artifact

This module contains NO live-inference code and is never imported by the
capture/flow/feature/detection pipeline -- it's purely for offline use via
the CLI in app.ml.train_cli.

Trains on the 38-feature ML-trainable subset (app.ml.ml_feature_schema),
not the full 41-feature live schema -- see that module for why
protocol_is_tcp/udp/icmp are excluded (the real CIC-IDS2017
MachineLearningCSV release has no Protocol column).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.ml.dataset import load_dataset_csv
from app.ml.feature_mapping import FeatureAlignmentError, align_features, build_compatibility_report, normalize_columns
from app.ml.persistence import save_model_artifact
from app.ml.preprocessing import clean_aligned_dataset, normalize_labels_binary, split_and_scale
from app.ml.training import ModelEvalResult, build_baseline_models, evaluate_fitted_model, select_best_model, train_and_evaluate

logger = logging.getLogger("nids.ml.pipeline")


@dataclass(frozen=True)
class TrainingRunResult:
    """Everything a caller (CLI or test) needs to inspect the outcome of a full training run."""

    compatibility_summary: str
    cleaning_summary: str
    val_results: dict[str, ModelEvalResult]
    test_results: dict[str, ModelEvalResult]
    best_model_name: str
    model_path: str
    scaler_path: str
    metadata_path: str


def run_training_pipeline(
    dataset_path: str,
    output_dir: str,
    sample_size: int | None = None,
    random_seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> TrainingRunResult:
    """
    Run the full Phase 6 pipeline against a CICFlowMeter-style CSV at
    `dataset_path`, saving the best-performing model (by macro F1 on the
    validation split, per `select_best_model`) into `output_dir`.

    Raises FeatureAlignmentError immediately, before any training happens,
    if the dataset cannot supply every ML-trainable feature (see
    app.ml.ml_feature_schema) or a recognizable label column -- this is
    Phase 6's "stop training if required features cannot be obtained
    reliably" requirement, enforced in code. A dataset missing ONLY
    protocol information (like the real CIC-IDS2017 MachineLearningCSV
    release) is NOT rejected -- protocol features are already excluded
    from ML training, so their absence doesn't block anything.
    """
    raw_df = load_dataset_csv(dataset_path, sample_size=sample_size, random_state=random_seed)

    normalized_df = normalize_columns(raw_df)
    report = build_compatibility_report(normalized_df)
    logger.info("\n%s", report.summary())
    if not report.is_ml_compatible:
        raise FeatureAlignmentError(f"Dataset is not compatible with the ML-trainable feature subset:\n{report.summary()}")

    features, raw_labels, _ = align_features(raw_df)
    binary_labels, _original_labels = normalize_labels_binary(raw_labels)

    features, binary_labels, cleaning_report = clean_aligned_dataset(features, binary_labels)
    logger.info(cleaning_report.summary())

    split = split_and_scale(features, binary_labels, val_size=val_size, test_size=test_size, random_state=random_seed)

    models = build_baseline_models(random_seed=random_seed)
    val_results: dict[str, ModelEvalResult] = {}
    for name, model in models.items():
        val_results[name] = train_and_evaluate(model, name, split.X_train, split.y_train, split.X_val, split.y_val)
        logger.info(val_results[name].summary())

    best_name = select_best_model(val_results)
    best_model = models[best_name]

    # Final, reported test-set evaluation of the selected model only --
    # val_results already fit/evaluated it once; re-evaluating the SAME
    # already-fitted model on the held-out test split (no refitting) gives
    # an unbiased estimate that never touched test data during selection.
    test_result = evaluate_fitted_model(best_model, best_name, split.X_test, split.y_test)
    test_results = {best_name: test_result}

    training_config: dict[str, Any] = {
        "dataset_path": dataset_path,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "val_size": val_size,
        "test_size": test_size,
        "model_hyperparameters": {k: str(v) for k, v in best_model.get_params().items()},
    }

    dataset_info: dict[str, Any] = {
        "dataset_path": dataset_path,
        "raw_row_count": len(raw_df),
        "sample_size": sample_size,
        "compatibility_summary": report.summary(),
        "cleaning_summary": cleaning_report.summary(),
    }

    model_path, scaler_path, metadata_path = save_model_artifact(
        model=best_model,
        scaler=split.scaler,
        model_name=best_name,
        evaluation=test_result,
        training_config=training_config,
        dataset_info=dataset_info,
        output_dir=output_dir,
    )

    return TrainingRunResult(
        compatibility_summary=report.summary(),
        cleaning_summary=cleaning_report.summary(),
        val_results=val_results,
        test_results=test_results,
        best_model_name=best_name,
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        metadata_path=str(metadata_path),
    )