"""
CLI entry point for Phase 6 offline model training.

Usage:
    python -m app.ml.train_cli --dataset-path /path/to/cic_ids2017.csv --output-dir ml/models

See run_training_pipeline() in app.ml.pipeline for the actual pipeline;
this module is just argument parsing and human-readable console output.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.ml.feature_mapping import FeatureAlignmentError
from app.ml.pipeline import run_training_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger("nids.ml.train_cli")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline NIDS intrusion-detection models from a CICFlowMeter-style CSV dataset."
    )
    parser.add_argument("--dataset-path", required=True, help="Path to the CIC-IDS2017 (or compatible) CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the trained model, scaler, and metadata into.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional: randomly subsample this many rows before training (for quick pipeline testing).",
    )
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducible splits and model training.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction of data held out for validation.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction of data held out for final testing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        result = run_training_pipeline(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            random_seed=args.random_seed,
            val_size=args.val_size,
            test_size=args.test_size,
        )
    except FeatureAlignmentError as exc:
        logger.error("Dataset incompatible with the ML-trainable feature subset -- aborting before training.\n%s", exc)
        return 1

    print("\n=== Feature compatibility ===")
    print(result.compatibility_summary)
    print("\n=== Data cleaning ===")
    print(result.cleaning_summary)
    print("\n=== Validation results (all baseline models) ===")
    for eval_result in result.val_results.values():
        print(eval_result.summary())
    print(f"\n=== Selected model: {result.best_model_name} (highest macro F1 on validation) ===")
    print("\n=== Final test-set evaluation (selected model, evaluated once) ===")
    for eval_result in result.test_results.values():
        print(eval_result.summary())
    print("\n=== Saved artifacts ===")
    print(f"Model:    {result.model_path}")
    print(f"Scaler:   {result.scaler_path}")
    print(f"Metadata: {result.metadata_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())