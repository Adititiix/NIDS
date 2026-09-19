"""
Cleaning, label normalization, and train/validation/test splitting for
Phase 6.

Every step that removes rows reports exactly how many and why -- this
project does not silently drop data. Every step that fits a transform
(the scaler) fits ONLY on the training split, applied afterward to
validation/test, to avoid leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("nids.ml.preprocessing")


@dataclass(frozen=True)
class CleaningReport:
    """Row counts at each cleaning stage -- printed/saved, never silently discarded."""

    initial_rows: int
    duplicate_rows_removed: int
    infinite_value_rows_removed: int
    missing_value_rows_removed: int
    final_rows: int

    def summary(self) -> str:
        return (
            f"Cleaning: {self.initial_rows} initial rows -> "
            f"-{self.duplicate_rows_removed} duplicates -> "
            f"-{self.infinite_value_rows_removed} with infinite feature values -> "
            f"-{self.missing_value_rows_removed} with missing feature values -> "
            f"{self.final_rows} final rows "
            f"({self.final_rows / self.initial_rows:.1%} retained)."
            if self.initial_rows > 0
            else "Cleaning: input was empty."
        )


def clean_aligned_dataset(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series, CleaningReport]:
    """
    Remove duplicate rows, rows with infinite feature values, and rows with
    missing (NaN) feature values, from an already-aligned feature DataFrame
    plus its parallel label Series. Reports exact counts removed at each
    stage rather than silently dropping data.

    Operates on whatever columns `features` actually has -- it doesn't
    hardcode the live 41-feature schema, so it works correctly whether
    given the full live feature set or the 38-feature ML-trainable subset
    (app.ml.ml_feature_schema) that align_features() now produces.
    """
    initial_rows = len(features)
    combined = features.copy()
    combined["__label__"] = labels.values

    before_dedup = len(combined)
    combined = combined.drop_duplicates()
    duplicate_rows_removed = before_dedup - len(combined)

    feature_cols = list(features.columns)
    inf_mask = np.isinf(combined[feature_cols].to_numpy(dtype=float)).any(axis=1)
    infinite_value_rows_removed = int(inf_mask.sum())
    combined = combined.loc[~inf_mask]

    nan_mask = combined[feature_cols].isna().any(axis=1)
    missing_value_rows_removed = int(nan_mask.sum())
    combined = combined.loc[~nan_mask]

    final_rows = len(combined)
    report = CleaningReport(
        initial_rows=initial_rows,
        duplicate_rows_removed=duplicate_rows_removed,
        infinite_value_rows_removed=infinite_value_rows_removed,
        missing_value_rows_removed=missing_value_rows_removed,
        final_rows=final_rows,
    )
    logger.info(report.summary())

    cleaned_features = combined[feature_cols].reset_index(drop=True)
    cleaned_labels = combined["__label__"].reset_index(drop=True)
    return cleaned_features, cleaned_labels, report


#: Binary label mapping. Anything not exactly "BENIGN" (case-insensitive,
#: whitespace-stripped) is treated as ATTACK. This intentionally collapses
#: CIC-IDS2017's many specific attack labels (DoS Hulk, PortScan, etc.) into
#: one ATTACK class for Phase 6's binary classifier -- see docs/methodology.md
#: for why multiclass is deferred rather than attempted here.
BINARY_LABEL_MAP = {"BENIGN": 0, "ATTACK": 1}


def normalize_labels_binary(raw_labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Convert raw dataset label strings into binary 0 (BENIGN) / 1 (ATTACK)
    labels. Returns (binary_labels, original_label_strings) -- the original
    strings are preserved (not discarded) in case a future phase wants
    multiclass labels without needing to re-load the raw dataset.
    """
    original = raw_labels.astype(str).str.strip()
    normalized_upper = original.str.upper()
    binary = (normalized_upper != "BENIGN").astype(int)
    return binary.reset_index(drop=True), original.reset_index(drop=True)


@dataclass(frozen=True)
class DatasetSplit:
    """Train/validation/test split, plus the fitted scaler (fit on train only)."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def split_and_scale(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplit:
    """
    Stratified train/validation/test split (stratified on the binary label
    to preserve class balance across splits given known class imbalance in
    intrusion datasets), followed by StandardScaler fit ONLY on the
    training split and applied to all three -- this is the leakage-avoidance
    requirement: validation/test data is never used to fit the scaler.

    KNOWN LIMITATION (documented, not hidden): this is a random row-level
    stratified split, not the scenario/time-aware split envisioned in the
    original project methodology to prevent the same attack session from
    appearing in both train and test. CICFlowMeter's standard CSV export
    doesn't carry an explicit session/scenario identifier to split on
    without additional dataset-specific engineering, so this is deferred --
    see docs/methodology.md.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=random_state, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit ONLY on training data
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info(
        "Split: train=%d val=%d test=%d (stratified, random_state=%d)",
        len(X_train), len(X_val), len(X_test), random_state,
    )

    return DatasetSplit(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train.to_numpy(),
        y_val=y_val.to_numpy(),
        y_test=y_test.to_numpy(),
        scaler=scaler,
    )