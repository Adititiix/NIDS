"""
Unit tests for Phase 6 (dataset preprocessing + ML training).

All tests use a small, synthetic, CICFlowMeter-column-shaped in-memory
DataFrame/CSV (never the real multi-gigabyte CIC-IDS2017 release), so this
suite stays fast and has no external data dependency. Column names,
leading-whitespace quirks, and unit conventions (microsecond durations)
mirror the real dataset closely enough to exercise the actual mapping
logic in app.ml.feature_mapping.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.features.schema import FEATURE_COUNT, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from app.ml.dataset import load_dataset_csv
from app.ml.feature_mapping import (
    FeatureAlignmentError,
    align_features,
    build_compatibility_report,
    normalize_columns,
)
from app.ml.ml_feature_schema import EXCLUDED_ML_FEATURES, ML_FEATURE_COUNT, ML_FEATURE_NAMES, ML_FEATURE_SCHEMA_VERSION
from app.ml.persistence import SchemaVersionMismatchError, load_model_artifact, save_model_artifact
from app.ml.pipeline import run_training_pipeline
from app.ml.preprocessing import clean_aligned_dataset, normalize_labels_binary, split_and_scale
from app.ml.training import build_baseline_models, evaluate_fitted_model, select_best_model, train_and_evaluate


def make_synthetic_dataset(n_benign: int = 30, n_attack: int = 30, seed: int = 0) -> pd.DataFrame:
    """
    Build a small, fully-compatible, CICFlowMeter-shaped synthetic dataset.
    BENIGN rows get modest values; ATTACK rows get exaggerated packet/byte
    rates and SYN counts, purely so a classifier has SOMETHING learnable --
    this is a structural/pipeline test fixture, not a claim about real
    attack traffic characteristics.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n_benign):
        fwd_packets = rng.integers(2, 10)
        bwd_packets = rng.integers(2, 10)
        rows.append(
            {
                " Destination Port": int(rng.choice([80, 443, 22])),
                "Protocol": 6,
                " Flow Duration": float(rng.integers(500_000, 3_000_000)),  # microseconds
                "Total Fwd Packets": int(fwd_packets),
                " Total Backward Packets": int(bwd_packets),
                "Total Length of Fwd Packets": float(fwd_packets * rng.integers(200, 800)),
                " Total Length of Bwd Packets": float(bwd_packets * rng.integers(200, 800)),
                " Fwd Packet Length Max": float(rng.integers(400, 900)),
                " Fwd Packet Length Min": float(rng.integers(40, 200)),
                " Fwd Packet Length Mean": float(rng.integers(200, 500)),
                " Fwd Packet Length Std": float(rng.integers(10, 100)),
                "Bwd Packet Length Max": float(rng.integers(400, 900)),
                " Bwd Packet Length Min": float(rng.integers(40, 200)),
                " Bwd Packet Length Mean": float(rng.integers(200, 500)),
                " Bwd Packet Length Std": float(rng.integers(10, 100)),
                "Flow Bytes/s": float(rng.integers(1_000, 50_000)),
                " Flow Packets/s": float(rng.integers(2, 20)),
                " Flow IAT Mean": float(rng.integers(50_000, 500_000)),
                " Flow IAT Std": float(rng.integers(1_000, 50_000)),
                " Flow IAT Max": float(rng.integers(500_000, 1_000_000)),
                " Flow IAT Min": float(rng.integers(0, 1_000)),
                "Fwd IAT Mean": float(rng.integers(50_000, 500_000)),
                " Fwd IAT Std": float(rng.integers(1_000, 50_000)),
                " Bwd IAT Mean": float(rng.integers(50_000, 500_000)),
                " Bwd IAT Std": float(rng.integers(1_000, 50_000)),
                "Fwd Packets/s": float(rng.integers(1, 10)),
                " Bwd Packets/s": float(rng.integers(1, 10)),
                " Min Packet Length": float(rng.integers(40, 100)),
                " Max Packet Length": float(rng.integers(500, 900)),
                " Packet Length Mean": float(rng.integers(200, 500)),
                " Packet Length Std": float(rng.integers(10, 100)),
                " SYN Flag Count": 1,
                " ACK Flag Count": int(fwd_packets + bwd_packets - 1),
                " FIN Flag Count": 1,
                " RST Flag Count": 0,
                " PSH Flag Count": int(rng.integers(0, 2)),
                " URG Flag Count": 0,
                " Label": "BENIGN",
            }
        )

    for _ in range(n_attack):
        fwd_packets = rng.integers(80, 200)
        rows.append(
            {
                " Destination Port": int(rng.choice([445, 3389, 8080])),
                "Protocol": 6,
                " Flow Duration": float(rng.integers(10_000, 100_000)),  # short, in microseconds
                "Total Fwd Packets": int(fwd_packets),
                " Total Backward Packets": 0,
                "Total Length of Fwd Packets": float(fwd_packets * 40),
                " Total Length of Bwd Packets": 0.0,
                " Fwd Packet Length Max": 40.0,
                " Fwd Packet Length Min": 40.0,
                " Fwd Packet Length Mean": 40.0,
                " Fwd Packet Length Std": 0.0,
                "Bwd Packet Length Max": 0.0,
                " Bwd Packet Length Min": 0.0,
                " Bwd Packet Length Mean": 0.0,
                " Bwd Packet Length Std": 0.0,
                "Flow Bytes/s": float(rng.integers(500_000, 2_000_000)),
                " Flow Packets/s": float(rng.integers(500, 2000)),
                " Flow IAT Mean": float(rng.integers(100, 1_000)),
                " Flow IAT Std": float(rng.integers(10, 100)),
                " Flow IAT Max": float(rng.integers(1_000, 5_000)),
                " Flow IAT Min": 0.0,
                "Fwd IAT Mean": float(rng.integers(100, 1_000)),
                " Fwd IAT Std": float(rng.integers(10, 100)),
                " Bwd IAT Mean": 0.0,
                " Bwd IAT Std": 0.0,
                "Fwd Packets/s": float(rng.integers(500, 2000)),
                " Bwd Packets/s": 0.0,
                " Min Packet Length": 40.0,
                " Max Packet Length": 40.0,
                " Packet Length Mean": 40.0,
                " Packet Length Std": 0.0,
                " SYN Flag Count": int(fwd_packets),
                " ACK Flag Count": 0,
                " FIN Flag Count": 0,
                " RST Flag Count": 0,
                " PSH Flag Count": 0,
                " URG Flag Count": 0,
                " Label": "PortScan",
            }
        )

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle


def make_real_cic_style_dataset(n_benign: int = 20, n_attack: int = 20, seed: int = 0) -> pd.DataFrame:
    """
    Approximates the ACTUAL real-world CIC-IDS2017 MachineLearningCSV
    release column shape (as found during real-dataset validation): no
    Protocol column at all, plus a realistic number of extra columns this
    project doesn't map to any live feature (Init_Win_bytes_*, Active/Idle
    stats, Subflow columns, header lengths, etc.) -- padding the column
    count up toward the real file's ~79 columns, so this fixture exercises
    the genuine "many extra unmapped columns, no Protocol column" shape
    rather than only the minimal, fully-mapped synthetic dataset above.
    """
    base = make_synthetic_dataset(n_benign=n_benign, n_attack=n_attack, seed=seed)
    df = base.drop(columns=["Protocol"])  # the real files simply do not have this column

    rng = np.random.default_rng(seed + 1)
    n = len(df)
    extra_columns = {
        "Fwd Header Length": rng.integers(20, 40, size=n),
        " Bwd Header Length": rng.integers(20, 40, size=n),
        " Down/Up Ratio": rng.random(size=n),
        " Average Packet Size": rng.integers(40, 900, size=n),
        " Fwd Avg Bytes/Bulk": np.zeros(n, dtype=int),
        " Fwd Avg Packets/Bulk": np.zeros(n, dtype=int),
        " Fwd Avg Bulk Rate": np.zeros(n, dtype=int),
        " Bwd Avg Bytes/Bulk": np.zeros(n, dtype=int),
        " Bwd Avg Packets/Bulk": np.zeros(n, dtype=int),
        "Bwd Avg Bulk Rate": np.zeros(n, dtype=int),
        "Subflow Fwd Packets": rng.integers(1, 10, size=n),
        " Subflow Fwd Bytes": rng.integers(100, 2000, size=n),
        " Subflow Bwd Packets": rng.integers(1, 10, size=n),
        " Subflow Bwd Bytes": rng.integers(100, 2000, size=n),
        "Init_Win_bytes_forward": rng.integers(0, 65535, size=n),
        " Init_Win_bytes_backward": rng.integers(0, 65535, size=n),
        " act_data_pkt_fwd": rng.integers(0, 10, size=n),
        " min_seg_size_forward": rng.integers(20, 40, size=n),
        "Active Mean": rng.integers(0, 100_000, size=n),
        " Active Std": rng.integers(0, 10_000, size=n),
        " Active Max": rng.integers(0, 200_000, size=n),
        " Active Min": rng.integers(0, 1_000, size=n),
        "Idle Mean": rng.integers(0, 1_000_000, size=n),
        " Idle Std": rng.integers(0, 100_000, size=n),
        " Idle Max": rng.integers(0, 2_000_000, size=n),
        " Idle Min": rng.integers(0, 1_000, size=n),
        "Fwd PSH Flags": np.zeros(n, dtype=int),
        " Bwd PSH Flags": np.zeros(n, dtype=int),
        " Fwd URG Flags": np.zeros(n, dtype=int),
        " Bwd URG Flags": np.zeros(n, dtype=int),
        "FWD Init Win Bytes": rng.integers(0, 65535, size=n),
        " CWE Flag Count": np.zeros(n, dtype=int),
        " ECE Flag Count": np.zeros(n, dtype=int),
        " Bwd IAT Total": rng.integers(0, 1_000_000, size=n),
        "Fwd IAT Total": rng.integers(0, 1_000_000, size=n),
        " Fwd IAT Max": rng.integers(0, 1_000_000, size=n),
        " Fwd IAT Min": rng.integers(0, 1_000, size=n),
        " Bwd IAT Max": rng.integers(0, 1_000_000, size=n),
        " Bwd IAT Min": rng.integers(0, 1_000, size=n),
        " Flow ID": [f"flow-{i}" for i in range(n)],
        " Source IP": "10.0.0.1",
        " Source Port": rng.integers(1024, 65535, size=n),
        " Destination IP": "10.0.0.2",
        " Timestamp": "2017-07-03 09:00:00",
    }
    for col, values in extra_columns.items():
        df[col] = values

    return df


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def test_load_dataset_csv_reads_all_rows(tmp_path) -> None:
    df = make_synthetic_dataset(n_benign=5, n_attack=5)
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_dataset_csv(str(csv_path))
    assert len(loaded) == 10


def test_load_dataset_csv_respects_sample_size(tmp_path) -> None:
    df = make_synthetic_dataset(n_benign=20, n_attack=20)
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_dataset_csv(str(csv_path), sample_size=10, random_state=1)
    assert len(loaded) == 10


# ---------------------------------------------------------------------------
# Column normalization + compatibility validation
# ---------------------------------------------------------------------------


def test_normalize_columns_strips_whitespace() -> None:
    df = pd.DataFrame({" Destination Port": [1], "Protocol ": [2]})
    normalized = normalize_columns(df)
    assert list(normalized.columns) == ["Destination Port", "Protocol"]


def test_real_cic_style_dataset_has_no_protocol_column_and_many_extra_columns() -> None:
    """Sanity-check the fixture itself actually reproduces the real-world shape before relying on it below."""
    df = make_real_cic_style_dataset(n_benign=5, n_attack=5)
    assert "Protocol" not in df.columns
    assert len(df.columns) >= 70  # approximating the real ~79-column file


def test_compatibility_report_on_real_cic_style_dataset_is_ml_compatible() -> None:
    df = make_real_cic_style_dataset(n_benign=10, n_attack=10)
    report = build_compatibility_report(normalize_columns(df))

    assert report.is_ml_compatible is True
    assert report.missing_ml_features == ()
    assert set(report.unavailable_live_only_features) == {"protocol_is_tcp", "protocol_is_udp", "protocol_is_icmp"}
    assert report.matched_protocol_features == ()
    # Every column this project doesn't map should be reported, not silently ignored.
    assert "Init_Win_bytes_forward" in report.unmapped_dataset_columns
    assert "Active Mean" in report.unmapped_dataset_columns


def test_run_training_pipeline_succeeds_on_real_cic_style_dataset_without_protocol(tmp_path) -> None:
    """
    The exact scenario reported from real CIC-IDS2017 validation: a
    79-column-shaped dataset with no Protocol column must train
    successfully end-to-end, not be rejected as incompatible.
    """
    df = make_real_cic_style_dataset(n_benign=40, n_attack=40, seed=11)
    csv_path = tmp_path / "real_cic_style.csv"
    df.to_csv(csv_path, index=False)

    result = run_training_pipeline(dataset_path=str(csv_path), output_dir=str(tmp_path / "models"), random_seed=42)

    assert "ML-trainable: True" in result.compatibility_summary
    assert "protocol_is_tcp" in result.compatibility_summary or "Live-only features unavailable" in result.compatibility_summary

    with open(result.metadata_path) as f:
        metadata = json.load(f)
    assert metadata["ml_feature_count"] == 38
    assert "protocol_is_tcp" not in metadata["ml_feature_names"]


def test_compatibility_report_on_fully_compatible_dataset_is_compatible() -> None:
    df = make_synthetic_dataset(n_benign=5, n_attack=5)
    normalized = normalize_columns(df)
    report = build_compatibility_report(normalized)

    assert report.is_ml_compatible is True
    assert report.missing_ml_features == ()
    assert report.label_column_found == "Label"
    # This synthetic dataset DOES include a Protocol column, so all 41 live
    # features (including protocol one-hot) are obtainable here.
    assert set(report.matched_live_features) == set(FEATURE_NAMES)
    assert report.unavailable_live_only_features == ()


def test_compatibility_report_detects_missing_features() -> None:
    df = make_synthetic_dataset(n_benign=5, n_attack=5)
    df = df.drop(columns=[" SYN Flag Count", " ACK Flag Count"])
    normalized = normalize_columns(df)
    report = build_compatibility_report(normalized)

    assert report.is_ml_compatible is False
    assert "syn_count" in report.missing_ml_features
    assert "ack_count" in report.missing_ml_features


def test_compatibility_report_is_ml_compatible_without_protocol_column() -> None:
    """
    The exact real-world scenario found in CIC-IDS2017's actual
    MachineLearningCSV release: no Protocol column at all. This must NOT
    be declared incompatible -- protocol_is_tcp/udp/icmp are excluded from
    ML training already, so their absence is informational, not blocking.
    """
    df = make_synthetic_dataset(n_benign=5, n_attack=5)
    df = df.drop(columns=["Protocol"])
    normalized = normalize_columns(df)
    report = build_compatibility_report(normalized)

    assert report.is_ml_compatible is True
    assert report.missing_ml_features == ()
    assert set(report.unavailable_live_only_features) == {"protocol_is_tcp", "protocol_is_udp", "protocol_is_icmp"}
    assert report.matched_protocol_features == ()


def test_compatibility_report_detects_missing_label_column() -> None:
    df = make_synthetic_dataset(n_benign=5, n_attack=5)
    df = df.drop(columns=[" Label"])
    normalized = normalize_columns(df)
    report = build_compatibility_report(normalized)

    assert report.label_column_found is None
    assert report.is_ml_compatible is False


def test_compatibility_report_lists_unmapped_columns() -> None:
    df = make_synthetic_dataset(n_benign=3, n_attack=3)
    df["Some Extra Unmapped Column"] = 0
    normalized = normalize_columns(df)
    report = build_compatibility_report(normalized)

    assert "Some Extra Unmapped Column" in report.unmapped_dataset_columns


# ---------------------------------------------------------------------------
# Feature alignment: exact schema order, unit conversion, protocol one-hot
# ---------------------------------------------------------------------------


def test_align_features_produces_exact_ml_schema_columns_in_order() -> None:
    df = make_synthetic_dataset(n_benign=5, n_attack=5)
    aligned, labels, report = align_features(df)

    assert list(aligned.columns) == list(ML_FEATURE_NAMES)
    assert len(aligned.columns) == ML_FEATURE_COUNT
    assert len(aligned.columns) == 38
    assert len(labels) == len(aligned)
    assert report.is_ml_compatible is True


def test_align_features_converts_microseconds_to_seconds() -> None:
    df = make_synthetic_dataset(n_benign=1, n_attack=0)
    raw_duration_us = df.loc[0, " Flow Duration"]
    aligned, _labels, _report = align_features(df)

    assert aligned.loc[0, "flow_duration_seconds"] == pytest.approx(raw_duration_us / 1_000_000)


def test_align_features_computes_total_packets_and_bytes() -> None:
    df = make_synthetic_dataset(n_benign=1, n_attack=0)
    aligned, _labels, _report = align_features(df)

    row = aligned.loc[0]
    assert row["total_packets"] == pytest.approx(row["total_forward_packets"] + row["total_backward_packets"])
    assert row["total_bytes"] == pytest.approx(row["total_forward_bytes"] + row["total_backward_bytes"])


def test_align_features_excludes_protocol_features_even_when_dataset_has_protocol_column() -> None:
    """
    The ML-trainable feature set is fixed at 38 features regardless of
    whether a particular dataset happens to provide a Protocol column --
    a model must always train on the same features. protocol_is_tcp/udp/icmp
    must never appear in the ML training frame.
    """
    df = make_synthetic_dataset(n_benign=1, n_attack=0)
    df.loc[0, "Protocol"] = 17  # UDP -- present in the source data, but must still be excluded
    aligned, _labels, _report = align_features(df)

    assert "protocol_is_tcp" not in aligned.columns
    assert "protocol_is_udp" not in aligned.columns
    assert "protocol_is_icmp" not in aligned.columns
    assert set(EXCLUDED_ML_FEATURES.keys()).isdisjoint(aligned.columns)


def test_align_features_works_without_protocol_column_and_fabricates_nothing() -> None:
    """
    Reproduces the real CIC-IDS2017 MachineLearningCSV scenario: no
    Protocol column in the source data at all. align_features() must still
    succeed (protocol is excluded from ML training anyway), and must NOT
    invent a protocol_is_* value out of thin air.
    """
    df = make_synthetic_dataset(n_benign=5, n_attack=5).drop(columns=["Protocol"])
    aligned, labels, report = align_features(df)  # must not raise

    assert list(aligned.columns) == list(ML_FEATURE_NAMES)
    assert len(aligned.columns) == 38
    assert "protocol_is_tcp" not in aligned.columns
    assert "protocol_is_udp" not in aligned.columns
    assert "protocol_is_icmp" not in aligned.columns
    assert report.is_ml_compatible is True
    assert len(labels) == len(aligned)


def test_align_features_raises_on_incompatible_dataset() -> None:
    df = make_synthetic_dataset(n_benign=5, n_attack=5).drop(columns=[" SYN Flag Count"])
    with pytest.raises(FeatureAlignmentError):
        align_features(df)


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------


def test_normalize_labels_binary_maps_benign_and_attack() -> None:
    raw = pd.Series(["BENIGN", "PortScan", "DoS Hulk", " benign ", "BENIGN"])
    binary, original = normalize_labels_binary(raw)

    assert list(binary) == [0, 1, 1, 0, 0]
    assert original.iloc[3] == "benign"  # whitespace-stripped but case preserved in the original copy


# ---------------------------------------------------------------------------
# Cleaning: duplicates, NaN, infinite values -- all counted
# ---------------------------------------------------------------------------


def test_clean_aligned_dataset_removes_duplicates() -> None:
    df = make_synthetic_dataset(n_benign=5, n_attack=0)
    aligned, labels, _ = align_features(df)
    duplicated = pd.concat([aligned, aligned.iloc[[0]]], ignore_index=True)
    duplicated_labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)

    cleaned, cleaned_labels, report = clean_aligned_dataset(duplicated, duplicated_labels)

    assert report.duplicate_rows_removed == 1
    assert report.final_rows == len(aligned)
    assert len(cleaned) == len(cleaned_labels)


def test_clean_aligned_dataset_removes_nan_rows() -> None:
    df = make_synthetic_dataset(n_benign=5, n_attack=0)
    aligned, labels, _ = align_features(df)
    aligned.loc[0, "packet_length_mean"] = np.nan

    cleaned, _cleaned_labels, report = clean_aligned_dataset(aligned, labels)

    assert report.missing_value_rows_removed == 1
    assert report.final_rows == len(aligned) - 1
    assert not cleaned.isna().any().any()


def test_clean_aligned_dataset_removes_infinite_rows() -> None:
    df = make_synthetic_dataset(n_benign=5, n_attack=0)
    aligned, labels, _ = align_features(df)
    aligned.loc[0, "total_bytes_per_second"] = np.inf

    cleaned, _cleaned_labels, report = clean_aligned_dataset(aligned, labels)

    assert report.infinite_value_rows_removed == 1
    assert not np.isinf(cleaned.to_numpy(dtype=float)).any()


def test_cleaning_report_counts_are_consistent() -> None:
    df = make_synthetic_dataset(n_benign=10, n_attack=0)
    aligned, labels, _ = align_features(df)

    _cleaned, _cleaned_labels, report = clean_aligned_dataset(aligned, labels)

    assert report.initial_rows == 10
    assert report.final_rows == (
        report.initial_rows - report.duplicate_rows_removed - report.infinite_value_rows_removed - report.missing_value_rows_removed
    )


# ---------------------------------------------------------------------------
# Train/validation/test split -- no leakage
# ---------------------------------------------------------------------------


def test_split_and_scale_produces_correctly_sized_splits() -> None:
    df = make_synthetic_dataset(n_benign=40, n_attack=40)
    aligned, raw_labels, _ = align_features(df)
    binary_labels, _ = normalize_labels_binary(raw_labels)
    aligned, binary_labels, _ = clean_aligned_dataset(aligned, binary_labels)

    split = split_and_scale(aligned, binary_labels, val_size=0.2, test_size=0.2, random_state=42)

    total = len(aligned)
    assert len(split.X_train) + len(split.X_val) + len(split.X_test) == total
    assert split.X_train.shape[1] == ML_FEATURE_COUNT


def test_split_and_scale_scaler_is_fit_only_on_training_data() -> None:
    df = make_synthetic_dataset(n_benign=40, n_attack=40)
    aligned, raw_labels, _ = align_features(df)
    binary_labels, _ = normalize_labels_binary(raw_labels)
    aligned, binary_labels, _ = clean_aligned_dataset(aligned, binary_labels)

    split = split_and_scale(aligned, binary_labels, random_state=42)

    # Scaled training data should have ~zero mean per feature, and ~unit
    # variance for any feature that actually varies. The ML-trainable frame
    # no longer includes protocol_is_* columns at all (excluded from ML
    # training -- see app.ml.ml_feature_schema), but some other synthetic
    # columns here may still be constant across all rows; StandardScaler
    # correctly leaves a zero-variance feature unscaled rather than
    # inventing a unit variance for it, so those columns are excluded from
    # the check. Note scaler.scale_ is set to 1.0 by sklearn for
    # zero-variance columns specifically to avoid a division by zero, so
    # scaler.var_ -- the actual observed variance -- is the correct signal
    # to filter on here, not scale_.
    varies = split.scaler.var_ > 1e-8
    assert np.allclose(split.X_train.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(split.X_train.std(axis=0)[varies], 1.0, atol=1e-6)


def test_split_is_stratified_on_label() -> None:
    df = make_synthetic_dataset(n_benign=50, n_attack=50)
    aligned, raw_labels, _ = align_features(df)
    binary_labels, _ = normalize_labels_binary(raw_labels)
    aligned, binary_labels, _ = clean_aligned_dataset(aligned, binary_labels)

    split = split_and_scale(aligned, binary_labels, val_size=0.2, test_size=0.2, random_state=42)

    # With balanced input classes and stratification, each split should be roughly balanced too.
    for y in (split.y_train, split.y_val, split.y_test):
        ratio = y.mean()
        assert 0.3 < ratio < 0.7


# ---------------------------------------------------------------------------
# Model training: Logistic Regression, Decision Tree, Random Forest
# ---------------------------------------------------------------------------


def _prepared_split(n_benign: int = 60, n_attack: int = 60, seed: int = 7):
    df = make_synthetic_dataset(n_benign=n_benign, n_attack=n_attack, seed=seed)
    aligned, raw_labels, _ = align_features(df)
    binary_labels, _ = normalize_labels_binary(raw_labels)
    aligned, binary_labels, _ = clean_aligned_dataset(aligned, binary_labels)
    return split_and_scale(aligned, binary_labels, random_state=42)


def test_logistic_regression_trains_and_evaluates() -> None:
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    result = train_and_evaluate(
        models["logistic_regression"], "logistic_regression", split.X_train, split.y_train, split.X_val, split.y_val
    )

    assert result.model_name == "logistic_regression"
    assert 0.0 <= result.accuracy <= 1.0
    assert result.true_positives + result.true_negatives + result.false_positives + result.false_negatives == len(split.y_val)


def test_decision_tree_trains_and_evaluates() -> None:
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    result = train_and_evaluate(models["decision_tree"], "decision_tree", split.X_train, split.y_train, split.X_val, split.y_val)

    assert result.model_name == "decision_tree"
    assert 0.0 <= result.f1 <= 1.0


def test_random_forest_trains_and_evaluates() -> None:
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    result = train_and_evaluate(models["random_forest"], "random_forest", split.X_train, split.y_train, split.X_val, split.y_val)

    assert result.model_name == "random_forest"
    assert 0.0 <= result.macro_f1 <= 1.0


def test_metric_calculation_confusion_matrix_shape_and_consistency() -> None:
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    result = train_and_evaluate(models["random_forest"], "random_forest", split.X_train, split.y_train, split.X_val, split.y_val)

    assert len(result.confusion_matrix) == 2
    assert all(len(row) == 2 for row in result.confusion_matrix)
    total_from_cm = sum(sum(row) for row in result.confusion_matrix)
    assert total_from_cm == len(split.y_val)


def test_select_best_model_picks_highest_macro_f1() -> None:
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    results = {
        name: train_and_evaluate(model, name, split.X_train, split.y_train, split.X_val, split.y_val)
        for name, model in models.items()
    }

    best = select_best_model(results)
    assert best in results
    assert results[best].macro_f1 == max(r.macro_f1 for r in results.values())


def test_evaluate_fitted_model_does_not_refit() -> None:
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    model = models["decision_tree"]
    model.fit(split.X_train, split.y_train)

    result_before = evaluate_fitted_model(model, "decision_tree", split.X_test, split.y_test)
    result_again = evaluate_fitted_model(model, "decision_tree", split.X_test, split.y_test)

    assert result_before.accuracy == result_again.accuracy  # deterministic, no refitting side effects
    assert result_before.training_time_seconds == 0.0


# ---------------------------------------------------------------------------
# Model persistence + schema-version validation
# ---------------------------------------------------------------------------


def test_save_and_load_model_artifact_round_trip(tmp_path) -> None:
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    model = models["decision_tree"]
    result = train_and_evaluate(model, "decision_tree", split.X_train, split.y_train, split.X_val, split.y_val)

    model_path, scaler_path, metadata_path = save_model_artifact(
        model=model,
        scaler=split.scaler,
        model_name="decision_tree",
        evaluation=result,
        training_config={"note": "unit test"},
        dataset_info={"dataset_path": "synthetic", "raw_row_count": 120},
        output_dir=str(tmp_path),
    )

    assert model_path.exists()
    assert scaler_path.exists()
    assert metadata_path.exists()

    with open(metadata_path) as f:
        metadata = json.load(f)
    assert metadata["live_feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert metadata["ml_feature_schema_version"] == ML_FEATURE_SCHEMA_VERSION
    assert metadata["ml_feature_names"] == list(ML_FEATURE_NAMES)
    assert metadata["ml_feature_count"] == 38
    assert metadata["label_mapping"] == {"BENIGN": 0, "ATTACK": 1}

    loaded_model, loaded_scaler, loaded_metadata = load_model_artifact(str(model_path), str(scaler_path), str(metadata_path))
    predictions_original = model.predict(split.X_test)
    predictions_loaded = loaded_model.predict(split.X_test)
    assert list(predictions_original) == list(predictions_loaded)
    assert loaded_metadata["model_type"] == "decision_tree"


def test_saved_metadata_records_excluded_features_dataset_and_scaler_info(tmp_path) -> None:
    """
    Covers Phase 6 requirement #7: saved metadata must record excluded
    features (and why), dataset information, and scaler information -- not
    just the model type and feature list.
    """
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    model = models["random_forest"]
    result = train_and_evaluate(model, "random_forest", split.X_train, split.y_train, split.X_val, split.y_val)

    dataset_info = {
        "dataset_path": "/data/Monday-WorkingHours.pcap_ISCX.csv",
        "raw_row_count": 529918,
        "sample_size": 10000,
    }

    _model_path, _scaler_path, metadata_path = save_model_artifact(
        model=model,
        scaler=split.scaler,
        model_name="random_forest",
        evaluation=result,
        training_config={"random_seed": 42},
        dataset_info=dataset_info,
        output_dir=str(tmp_path),
    )

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert metadata["excluded_features"] == dict(EXCLUDED_ML_FEATURES)
    assert set(metadata["excluded_features"].keys()) == {"protocol_is_tcp", "protocol_is_udp", "protocol_is_icmp"}
    assert all(isinstance(reason, str) and len(reason) > 0 for reason in metadata["excluded_features"].values())

    assert metadata["dataset_info"] == dataset_info

    assert metadata["scaler_info"]["type"] == "StandardScaler"
    assert metadata["scaler_info"]["n_features_in"] == 38


def test_load_model_artifact_rejects_ml_schema_version_mismatch(tmp_path) -> None:
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    model = models["decision_tree"]
    result = train_and_evaluate(model, "decision_tree", split.X_train, split.y_train, split.X_val, split.y_val)

    model_path, scaler_path, metadata_path = save_model_artifact(
        model=model,
        scaler=split.scaler,
        model_name="decision_tree",
        evaluation=result,
        training_config={},
        dataset_info={},
        output_dir=str(tmp_path),
    )

    with open(metadata_path) as f:
        metadata = json.load(f)
    metadata["ml_feature_schema_version"] = "0.0.1-old-ml38"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    with pytest.raises(SchemaVersionMismatchError):
        load_model_artifact(str(model_path), str(scaler_path), str(metadata_path))


def test_load_model_artifact_rejects_ml_feature_name_mismatch(tmp_path) -> None:
    split = _prepared_split()
    models = build_baseline_models(random_seed=42)
    model = models["decision_tree"]
    result = train_and_evaluate(model, "decision_tree", split.X_train, split.y_train, split.X_val, split.y_val)

    model_path, scaler_path, metadata_path = save_model_artifact(
        model=model,
        scaler=split.scaler,
        model_name="decision_tree",
        evaluation=result,
        training_config={},
        dataset_info={},
        output_dir=str(tmp_path),
    )

    with open(metadata_path) as f:
        metadata = json.load(f)
    metadata["ml_feature_names"][0] = "totally_different_feature"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    with pytest.raises(SchemaVersionMismatchError):
        load_model_artifact(str(model_path), str(scaler_path), str(metadata_path))


# ---------------------------------------------------------------------------
# Full pipeline integration (still synthetic data, but exercises everything together)
# ---------------------------------------------------------------------------


def test_run_training_pipeline_end_to_end(tmp_path) -> None:
    df = make_synthetic_dataset(n_benign=60, n_attack=60, seed=3)
    csv_path = tmp_path / "synthetic_full.csv"
    df.to_csv(csv_path, index=False)
    output_dir = tmp_path / "models"

    result = run_training_pipeline(dataset_path=str(csv_path), output_dir=str(output_dir), random_seed=42)

    assert result.best_model_name in {"logistic_regression", "decision_tree", "random_forest"}
    assert len(result.val_results) == 3
    assert len(result.test_results) == 1
    assert "ML-trainable: True" in result.compatibility_summary

    from pathlib import Path

    assert Path(result.model_path).exists()
    assert Path(result.scaler_path).exists()
    assert Path(result.metadata_path).exists()


def test_run_training_pipeline_raises_on_incompatible_dataset(tmp_path) -> None:
    df = make_synthetic_dataset(n_benign=10, n_attack=10)
    df = df.drop(columns=[" SYN Flag Count", " ACK Flag Count", " Label"])
    csv_path = tmp_path / "broken.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(FeatureAlignmentError):
        run_training_pipeline(dataset_path=str(csv_path), output_dir=str(tmp_path / "models"))


def test_run_training_pipeline_respects_sample_size(tmp_path) -> None:
    df = make_synthetic_dataset(n_benign=100, n_attack=100, seed=9)
    csv_path = tmp_path / "large.csv"
    df.to_csv(csv_path, index=False)

    result = run_training_pipeline(
        dataset_path=str(csv_path), output_dir=str(tmp_path / "models"), sample_size=40, random_seed=42
    )
    best = result.best_model_name
    total_evaluated = (
        result.val_results[best].true_positives
        + result.val_results[best].true_negatives
        + result.val_results[best].false_positives
        + result.val_results[best].false_negatives
    )
    assert total_evaluated <= 40