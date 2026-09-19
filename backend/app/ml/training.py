"""
Baseline model training and evaluation for Phase 6.

Trains Logistic Regression, Decision Tree, and Random Forest on the same
train/val/test split and reports the same full metric set for each, so they
can be compared fairly. No model is assumed to be "best" in advance -- see
`select_best_model()` for the (documented, simple) selection criterion
actually used.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42


@dataclass(frozen=True)
class ModelEvalResult:
    """
    Full evaluation of one trained model on one dataset split (val or test).

    Includes both standard classification metrics and the security-specific
    ones (FPR, FNR, raw TP/TN/FP/FN counts) -- accuracy alone is
    deliberately not treated as sufficient, per the project's evaluation
    requirements. All values here come from an actual sklearn computation
    on real predictions; nothing is a placeholder or assumed figure.
    """

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    weighted_f1: float
    roc_auc: float | None
    pr_auc: float | None
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    confusion_matrix: list[list[int]]
    training_time_seconds: float
    inference_time_seconds_total: float
    inference_batch_size: int

    def summary(self) -> str:
        return (
            f"{self.model_name}: acc={self.accuracy:.4f} precision={self.precision:.4f} "
            f"recall={self.recall:.4f} f1={self.f1:.4f} macro_f1={self.macro_f1:.4f} "
            f"FPR={self.false_positive_rate:.4f} FNR={self.false_negative_rate:.4f} "
            f"(TP={self.true_positives} TN={self.true_negatives} FP={self.false_positives} FN={self.false_negatives}) "
            f"train_time={self.training_time_seconds:.3f}s"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def build_baseline_models(random_seed: int = RANDOM_SEED) -> dict[str, object]:
    """
    The three required baseline models, with modest, understandable
    hyperparameters -- not the result of a hyperparameter search. Random
    seeds are fixed for reproducibility.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_seed,
            class_weight="balanced",  # intrusion datasets are class-imbalanced; see docs/methodology.md
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=15,
            random_state=random_seed,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=random_seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def _compute_eval_result(
    model: object,
    model_name: str,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    training_time_seconds: float,
) -> ModelEvalResult:
    """
    Shared metric computation for an ALREADY-FITTED model against an
    evaluation split. `training_time_seconds` is passed in rather than
    measured here since this helper doesn't do the fitting -- callers that
    fit the model measure that separately (see `train_and_evaluate`);
    callers that reuse an already-fitted model (see
    `app.ml.pipeline._evaluate_only`) pass 0.0.
    """
    start = time.perf_counter()
    y_pred = model.predict(X_eval)
    inference_time = time.perf_counter() - start

    y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None
    labels_present = set(np.unique(y_eval))
    roc_auc = float(roc_auc_score(y_eval, y_proba)) if y_proba is not None and len(labels_present) > 1 else None
    pr_auc = float(average_precision_score(y_eval, y_proba)) if y_proba is not None and len(labels_present) > 1 else None

    cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return ModelEvalResult(
        model_name=model_name,
        accuracy=float(accuracy_score(y_eval, y_pred)),
        precision=float(precision_score(y_eval, y_pred, zero_division=0)),
        recall=float(recall_score(y_eval, y_pred, zero_division=0)),
        f1=float(f1_score(y_eval, y_pred, zero_division=0)),
        macro_f1=float(f1_score(y_eval, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_eval, y_pred, average="weighted", zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        true_positives=int(tp),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        confusion_matrix=cm.tolist(),
        training_time_seconds=training_time_seconds,
        inference_time_seconds_total=inference_time,
        inference_batch_size=len(X_eval),
    )


def train_and_evaluate(model, model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray) -> ModelEvalResult:
    """Fit `model` on the training split and evaluate it on `X_eval`/`y_eval` (validation or test)."""
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    return _compute_eval_result(model, model_name, X_eval, y_eval, training_time_seconds=training_time)


def evaluate_fitted_model(model: object, model_name: str, X_eval: np.ndarray, y_eval: np.ndarray) -> ModelEvalResult:
    """
    Evaluate an ALREADY-FITTED model without refitting it. Used for a final
    test-set pass on a model already selected/fitted via `train_and_evaluate`
    on the validation split, so the test set is touched exactly once.
    """
    return _compute_eval_result(model, model_name, X_eval, y_eval, training_time_seconds=0.0)


def select_best_model(results: dict[str, ModelEvalResult]) -> str:
    """
    Select the best-performing model by macro F1 on the evaluation split.

    Macro F1 (unweighted average of per-class F1) is chosen specifically
    because it does not let a large BENIGN class majority mask poor ATTACK-
    class performance the way plain accuracy or weighted F1 could -- this
    is an explicit, documented tie-breaking choice, not a claim that macro
    F1 is universally "correct" for every intrusion-detection use case.
    """
    return max(results, key=lambda name: results[name].macro_f1)
