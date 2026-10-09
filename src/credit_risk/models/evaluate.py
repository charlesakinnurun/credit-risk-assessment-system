"""Evaluation metrics, threshold selection, plots, and reporting.

Credit risk is cost-asymmetric: a false negative (approving an applicant who
defaults) and a false positive (declining a creditworthy applicant) have
different, context-dependent financial costs. We therefore report a broad,
calibration-aware metric suite and select the operating threshold from
**validation data** using an explicit business-cost model, never from the test
set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full classification/risk metric suite.

    Args:
        y_true: Ground-truth default labels.
        y_proba: Predicted default probabilities.
        threshold: Threshold used to derive hard labels (recorded in output).

    Returns:
        Mapping of metric name to value, with ``1`` (default) as the positive
        class.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, np.clip(y_proba, 1e-15, 1 - 1e-15))),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "n": int(len(y_true)),
        "n_defaults": int(y_true.sum()),
    }


def expected_business_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_false_negative: float,
    cost_false_positive: float,
) -> float:
    """Expected cost of the confusion matrix at ``threshold``.

    Cost is expressed in arbitrary "cost units" proportional to the configured
    per-error costs, so it is only meaningful for comparing thresholds on the
    same dataset.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(cost_false_negative * fn + cost_false_positive * fp)


def threshold_sweep(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """Sweep thresholds and return precision/recall/F1/FPR/FNR/cost at each."""
    cfg = get_config()
    thresholds = np.arange(cfg.threshold.min, cfg.threshold.max + 1e-9, cfg.threshold.step)
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append(
            {
                "threshold": float(round(float(threshold), 4)),
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
                "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
                "fnr": fn / (fn + tp) if (fn + tp) else 0.0,
                "expected_cost": expected_business_cost(
                    y_true,
                    y_proba,
                    threshold,
                    cfg.threshold.cost_false_negative,
                    cfg.threshold.cost_false_positive,
                ),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float | str]:
    """Choose the operating threshold on held-out (validation) data.

    Two candidates are computed and both reported:

    * the **F1-optimal** threshold, which balances precision and recall;
    * the **business-cost-optimal** threshold, which minimises expected cost
      under the configured FN/FP cost ratio (default: an FN costs 5x an FP).

    The business-cost optimum is returned as the operating point because the
    system's purpose is to make a cost-aware credit decision; the F1 optimum is
    retained for transparency.
    """
    table = threshold_sweep(y_true, y_proba)
    f1_row = table.loc[table["f1"].idxmax()]
    cost_row = table.loc[table["expected_cost"].idxmin()]
    logger.info(
        "Threshold selection (validation): F1-optimal=%.3f (F1=%.4f) | cost-optimal=%.3f (cost=%.1f)",
        f1_row["threshold"],
        f1_row["f1"],
        cost_row["threshold"],
        cost_row["expected_cost"],
    )
    return {
        "operating_threshold": float(cost_row["threshold"]),
        "f1_optimal_threshold": float(f1_row["threshold"]),
        "cost_optimal_threshold": float(cost_row["threshold"]),
        "f1_at_operating": float(f1_row["f1"]),
        "expected_cost_operating": float(cost_row["expected_cost"]),
        "selection_rule": "business_cost_minimisation",
    }


def group_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    groups: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Per-group classification metrics for a responsible-ML audit.

    Returns precision, recall, FPR, FNR, and selection rate for each group value
    at the frozen operating threshold. Small groups are still reported but the
    caller should treat their metrics as high variance.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = (y_proba >= threshold).astype(int)
    labels = pd.Series(groups).astype(str).to_numpy()
    rows = []
    for value in sorted(pd.unique(labels), key=str):
        idx = np.where(labels == value)[0]
        yt, yp = y_true[idx], y_pred[idx]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append(
            {
                "group": value,
                "n": int(len(idx)),
                "n_defaults": int(yt.sum()),
                "base_rate": float(yt.mean()),
                "selection_rate": float(yp.mean()),
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
                "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def metrics_frame(rows: list[tuple[str, dict[str, float]]]) -> pd.DataFrame:
    """Build a tidy comparison table from ``(name, metrics)`` pairs."""
    frame = pd.DataFrame([{"model": name, **metrics} for name, metrics in rows])
    rounded = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "false_positive_rate",
        "false_negative_rate",
    ]
    for col in rounded:
        if col in frame.columns:
            frame[col] = frame[col].round(4)
    return frame


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Persist a figure to the configured figures directory and close it."""
    cfg = get_config()
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.figures_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_json(data: dict[str, Any], name: str, directory: Path | None = None) -> Path:
    """Write a JSON report to the configured reports directory."""
    cfg = get_config()
    directory = directory or cfg.reports_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true, y_pred, name: str = "confusion_matrix.png") -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["No default", "Default"])
    ax.set_yticks([0, 1], ["No default", "Default"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return save_figure(fig, name)


def plot_roc_curve(y_true, y_proba, name: str = "roc_curve.png") -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    return save_figure(fig, name)


def plot_pr_curve(y_true, y_proba, name: str = "pr_curve.png") -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Model (AP = {ap:.3f})", lw=2)
    ax.axhline(baseline, color="gray", ls="--", lw=1, label=f"Prevalence = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (test set)")
    ax.legend(loc="lower left")
    return save_figure(fig, name)


def plot_calibration_curve(y_true, y_proba, name: str = "calibration_curve.png") -> Path:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    brier = brier_score_loss(y_true, y_proba)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Calibration curve (test set)\nBrier score = {brier:.4f}")
    ax.legend()
    return save_figure(fig, name)


def plot_threshold_analysis(y_true, y_proba, name: str = "threshold_analysis.png") -> Path:
    table = threshold_sweep(y_true, y_proba)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for col, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("fpr", "False positive rate"),
        ("fnr", "False negative rate"),
    ]:
        axes[0].plot(table["threshold"], table[col], marker="o", ms=3, label=label)
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Metric")
    axes[0].set_title("Metric vs threshold (test set)")
    axes[0].legend(fontsize=8)

    axes[1].plot(table["threshold"], table["expected_cost"], color="#C44E52", marker="o", ms=3)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Expected cost (units)")
    axes[1].set_title("Expected business cost vs threshold")
    return save_figure(fig, name)


def plot_target_distribution(data: pd.Series, name: str = "target_distribution.png") -> Path:
    counts = data.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["No default (0)", "Default (1)"], counts.values, color=["#4C72B0", "#C44E52"])
    for i, value in enumerate(counts.values):
        ax.text(i, value, f"{value:,}\n({value / len(data):.1%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_ylabel("Count")
    ax.set_title("Target class distribution")
    return save_figure(fig, name)


def plot_feature_importance(importance: pd.DataFrame, name: str = "feature_importance.png",
                            top_n: int = 20) -> Path:
    top = importance.sort_values("importance", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
    ax.barh(top["feature"], top["importance"], color="#55A868")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {len(top)} features by importance (final model)")
    return save_figure(fig, name)


def evaluate_saved_model(split: str = "test") -> dict[str, Any]:
    """Load the persisted model and evaluate it on a held-out split.

    Reports metrics at the frozen operating threshold and, for transparency, at
    the F1-optimal threshold as well. This is a read-only audit of the saved
    artifact — it never refits or tunes anything.

    Args:
        split: ``"test"`` (default), ``"validation"``, or ``"train"``.
    """
    from credit_risk.data.ingestion import load_raw_data
    from credit_risk.data.preprocessing import build_splits, clean_data
    from credit_risk.models.predict import load_model

    model = load_model()
    df = clean_data(load_raw_data())
    splits = build_splits(df)
    frames = {
        "train": (splits.X_train, splits.y_train),
        "validation": (splits.X_val, splits.y_val),
        "test": (splits.X_test, splits.y_test),
    }
    if split not in frames:
        raise ValueError(f"split must be one of {sorted(frames)}")
    X, y = frames[split]
    proba = model.predict_proba(X)

    operating = float(model.approve_threshold)
    at_operating = compute_metrics(y.to_numpy(), proba, threshold=operating)
    f1_optimal = float(select_threshold(y.to_numpy(), proba)["f1_optimal_threshold"])
    at_f1 = compute_metrics(y.to_numpy(), proba, threshold=f1_optimal)
    return {
        "split": split,
        "model_version": model.metadata.get("model_version", "unknown"),
        "operating_threshold": operating,
        "metrics_at_operating_threshold": at_operating,
        "metrics_at_f1_threshold": at_f1,
    }


def main() -> None:
    """CLI: evaluate the persisted model on the test split."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the saved credit-risk model.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    args = parser.parse_args()
    result = evaluate_saved_model(args.split)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
