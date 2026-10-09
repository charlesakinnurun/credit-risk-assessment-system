"""Evaluation metrics and threshold logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import get_config
from credit_risk.models import evaluate as ev


def test_perfect_predictions_give_perfect_metrics() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.01, 0.02, 0.98, 0.99])
    metrics = ev.compute_metrics(y, proba, threshold=0.5)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["brier_score"] < 0.01


def test_probability_bounds_are_not_assumed_but_scored() -> None:
    # a probability exactly at the threshold counts as positive
    y = np.array([0, 1])
    metrics = ev.compute_metrics(y, np.array([0.5, 0.5]), threshold=0.5)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1


def test_threshold_sweep_covers_configured_range() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    proba = rng.random(200)
    table = ev.threshold_sweep(y, proba)
    cfg = get_config()
    assert table["threshold"].min() >= cfg.threshold.min
    assert table["threshold"].max() <= cfg.threshold.max
    assert {"precision", "recall", "f1", "expected_cost"}.issubset(table.columns)


def test_select_threshold_returns_valid_operating_point() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    proba = np.clip(0.3 * y + rng.random(400) * 0.5, 0, 1)
    info = ev.select_threshold(y, proba)
    cfg = get_config()
    assert cfg.threshold.min <= info["operating_threshold"] <= cfg.threshold.max
    assert info["operating_threshold"] == info["cost_optimal_threshold"]


def test_group_metrics_reports_each_group() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    groups = pd.Series(["a", "a", "a", "b", "b", "b"])
    table = ev.group_metrics(y, proba, groups, threshold=0.5)
    assert set(table["group"]) == {"a", "b"}
    assert {"precision", "recall", "false_positive_rate", "false_negative_rate"}.issubset(table.columns)
