"""Responsible-ML / fairness auditing.

The model is trained **without** the sensitive attributes (``sex``,
``marriage``); those columns are retained solely to measure whether the model's
errors fall unevenly across groups. This module never feeds them into a model.

What this analysis can and cannot say
-------------------------------------
* It measures *outcome disparities* at the frozen operating threshold: equal
  treatment across measured groups is not guaranteed, and equal group metrics
  would **not** by themselves prove fairness.
* It cannot detect bias from attributes that are absent from the data (race,
  ethnicity, disability, income) — an acknowledged limitation.
* Any credit decisioning in production requires human review; the flags here
  are a screening tool, not an approval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import get_config
from credit_risk.models.evaluate import group_metrics

# Codebook labels for readable audit output.
GROUP_LABELS: dict[str, dict[int, str]] = {
    "sex": {1: "male", 2: "female"},
    "marriage": {1: "married", 2: "single", 3: "other"},
    "education": {1: "graduate", 2: "university", 3: "high_school", 4: "other"},
}


def _age_band(age: pd.Series) -> pd.Series:
    """Bucket ages into readable audit bands; out-of-range values -> 'unknown'."""
    banded = pd.cut(
        age,
        bins=[0, 25, 35, 45, 55, np.inf],
        labels=["<=25", "26-35", "36-45", "46-55", "56+"],
    )
    return pd.Series(banded, index=age.index).astype(object).fillna("unknown").astype(str)


def audit(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    X_raw: pd.DataFrame,
    sensitive: pd.DataFrame,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    """Compute per-group metrics for each protected/audit attribute.

    Args:
        y_true: Ground-truth labels.
        y_proba: Calibrated default probabilities.
        X_raw: Raw feature frame (used to derive the age band).
        sensitive: Frame containing the sensitive columns.
        threshold: Frozen operating threshold.

    Returns:
        Mapping of attribute name -> per-group metrics DataFrame.
    """
    cfg = get_config()
    results: dict[str, pd.DataFrame] = {}

    for column in cfg.columns.sensitive:
        if column not in sensitive.columns:
            continue
        groups = sensitive[column]
        if column in GROUP_LABELS:
            groups = groups.map(GROUP_LABELS[column]).fillna(groups.astype(str))
        results[column] = group_metrics(y_true, y_proba, groups, threshold)

    if "age" in X_raw.columns:
        results["age_band"] = group_metrics(y_true, y_proba, _age_band(X_raw["age"]), threshold)

    return results


def disparity_summary(audit_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarise the max-min spread of key metrics across groups per attribute.

    The spread is a *descriptive* measure of disparity. A non-zero spread is a
    prompt for review, not a verdict of discrimination.
    """
    rows = []
    for attribute, frame in audit_results.items():
        for metric in ("recall", "false_positive_rate", "false_negative_rate", "selection_rate"):
            values = frame[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "attribute": attribute,
                    "metric": metric,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "spread": float(np.max(values) - np.min(values)),
                }
            )
    return pd.DataFrame(rows)
