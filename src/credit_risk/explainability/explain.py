"""Model explainability.

Two complementary views are provided, and both are described as *model
associations*, never as causal effects:

1. **Global importance** — native coefficients/tree importances and
   model-agnostic permutation importance over the raw input features.
2. **Local attribution** — exact additive contributions for a linear model
   (``coefficient x standardised value``), or SHAP values when ``shap`` is
   installed. For a linear model the decomposition is exact, so no sampling
   approximation is involved.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)

# Human-readable labels used in explanations. Engineered features first, then
# the raw statement-history columns.
FEATURE_LABELS: dict[str, str] = {
    "payment_delay_rate": "late-payment frequency",
    "serious_delay_rate": "serious delinquency frequency",
    "max_delay_months": "worst payment delay (months)",
    "pay_duly_rate": "share of months paid in full",
    "revolving_use_rate": "share of months with revolving balance",
    "total_bill_amount": "total billed amount",
    "avg_bill_amount": "average monthly bill",
    "bill_volatility": "bill volatility",
    "total_payment_amount": "total repaid amount",
    "avg_payment_amount": "average monthly repayment",
    "payment_volatility": "repayment volatility",
    "net_cash_flow": "net cash flow (repayments - bills)",
    "payment_to_bill_ratio": "repayment-to-bill ratio",
    "credit_utilization": "credit utilisation",
    "payment_to_limit_ratio": "repayment-to-limit ratio",
    "bill_trend": "bill trend over window",
    "recent_delay_flag": "most recent month delinquency",
    "limit_bal": "credit limit",
    "age": "age",
    "education": "education level",
}
for _i in range(7):
    FEATURE_LABELS[f"pay_{_i}"] = f"repayment status (month -{_i})"
for _i in range(1, 7):
    FEATURE_LABELS[f"bill_amt{_i}"] = f"statement balance (month -{_i})"
    FEATURE_LABELS[f"pay_amt{_i}"] = f"amount repaid (month -{_i})"


def humanize(feature: str) -> str:
    """Return a readable label for a transformed (possibly one-hot) feature name.

    One-hot names such as ``cat__education_3.0`` are reduced to the underlying
    column and the encoded level so each column keeps a distinct label.
    """
    base = feature.split("__")[-1]
    if base in FEATURE_LABELS:
        return FEATURE_LABELS[base]
    # One-hot encoded categorical: "<column>_<level>".
    for prefix, label in (
        ("education_", "education level "),
        ("age_", "age band "),
        ("sex_", "sex "),
        ("marriage_", "marital status "),
    ):
        if base.startswith(prefix):
            return f"{label}{base[len(prefix):]}"
    return base.replace("_", " ")


def get_transformed_feature_names(preprocessor: Any) -> list[str]:
    """Return the model-input feature names produced by a fitted pipeline."""
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError("Preprocessor cannot report feature names; fit it first.") from exc


def feature_importance_df(estimator: BaseEstimator, feature_names: list[str]) -> pd.DataFrame:
    """Global importance from tree importances or absolute linear coefficients."""
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:  # pragma: no cover
        raise ValueError(
            f"{type(estimator).__name__} exposes neither feature_importances_ nor coef_."
        )
    if len(values) != len(feature_names):
        raise ValueError(
            f"Importance length {len(values)} does not match {len(feature_names)} feature names."
        )
    frame = pd.DataFrame({"feature": feature_names, "importance": values})
    frame["label"] = frame["feature"].map(humanize)
    return frame.sort_values("importance", ascending=False, ignore_index=True)


def permutation_importance_df(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    n_repeats: int = 10,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Model-agnostic permutation importance over the **raw** input features.

    ``pipeline`` must be the full preprocessed-classifier fitted on raw feature
    space, and ``X`` a raw-feature DataFrame. Importance is the drop in ROC-AUC
    when a raw column is shuffled — a statement about the model, not the feature.
    """
    cfg = get_config()
    random_state = cfg.seed if random_state is None else random_state
    result = permutation_importance(
        pipeline, X, y, n_repeats=n_repeats, random_state=random_state,
        n_jobs=1, scoring="roc_auc",
    )
    frame = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    frame["label"] = frame["feature"].map(humanize)
    return frame.sort_values("importance_mean", ascending=False, ignore_index=True)


def is_linear(estimator: BaseEstimator) -> bool:
    """Whether the estimator exposes signed linear coefficients."""
    return hasattr(estimator, "coef_") and not hasattr(estimator, "feature_importances_")


def local_contributions(
    preprocessor: Any,
    classifier: BaseEstimator,
    X_raw: pd.DataFrame,
    top_k: int = 8,
) -> dict[str, list[dict[str, float | str]]] | None:
    """Exact per-feature contributions for a linear classifier.

    For a standardised linear model the log-odds decompose additively as
    ``sum_i coef_i * x_i``. This returns the largest positive (risk-increasing)
    and negative (protective) contributions for a single record — an exact
    explanation, not a sampled approximation.

    Returns ``None`` when the classifier is not linear.
    """
    if not is_linear(classifier):
        return None
    names = get_transformed_feature_names(preprocessor)
    transformed = np.asarray(preprocessor.transform(X_raw), dtype=float)
    coef = np.asarray(classifier.coef_, dtype=float).ravel()
    if transformed.shape[0] != 1:
        raise ValueError("local_contributions expects exactly one record.")
    contributions = coef * transformed[0]
    frame = pd.DataFrame(
        {"feature": names, "label": [humanize(n) for n in names], "contribution": contributions}
    )
    risk_increasing = frame[frame["contribution"] > 0].sort_values("contribution", ascending=False).head(top_k)
    protective = frame[frame["contribution"] < 0].sort_values("contribution").head(top_k)

    def _records(part: pd.DataFrame) -> list[dict[str, float | str]]:
        return [
            {"feature": row.label, "contribution": round(float(row.contribution), 4)}
            for row in part.itertuples()
        ]

    return {"risk_increasing": _records(risk_increasing), "protective": _records(protective)}


def ablation_contributions(
    model: Any,
    record: pd.DataFrame,
    reference: pd.DataFrame,
    top_k: int = 8,
) -> dict[str, list[dict[str, float | str]]]:
    """Model-agnostic local attribution by occlusion (ablation).

    The record's calibrated probability is compared with the probability when a
    single feature is reset to a reference value (the training median). The
    change isolates that feature's effect *for this model on this record* — it is
    an explanation of the model, not a causal effect of the feature.

    Returns the top risk-increasing and protective features.
    """
    baseline = float(np.asarray(model.predict_proba(record))[0])
    rows: list[dict[str, float | str]] = []
    for column in record.columns:
        perturbed = record.copy()
        perturbed[column] = reference[column].median()
        perturbed_proba = float(np.asarray(model.predict_proba(perturbed))[0])
        rows.append(
            {
                "feature": humanize(column),
                "contribution": round(baseline - perturbed_proba, 4),
            }
        )
    frame = pd.DataFrame(rows)
    risk_increasing = frame[frame["contribution"] > 0].sort_values("contribution", ascending=False).head(top_k)
    protective = frame[frame["contribution"] < 0].sort_values("contribution").head(top_k)
    return {
        "risk_increasing": risk_increasing.to_dict(orient="records"),
        "protective": protective.to_dict(orient="records"),
    }


def fit_shap_explainer(estimator: BaseEstimator, X_transformed: np.ndarray):
    """Fit a SHAP explainer when the optional dependency is installed."""
    try:
        import shap  # noqa: PLC0415 - optional dependency
    except ImportError:
        logger.info("shap not installed; using native/permutation explanations instead.")
        return None
    try:
        if hasattr(estimator, "feature_importances_"):
            return shap.TreeExplainer(estimator)
        return shap.Explainer(estimator, X_transformed)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Failed to build SHAP explainer: %s", exc)
        return None


def plot_shap_summary(
    explainer: Any,
    X_transformed: np.ndarray,
    feature_names: list[str],
    name: str = "shap_summary.png",
    n_samples: int = 1000,
) -> str | None:
    """Render and save a SHAP beeswarm summary if SHAP is available."""
    if explainer is None:
        return None
    try:
        import shap  # noqa: PLC0415

        X = np.asarray(X_transformed)
        if X.shape[0] > n_samples:
            rng = np.random.default_rng(42)
            X = X[rng.choice(X.shape[0], n_samples, replace=False)]
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[-1]
        shap.summary_plot(values, X, feature_names=feature_names, show=False, max_display=20)
        fig = plt.gcf()
        cfg = get_config()
        path = cfg.figures_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("SHAP plotting failed: %s", exc)
        return None
