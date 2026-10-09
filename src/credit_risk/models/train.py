"""End-to-end, reproducible training pipeline.

Order of operations (each step uses only information available before it):

1.  Load → validate → clean the raw dataset; compute a dataset fingerprint.
2.  Stratified train / validation / test split.
3.  Fit the preprocessing pipeline on the **training split only**.
4.  Baseline model progression (Dummy, Logistic Regression, Decision Tree,
    Random Forest, LightGBM) scored on validation.
5.  Hyper-parameter search on the training split (inner cross-validation).
6.  Select the final model on validation (PR-AUC primary, simplicity tie-break).
7.  Compare class-imbalance strategies (class weight vs. none) empirically.
8.  Calibrate probabilities on validation; select the method by **validation**
    Brier score (never the test set).
9.  Select the operating threshold on **validation** by expected business cost.
10. Evaluate exactly once on the untouched **test** split.
11. Audit fairness on the test split (sensitive columns are audit-only).
12. Persist figures, metrics, a comparison table, experiment record, and the
    complete :class:`CreditRiskModel` inference artifact.

Run with ``python -m credit_risk.models.train``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from credit_risk.config import ensure_directories, get_config
from credit_risk.data.ingestion import dataset_fingerprint, load_raw_data
from credit_risk.data.preprocessing import (
    build_preprocessing_pipeline,
    build_splits,
    clean_data,
)
from credit_risk.data.validation import validate_data
from credit_risk.explainability import explain
from credit_risk.features.engineering import ENGINEERED_FEATURES
from credit_risk.models import evaluate as ev
from credit_risk.models.calibration import ProbabilityCalibrator
from credit_risk.models.predict import CreditRiskModel
from credit_risk.responsible import fairness
from credit_risk.tracking import ExperimentTracker, get_git_commit
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)

# Simplicity ranking for the model-selection tie-break: lower is simpler.
COMPLEXITY_ORDER = {
    "DummyClassifier": -1,
    "LogisticRegression": 0,
    "DecisionTree": 1,
    "RandomForest": 2,
    "LightGBM": 3,
}

SEARCH_SCORE = "average_precision"  # PR-AUC; sensitive to the minority class


def _make_lightgbm(cfg) -> Any | None:
    """Return a LightGBM classifier, or ``None`` if the package is unavailable."""
    try:
        from lightgbm import LGBMClassifier  # noqa: PLC0415 - optional dependency
    except ImportError:  # pragma: no cover - optional
        logger.warning("lightgbm not installed; skipping the gradient-boosted models.")
        return None
    return LGBMClassifier(
        n_estimators=cfg.model.n_estimators,
        random_state=cfg.seed,
        n_jobs=cfg.model.n_jobs,
        verbose=-1,
    )


def _baselines(cfg) -> dict[str, Any]:
    """Fresh, untuned estimators for the baseline progression."""
    zoo: dict[str, Any] = {
        "DummyClassifier": DummyClassifier(strategy="stratified", random_state=cfg.seed),
        "LogisticRegression": LogisticRegression(max_iter=cfg.model.max_iter, random_state=cfg.seed),
        "DecisionTree": DecisionTreeClassifier(random_state=cfg.seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=cfg.model.n_estimators, n_jobs=cfg.model.n_jobs, random_state=cfg.seed
        ),
    }
    lightgbm = _make_lightgbm(cfg)
    if lightgbm is not None:
        zoo["LightGBM"] = lightgbm
    return zoo


def _search_spaces(cfg) -> dict[str, tuple[Any, dict[str, Any]]]:
    """Estimator + hyper-parameter distributions for the tuned families."""
    spaces: dict[str, tuple[Any, dict[str, Any]]] = {
        "LogisticRegression": (
            LogisticRegression(max_iter=cfg.model.max_iter, random_state=cfg.seed),
            {
                "C": loguniform(0.01, 10.0),
                "class_weight": list(cfg.training.class_weight_options),
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=cfg.model.n_estimators, n_jobs=cfg.model.n_jobs, random_state=cfg.seed
            ),
            {
                "max_depth": [None, 8, 12, 18, 24],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced", "balanced_subsample"],
            },
        ),
    }
    lightgbm = _make_lightgbm(cfg)
    if lightgbm is not None:
        spaces["LightGBM"] = (
            lightgbm,
            {
                "num_leaves": [15, 31, 63],
                "learning_rate": loguniform(0.01, 0.2),
                "n_estimators": [200, 300],
                "min_child_samples": [10, 20, 40],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_lambda": [0.0, 1.0, 5.0],
                "class_weight": [None, "balanced"],
            },
        )
    return spaces


def _score_on_validation(estimator: Any, X_val_t: np.ndarray, y_val: pd.Series) -> dict[str, float]:
    proba = estimator.predict_proba(X_val_t)[:, 1]
    return ev.compute_metrics(y_val.to_numpy(), proba, threshold=0.5)


def _tune(
    name: str, estimator: Any, params: dict[str, Any], X_train_t: np.ndarray, y_train: pd.Series, cfg
) -> tuple[Any, dict[str, Any], float]:
    folds = cfg.training.cv_folds
    if name == "RandomForest":
        folds = min(folds, 3)
    n_iter = cfg.training.search_iterations.get(name.lower(), 20)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=params,
        n_iter=n_iter,
        cv=folds,
        scoring=SEARCH_SCORE,
        n_jobs=cfg.training.n_jobs_search,
        random_state=cfg.seed,
        refit=True,
    )
    search.fit(X_train_t, y_train)
    logger.info("[tune:%s] best %s (CV %s=%.4f)", name, search.best_params_, SEARCH_SCORE, search.best_score_)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def _complexity(name: str) -> int:
    """Simplicity rank for the model-selection tie-break (lower is simpler)."""
    return COMPLEXITY_ORDER.get(name, 99)


def select_final_model(val_metrics: dict[str, dict[str, float]], tolerance: float) -> str:
    """Select the final model: best validation PR-AUC, simplicity tie-break.

    Models whose PR-AUC is within ``tolerance`` of the best are treated as
    tied; among the tied set the simplest model (lowest complexity rank) wins.
    This encodes the regulated-credit preference for an interpretable model when
    performance is comparable, without ever sacrificing a material gain.
    """
    best_name = max(val_metrics, key=lambda n: val_metrics[n]["pr_auc"])
    best_score = val_metrics[best_name]["pr_auc"]
    tied = [n for n, m in val_metrics.items() if m["pr_auc"] >= best_score - tolerance]
    final = min(tied, key=_complexity)
    logger.info(
        "[selection] best PR-AUC: %s=%.4f | tied within %.3f: %s | selected: %s",
        best_name, best_score, tolerance, tied, final,
    )
    return final


def _balance_tag(estimator: Any) -> str:
    value = getattr(estimator, "class_weight", None)
    return "balanced" if value is not None else "none"


def _compare_imbalance(
    estimator: Any, X_train_t: np.ndarray, y_train: pd.Series, X_val_t: np.ndarray, y_val: pd.Series
) -> dict[str, float]:
    """Compare class-weight strategies for the chosen family on validation PR-AUC.

    Only runs when the estimator exposes ``class_weight``. SMOTE is deliberately
    not used: it synthesises minority points in feature space, which is hard to
    justify for discrete credit-repayment features, and the calibration +
    threshold stage already handles imbalance at the decision boundary.
    """
    if not hasattr(estimator, "get_params") or "class_weight" not in estimator.get_params():
        return {}
    results: dict[str, float] = {}
    for label, weight in (("unweighted", None), ("class_weight_balanced", "balanced")):
        candidate = clone(estimator).set_params(class_weight=weight)
        candidate.fit(X_train_t, y_train)
        proba = candidate.predict_proba(X_val_t)[:, 1]
        results[label] = float(ev.compute_metrics(y_val.to_numpy(), proba)["pr_auc"])
    logger.info("[imbalance] validation PR-AUC by class weight: %s", results)
    return results


def _fit_calibrators(base_pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, cfg):
    """Fit each calibration method on validation and select by validation Brier."""
    calibrators: dict[str, ProbabilityCalibrator] = {}
    brier: dict[str, float] = {}
    for method in cfg.calibration_methods:
        calibrator = ProbabilityCalibrator(base_pipeline, method=method).fit(X_val, y_val)
        calibrators[method] = calibrator
        brier[method] = float(brier_score_loss(y_val, calibrator.predict_proba(X_val)))
    best_method = min(brier, key=lambda name: brier[name])
    logger.info("[calibration] validation Brier by method: %s -> selected %s", brier, best_method)
    return calibrators, brier, best_method


def main() -> None:
    """Run the full training pipeline."""
    ensure_directories()
    cfg = get_config()
    tracker = ExperimentTracker(run_name="training")

    # 1. Load, validate, clean ------------------------------------------------
    raw_df = load_raw_data()
    report = validate_data(raw_df)
    fingerprint = dataset_fingerprint(raw_df)
    df = clean_data(raw_df)
    logger.info(
        "[data] %d rows x %d cols, %d defaults (%.1f%%), fingerprint=%s",
        report.n_rows, report.n_columns, report.n_defaults, report.default_rate * 100, fingerprint,
    )

    # 2. Split ----------------------------------------------------------------
    splits = build_splits(df)
    logger.info(
        "[split] train=%d val=%d test=%d (stratified)",
        len(splits.X_train), len(splits.X_val), len(splits.X_test),
    )

    # 3. Preprocessing fitted on training only --------------------------------
    preprocessor = build_preprocessing_pipeline().fit(splits.X_train)
    X_train_t = preprocessor.transform(splits.X_train)
    X_val_t = preprocessor.transform(splits.X_val)
    X_test_t = preprocessor.transform(splits.X_test)

    # 4. Baseline progression -------------------------------------------------
    baseline_metrics: dict[str, dict[str, float]] = {}
    comparison_rows: list[tuple[str, dict[str, float]]] = []
    for name, estimator in _baselines(cfg).items():
        estimator.fit(X_train_t, splits.y_train)
        metrics = _score_on_validation(estimator, X_val_t, splits.y_val)
        baseline_metrics[name] = metrics
        comparison_rows.append((name, metrics))
        logger.info(
            "[baseline:%s] val ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f",
            name, metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
        )

    # 5. Hyper-parameter search ----------------------------------------------
    tuned_estimators: dict[str, Any] = {}
    tuned_params: dict[str, Any] = {}
    tuned_metrics: dict[str, dict[str, float]] = {}
    for name, (estimator, params) in _search_spaces(cfg).items():
        best_estimator, best_params, _ = _tune(
            name, estimator, params, X_train_t, splits.y_train, cfg
        )
        metrics = _score_on_validation(best_estimator, X_val_t, splits.y_val)
        tuned_estimators[name] = best_estimator
        tuned_params[name] = best_params
        tuned_metrics[name] = metrics
        comparison_rows.append((f"tuned_{name}", metrics))
        logger.info(
            "[tuned:%s] val ROC-AUC=%.4f PR-AUC=%.4f F1=%.4f",
            name, metrics["roc_auc"], metrics["pr_auc"], metrics["f1"],
        )

    # 6. Final model selection (validation only) ------------------------------
    final_name = select_final_model(tuned_metrics, cfg.training.model_selection_tolerance)
    final_estimator = tuned_estimators[final_name]
    logger.info("[final] selected estimator: %s", final_name)

    # 7. Imbalance comparison -------------------------------------------------
    imbalance = _compare_imbalance(
        final_estimator, X_train_t, splits.y_train, X_val_t, splits.y_val
    )

    # 8. Calibration (validation only) ---------------------------------------
    base_pipeline = Pipeline(
        steps=[("preprocessing", preprocessor), ("classifier", final_estimator)]
    )
    calibrators, calib_brier, best_method = _fit_calibrators(
        base_pipeline, splits.X_val, splits.y_val, cfg
    )
    final_calibrator = calibrators[best_method]

    # 9. Threshold selection (validation only) --------------------------------
    val_proba_cal = final_calibrator.predict_proba(splits.X_val)
    threshold_info = ev.select_threshold(splits.y_val.to_numpy(), val_proba_cal)
    operating_threshold = float(threshold_info["operating_threshold"])

    # 10. Single evaluation on the untouched test split -----------------------
    test_proba_cal = final_calibrator.predict_proba(splits.X_test)
    test_proba_raw = base_pipeline.predict_proba(splits.X_test)[:, 1]
    test_metrics = ev.compute_metrics(
        splits.y_test.to_numpy(), test_proba_cal, threshold=operating_threshold
    )
    logger.info(
        "[test] ROC-AUC=%.4f PR-AUC=%.4f Precision=%.4f Recall=%.4f F1=%.4f Brier=%.4f",
        test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["precision"],
        test_metrics["recall"], test_metrics["f1"], test_metrics["brier_score"],
    )
    logger.info(
        "[test] Brier uncalibrated=%.4f calibrated=%.4f (improvement %.4f)",
        float(brier_score_loss(splits.y_test, test_proba_raw)),
        test_metrics["brier_score"],
        float(brier_score_loss(splits.y_test, test_proba_raw)) - test_metrics["brier_score"],
    )

    # 11. Fairness audit on test ---------------------------------------------
    audit_results = fairness.audit(
        splits.y_test.to_numpy(), test_proba_cal,
        splits.X_test, splits.covariates_test, operating_threshold,
    )
    disparity = fairness.disparity_summary(audit_results)
    for attribute, frame in audit_results.items():
        frame.to_csv(cfg.reports_dir / f"fairness_{attribute}.csv", index=False)
    disparity.to_csv(cfg.reports_dir / "fairness_disparity_summary.csv", index=False)

    # 12. Persist the model and reports BEFORE plotting, so a cosmetic plotting
    #     failure can never discard a trained artifact.
    comparison = ev.metrics_frame(comparison_rows)
    comparison.to_csv(cfg.reports_dir / "model_comparison_validation.csv", index=False)
    ev.save_json(test_metrics, "test_metrics.json")
    ev.save_json(threshold_info, "threshold_selection.json")
    ev.save_json(
        {"validation_brier": calib_brier, "selected": best_method,
         "test_calibrated_brier": test_metrics["brier_score"]},
        "calibration_report.json",
    )
    ev.save_json({k: v.to_dict(orient="records") for k, v in audit_results.items()}, "fairness_audit.json")
    ev.save_json(report.as_dict(), "data_quality_report.json")

    model_version = f"{cfg.version}+{tracker.run_id}"
    metadata: dict[str, Any] = {
        "model_name": final_name,
        "model_version": model_version,
        "project_version": cfg.version,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "random_seed": cfg.seed,
        "class_weight": _balance_tag(final_estimator),
        "dataset": {
            "source": cfg.data.source_url,
            "fingerprint": fingerprint,
            "n_rows": report.n_rows,
            "n_features": report.n_features,
            "default_rate": round(report.default_rate, 6),
        },
        "split_sizes": {
            "train": len(splits.X_train),
            "validation": len(splits.X_val),
            "test": len(splits.X_test),
        },
        "feature_columns": list(cfg.columns.features),
        "reference_values": {c: float(splits.X_train[c].median()) for c in cfg.columns.features},
        "sensitive_columns_excluded": list(cfg.columns.sensitive),
        "best_hyperparameters": tuned_params,
        "baseline_validation_metrics": baseline_metrics,
        "tuned_validation_metrics": tuned_metrics,
        "imbalance_comparison_validation_pr_auc": imbalance,
        "calibration": {
            "method": best_method,
            "validation_brier_by_method": calib_brier,
            "test_brier": test_metrics["brier_score"],
        },
        "threshold": threshold_info,
        "test_metrics": test_metrics,
    }
    cfg.serving.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.serving.metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    model = CreditRiskModel(
        preprocessor=preprocessor,
        classifier=final_estimator,
        calibrator=final_calibrator,
        approve_threshold=operating_threshold,
        decline_threshold=cfg.risk.decline_threshold,
        feature_columns=list(cfg.columns.features),
        reference_values=metadata["reference_values"],
        metadata=metadata,
    )
    model.save()

    tracker.log(
        model_name=final_name,
        params={"best_hyperparameters": tuned_params, "threshold": operating_threshold,
                "calibration_method": best_method, "class_weight": _balance_tag(final_estimator)},
        metrics=test_metrics,
        dataset_fingerprint=fingerprint,
        feature_config={
            "features": list(cfg.columns.features),
            "engineered": list(ENGINEERED_FEATURES),
        },
        artifacts={
            "model": str(cfg.serving.model_path),
            "metadata": str(cfg.serving.metadata_path),
        },
        tags={"model_version": model_version},
    )

    # 13. Figures and explainability (best-effort; never blocks the artifact) --
    try:
        transformed_names = explain.get_transformed_feature_names(preprocessor)
        ev.plot_confusion_matrix(
            splits.y_test.to_numpy(),
            (test_proba_cal >= operating_threshold).astype(int),
        )
        ev.plot_roc_curve(splits.y_test.to_numpy(), test_proba_cal)
        ev.plot_pr_curve(splits.y_test.to_numpy(), test_proba_cal)
        ev.plot_calibration_curve(splits.y_test.to_numpy(), test_proba_cal)
        ev.plot_threshold_analysis(splits.y_test.to_numpy(), test_proba_cal)
        ev.plot_target_distribution(raw_df[cfg.columns.target])

        native_importance = explain.feature_importance_df(final_estimator, transformed_names)
        native_importance.to_csv(cfg.reports_dir / "feature_importance.csv", index=False)
        native_plot = native_importance[["label", "importance"]].rename(columns={"label": "feature"})
        ev.plot_feature_importance(native_plot, "feature_importance.png")

        perm_importance = explain.permutation_importance_df(
            base_pipeline, splits.X_test, splits.y_test.to_numpy(),
            n_repeats=cfg.training.permutation_repeats,
        )
        perm_importance.to_csv(cfg.reports_dir / "permutation_importance.csv", index=False)
        perm_plot = perm_importance[["label", "importance_mean"]].rename(
            columns={"label": "feature", "importance_mean": "importance"}
        )
        ev.plot_feature_importance(perm_plot, "permutation_importance.png")

        shap_path = explain.plot_shap_summary(
            explain.fit_shap_explainer(final_estimator, X_test_t), X_test_t, transformed_names
        )
        logger.info("[explain] SHAP summary: %s", shap_path or "unavailable (shap not installed)")
    except Exception as exc:  # pragma: no cover - plotting is non-critical
        logger.warning("[explain] figure generation failed (artifact already saved): %s", exc)

    logger.info("[done] model_version=%s; artifact=%s", model_version, cfg.serving.model_path)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
