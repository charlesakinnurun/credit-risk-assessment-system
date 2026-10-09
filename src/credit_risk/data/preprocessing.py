"""Cleaning, splitting, and the leakage-safe preprocessing pipeline.

The preprocessing steps are wrapped in a single scikit-learn ``Pipeline`` so
that feature engineering, imputation, winsorisation, scaling, and encoding all
run through **one transform path** that is fitted once on the training split and
re-used verbatim at inference time. This is what prevents train/serve skew.

Leakage guarantees:

* every fitted statistic (medians, winsorisation quantiles, scaler parameters,
  encoder vocabulary) is fit **exclusively** on the training split;
* feature engineering is row-wise and stateless — it never looks at other rows;
* the validation split is used for calibration and threshold selection only;
* the test split is touched exactly once, after all choices are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk.config import get_config
from credit_risk.data.ingestion import load_raw_data
from credit_risk.data.validation import validate_data
from credit_risk.features.engineering import ENGINEERED_FEATURES, engineer_features
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetSplits:
    """Container for the three stratified splits with covariates kept aside."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    covariates_train: pd.DataFrame
    covariates_val: pd.DataFrame
    covariates_test: pd.DataFrame


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise out-of-codebook categorical values.

    The original dataset contains education codes (0, 4, 5, 6) and marriage code
    (0) outside the published codebook. These are mapped to an explicit
    ``"other"`` bin (4 and 3 respectively) rather than silently dropped, which
    preserves every record while keeping the categories meaningful.

    Args:
        df: Raw (renamed) credit-card DataFrame.

    Returns:
        A cleaned copy of ``df``.
    """
    from credit_risk.data.validation import (
        VALID_EDUCATION_CODES,
        VALID_MARRIAGE_CODES,
        VALID_SEX_CODES,
    )

    out = df.copy()
    out["education"] = out["education"].where(out["education"].isin(VALID_EDUCATION_CODES), 4)
    out["marriage"] = out["marriage"].where(out["marriage"].isin(VALID_MARRIAGE_CODES), 3)
    out["sex"] = out["sex"].where(out["sex"].isin(VALID_SEX_CODES), np.nan)
    return out


def split_data(
    df: pd.DataFrame,
    train_frac: float | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split.

    The dataset is a cross-sectional snapshot (one row per client, observation
    window ending one month before the label), so there is no temporal ordering
    to respect. A **stratified random split** is therefore the correct choice;
    it guarantees the rare default class keeps the same prevalence in every
    fold. Stratification uses the target only, never a feature.

    Args:
        df: Cleaned DataFrame.
        train_frac, val_frac, test_frac: Split proportions (default from config).
        random_state: Seed (default from config).

    Returns:
        ``(train, val, test)`` DataFrames.
    """
    cfg = get_config()
    train_frac = cfg.split.train if train_frac is None else train_frac
    val_frac = cfg.split.validation if val_frac is None else val_frac
    test_frac = cfg.split.test if test_frac is None else test_frac
    random_state = cfg.seed if random_state is None else random_state

    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    y = df[cfg.columns.target]
    train_df, remainder = train_test_split(
        df, train_size=train_frac, stratify=y, random_state=random_state
    )
    val_df, test_df = train_test_split(
        remainder,
        train_size=val_frac / (val_frac + test_frac),
        stratify=remainder[cfg.columns.target],
        random_state=random_state,
    )
    return train_df, val_df, test_df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target and non-feature columns.

    Returns only the configured feature columns, so identifiers and sensitive
    attributes can never leak into the model by accident.
    """
    cfg = get_config()
    return df[list(cfg.columns.features)].copy(), df[cfg.columns.target].copy()


def split_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Return the audit-only columns (identifiers and sensitive attributes)."""
    cfg = get_config()
    columns = [cfg.columns.id, *cfg.columns.sensitive]
    return df[[c for c in columns if c in df.columns]].copy()


def build_splits(df: pd.DataFrame) -> DatasetSplits:
    """Run the stratified split and isolate features, target, and covariates."""
    train_df, val_df, test_df = split_data(df)
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)
    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        covariates_train=split_covariates(train_df),
        covariates_val=split_covariates(val_df),
        covariates_test=split_covariates(test_df),
    )


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns to quantile bounds fitted on the training set.

    Extreme bill/payment amounts distort linear models and tree splits. Rather
    than dropping rows, values are capped at the training-set 1st/99th
    percentiles — a standard robust winsorisation.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y: Any = None) -> Winsorizer:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got shape {X.shape}")
        self.n_features_in_ = X.shape[1]
        self.lower_bounds_ = np.nanpercentile(X, self.lower * 100, axis=0)
        self.upper_bounds_ = np.nanpercentile(X, self.upper * 100, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.asarray(input_features, dtype=object)


class ColumnEngineeringHolder(BaseEstimator, TransformerMixin):
    """Pipeline step that applies the stateless credit feature engineering.

    Keeping engineering inside the pipeline guarantees the exact same
    transformation is applied during training and inference.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> ColumnEngineeringHolder:
        self.feature_names_in_ = list(X.columns) if hasattr(X, "columns") else None
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        if input_features is None:
            input_features = getattr(self, "feature_names_in_", None)
        names: list[Any] = [*(input_features or []), *ENGINEERED_FEATURES]
        return np.asarray(names, dtype=object)


def build_preprocessing_pipeline() -> Pipeline:
    """Build the full fitted-on-train preprocessing pipeline.

    Returns:
        ``Pipeline([engineering, ColumnTransformer(numeric, categorical)])``.
        The numeric branch imputes (median) → winsorises → standardises; the
        categorical branch imputes (most frequent) → one-hot encodes with
        ``drop="first"`` (avoids the dummy-variable trap for linear models).
    """
    cfg = get_config()
    numeric_columns = list(cfg.columns.numeric) + list(ENGINEERED_FEATURES)
    categorical_columns = list(cfg.columns.categorical)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsorizer", Winsorizer(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("engineering", ColumnEngineeringHolder()),
            ("preprocessor", column_transformer),
        ]
    )


def load_and_clean_data(path=None, *, download: bool = True) -> pd.DataFrame:
    """Load, validate, and clean the raw dataset in one reproducible step."""
    df = load_raw_data(path, download=download)
    report = validate_data(df)
    logger.info(
        "Loaded dataset: %d rows x %d cols, default rate %.3f, %d out-of-codebook categorical values",
        report.n_rows,
        report.n_columns,
        report.default_rate,
        sum(sum(v.values()) for v in report.out_of_codebook.values()),
    )
    return clean_data(df)
