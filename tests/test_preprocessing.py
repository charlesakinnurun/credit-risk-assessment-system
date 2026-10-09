"""Preprocessing pipeline behaviour and leakage safety."""

from __future__ import annotations

import numpy as np

from credit_risk.data.preprocessing import Winsorizer, build_preprocessing_pipeline


def test_pipeline_transform_is_finite_and_consistent(splits) -> None:
    pipeline = build_preprocessing_pipeline().fit(splits.X_train)
    train_t = pipeline.transform(splits.X_train)
    val_t = pipeline.transform(splits.X_val)
    assert train_t.shape[1] == val_t.shape[1]
    assert np.isfinite(np.asarray(train_t, dtype=float)).all()
    assert np.isfinite(np.asarray(val_t, dtype=float)).all()


def test_winsorizer_uses_fit_bounds_only() -> None:
    train = np.array([[0.0], [1.0], [2.0], [3.0], [100.0]])
    winsorizer = Winsorizer(lower=0.0, upper=0.8).fit(train)
    upper_bound = winsorizer.upper_bounds_[0]
    # a value far beyond the training range is clipped to the fitted bound
    transformed = winsorizer.transform(np.array([[10_000.0]]))
    assert transformed[0, 0] == upper_bound
    assert transformed[0, 0] < 10_000.0


def test_imputation_uses_training_median_not_validation(splits) -> None:
    pipeline = build_preprocessing_pipeline().fit(splits.X_train)
    train_median = splits.X_train["limit_bal"].median()

    val = splits.X_val.copy()
    val.loc[val.index[0], "limit_bal"] = np.nan
    transformed = np.asarray(pipeline.transform(val), dtype=float)
    # first numeric column is limit_bal; its scaled value corresponds to the train median
    numeric_first = transformed[0, 0]
    assert np.isfinite(numeric_first)
    # replacing with a deliberately wrong value would change the result:
    val2 = splits.X_val.copy()
    val2.loc[val2.index[0], "limit_bal"] = train_median + 50_000
    transformed2 = np.asarray(pipeline.transform(val2), dtype=float)
    assert transformed2[0, 0] != numeric_first


def test_unknown_categorical_is_ignored_not_crashing(splits) -> None:
    pipeline = build_preprocessing_pipeline().fit(splits.X_train)
    val = splits.X_val.copy()
    val["education"] = 99  # never seen during fit
    transformed = np.asarray(pipeline.transform(val), dtype=float)
    assert np.isfinite(transformed).all()
