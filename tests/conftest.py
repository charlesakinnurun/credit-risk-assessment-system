"""Shared pytest fixtures.

All fixtures are synthetic so the suite runs in CI without the (git-ignored)
real dataset. Test data deliberately includes out-of-codebook categorical codes
and a controlled default rate so validation and prevalence checks are exercised.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from credit_risk.config import get_config, reload_config
from credit_risk.data.preprocessing import build_preprocessing_pipeline, build_splits, clean_data
from credit_risk.models.predict import CreditRiskModel


def _synthetic_raw(n: int = 600, seed: int = 0, default_rate: float = 0.22) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pay_cols = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
    bill_cols = [f"bill_amt{i}" for i in range(1, 7)]
    pay_amt_cols = [f"pay_amt{i}" for i in range(1, 7)]

    n_defaults = int(round(default_rate * n))
    target = np.zeros(n, dtype=int)
    target[:n_defaults] = 1
    rng.shuffle(target)

    data: dict[str, object] = {
        "id": np.arange(1, n + 1),
        "limit_bal": rng.integers(10_000, 500_000, n).astype(float),
        "sex": rng.integers(1, 3, n),
        # Include out-of-codebook codes (0, 5, 6) to exercise cleaning.
        "education": rng.integers(0, 7, n),
        "marriage": rng.integers(0, 4, n),
        "age": rng.integers(21, 70, n),
    }
    for col in pay_cols:
        data[col] = rng.integers(-2, 9, n)
    for col in bill_cols:
        data[col] = rng.integers(-5_000, 200_000, n).astype(float)
    for col in pay_amt_cols:
        data[col] = rng.integers(0, 50_000, n).astype(float)
    data["default_payment_next_month"] = target
    return pd.DataFrame(data)


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Raw (renamed) synthetic dataset with a controlled default rate."""
    return _synthetic_raw()


@pytest.fixture()
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(raw_df)


@pytest.fixture()
def splits(clean_df: pd.DataFrame):
    return build_splits(clean_df)


@pytest.fixture()
def tiny_model(splits) -> CreditRiskModel:
    """A small but complete :class:`CreditRiskModel` for fast inference tests."""
    preprocessor = build_preprocessing_pipeline().fit(splits.X_train)
    X_train_t = preprocessor.transform(splits.X_train)
    classifier = LogisticRegression(max_iter=500, random_state=0).fit(X_train_t, splits.y_train)
    cfg = get_config()
    reference = {c: float(splits.X_train[c].median()) for c in cfg.columns.features}
    return CreditRiskModel(
        preprocessor,
        classifier,
        calibrator=None,
        approve_threshold=0.5,
        decline_threshold=0.6,
        feature_columns=list(cfg.columns.features),
        reference_values=reference,
        metadata={"model_version": "test", "model_name": "LogisticRegression"},
    )


@pytest.fixture()
def model_env(tmp_path, tiny_model, monkeypatch):
    """Persist ``tiny_model`` and point the config at it via env var."""
    path = tmp_path / "credit_risk_model.joblib"
    tiny_model.save(path)
    monkeypatch.setenv("CREDIT_RISK_MODEL_PATH", str(path))
    reload_config()
    yield tiny_model
    monkeypatch.delenv("CREDIT_RISK_MODEL_PATH", raising=False)
    reload_config()
