"""Feature engineering correctness and leakage-safety."""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import get_config
from credit_risk.features.engineering import ENGINEERED_FEATURES, engineer_features


def test_engineering_adds_all_features_without_mutation(clean_df: pd.DataFrame) -> None:
    cfg = get_config()
    original = clean_df[list(cfg.columns.features)].copy()
    engineered = engineer_features(clean_df)
    for feature in ENGINEERED_FEATURES:
        assert feature in engineered.columns
    # input frame is not modified in place
    pd.testing.assert_frame_equal(clean_df[list(cfg.columns.features)], original)


def test_engineered_values_are_finite(clean_df: pd.DataFrame) -> None:
    engineered = engineer_features(clean_df)
    block = engineered[list(ENGINEERED_FEATURES)].to_numpy(dtype=float)
    assert np.isfinite(block).all()


def test_credit_utilization_matches_definition() -> None:
    cfg = get_config()
    row = dict.fromkeys(cfg.columns.features, 0.0)
    row.update(
        {
            "limit_bal": 10000.0,
            "pay_0": -1, "pay_2": -1, "pay_3": -1, "pay_4": -1, "pay_5": -1, "pay_6": -1,
            "bill_amt1": 6000, "bill_amt2": 6000, "bill_amt3": 6000,
            "bill_amt4": 6000, "bill_amt5": 6000, "bill_amt6": 6000,
            "education": 2,
        }
    )
    out = engineer_features(pd.DataFrame([row]))
    # avg bill 6000 / limit 10000 = 0.6
    assert abs(out["credit_utilization"].iloc[0] - 0.6) < 1e-9
    assert out["recent_delay_flag"].iloc[0] == 0


def test_recent_delay_flag_positive() -> None:
    cfg = get_config()
    row = dict.fromkeys(cfg.columns.features, 0.0)
    row.update({"limit_bal": 1000.0, "pay_0": 2, "education": 2, "age": 30})
    out = engineer_features(pd.DataFrame([row]))
    assert out["recent_delay_flag"].iloc[0] == 1
    assert out["max_delay_months"].iloc[0] >= 2


def test_safe_ratio_handles_zero_denominator() -> None:
    cfg = get_config()
    row = dict.fromkeys(cfg.columns.features, 0.0)
    row.update({"limit_bal": 0.0, "education": 2, "age": 30})
    out = engineer_features(pd.DataFrame([row]))
    # zero limit must not produce inf/NaN
    assert np.isfinite(out["credit_utilization"].iloc[0])
    assert np.isfinite(out["payment_to_limit_ratio"].iloc[0])
