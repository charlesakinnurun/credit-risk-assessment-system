"""Data ingestion, validation, cleaning, and splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.data.ingestion import dataset_fingerprint
from credit_risk.data.preprocessing import clean_data, split_data
from credit_risk.data.validation import DataValidationError, validate_data


def test_validate_data_reports_expected_fields(raw_df: pd.DataFrame) -> None:
    report = validate_data(raw_df)
    assert report.n_rows == len(raw_df)
    assert report.n_features == 21
    assert report.n_missing == 0
    assert 0.10 <= report.default_rate <= 0.40


def test_validate_data_flags_out_of_codebook(raw_df: pd.DataFrame) -> None:
    report = validate_data(raw_df)
    # education codes 0,5,6 and marriage code 0 exist in the synthetic data.
    assert "education" in report.out_of_codebook
    assert "marriage" in report.out_of_codebook


def test_missing_column_raises(raw_df: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        validate_data(raw_df.drop(columns=["limit_bal"]))


def test_non_binary_target_raises(raw_df: pd.DataFrame) -> None:
    bad = raw_df.copy()
    bad.loc[0, "default_payment_next_month"] = 7
    with pytest.raises(DataValidationError):
        validate_data(bad)


def test_clean_data_maps_out_of_codebook(raw_df: pd.DataFrame) -> None:
    cleaned = clean_data(raw_df)
    assert set(cleaned["education"].unique()).issubset({1, 2, 3, 4})
    assert set(cleaned["marriage"].unique()).issubset({1, 2, 3})
    # every raw code outside the codebook is remapped, not dropped
    assert len(cleaned) == len(raw_df)


def test_clean_data_does_not_mutate_input(raw_df: pd.DataFrame) -> None:
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_fingerprint_is_stable_and_changes_with_data(raw_df: pd.DataFrame) -> None:
    assert dataset_fingerprint(raw_df) == dataset_fingerprint(raw_df.copy())
    changed = raw_df.copy()
    changed.loc[0, "default_payment_next_month"] = 1 - raw_df.loc[0, "default_payment_next_month"]
    assert dataset_fingerprint(changed) != dataset_fingerprint(raw_df)


def test_split_sizes_and_stratification(clean_df: pd.DataFrame) -> None:
    train, val, test = split_data(clean_df)
    total = len(clean_df)
    assert len(train) + len(val) + len(test) == total
    assert abs(len(train) / total - 0.70) < 0.02
    rates = [df["default_payment_next_month"].mean() for df in (train, val, test)]
    overall = clean_df["default_payment_next_month"].mean()
    assert all(abs(rate - overall) < 0.05 for rate in rates)
