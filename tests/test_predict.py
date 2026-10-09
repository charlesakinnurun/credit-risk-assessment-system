"""Inference artifact: prediction schema, probability bounds, persistence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import get_config
from credit_risk.models.predict import CreditRiskModel, load_model, to_risk_score


def _sample_record() -> dict[str, float | int]:
    cfg = get_config()
    record: dict[str, float | int] = dict.fromkeys(cfg.columns.features, 0)
    record.update({"limit_bal": 120000, "age": 33, "education": 2, "pay_0": -1, "pay_2": -1})
    return record


def test_predict_proba_bounded(tiny_model: CreditRiskModel, splits) -> None:
    proba = tiny_model.predict_proba(splits.X_val)
    assert proba.shape[0] == len(splits.X_val)
    assert np.all((proba >= 0) & (proba <= 1))


def test_predict_returns_binary_labels(tiny_model: CreditRiskModel, splits) -> None:
    labels = tiny_model.predict(splits.X_val)
    assert set(np.unique(labels)).issubset({0, 1})


def test_score_output_contract(tiny_model: CreditRiskModel, splits) -> None:
    scored = tiny_model.score(splits.X_val)
    assert list(scored.columns) == ["default_probability", "risk_tier", "decision", "risk_score"]
    assert scored["default_probability"].between(0, 1).all()
    assert set(scored["risk_tier"]).issubset({"LOW", "MEDIUM", "HIGH"})
    assert set(scored["decision"]).issubset({"APPROVE", "REVIEW", "DECLINE"})


def test_score_record_schema(tiny_model: CreditRiskModel) -> None:
    result = tiny_model.score_record(_sample_record())
    assert set(result) == {
        "default_probability", "risk_tier", "decision", "risk_score", "model_version",
    }
    assert 0.0 <= result["default_probability"] <= 1.0
    assert 0 <= result["risk_score"] <= 1000


def test_tier_and_decision_are_consistent(tiny_model: CreditRiskModel) -> None:
    assert tiny_model.tier(0.01) == "LOW"
    assert tiny_model.decision(0.01) == "APPROVE"
    assert tiny_model.tier(0.99) == "HIGH"
    assert tiny_model.decision(0.99) == "DECLINE"


def test_missing_feature_raises(tiny_model: CreditRiskModel) -> None:
    bad = pd.DataFrame([{"limit_bal": 1000, "age": 30}])
    with pytest.raises(ValueError):
        tiny_model.predict_proba(bad)


def test_to_risk_score_is_monotonic() -> None:
    assert to_risk_score(0.0) == 1000
    assert to_risk_score(1.0) == 0
    assert to_risk_score(0.3) > to_risk_score(0.6)


def test_save_load_roundtrip(tmp_path, tiny_model: CreditRiskModel, splits) -> None:
    path = tmp_path / "model.joblib"
    tiny_model.save(path)
    loaded = load_model(path)
    assert isinstance(loaded, CreditRiskModel)
    np.testing.assert_allclose(
        loaded.predict_proba(splits.X_val), tiny_model.predict_proba(splits.X_val)
    )


def test_explain_record_returns_factor_lists(tiny_model: CreditRiskModel) -> None:
    factors = tiny_model.explain_record(_sample_record())
    assert set(factors) == {"risk_increasing", "protective"}
    assert all("feature" in item and "contribution" in item for item in factors["risk_increasing"])


def test_load_model_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.joblib")
