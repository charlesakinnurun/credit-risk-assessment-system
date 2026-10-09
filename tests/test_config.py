"""Configuration invariants."""

from __future__ import annotations

from credit_risk.config import get_config


def test_config_loads_and_has_expected_shape() -> None:
    cfg = get_config()
    assert cfg.seed == 42
    assert cfg.columns.target == "default_payment_next_month"
    assert cfg.columns.id == "id"


def test_sensitive_columns_are_excluded_from_features() -> None:
    cfg = get_config()
    features = set(cfg.columns.features)
    for column in cfg.columns.sensitive:
        assert column not in features, f"sensitive column {column!r} leaked into model features"


def test_splits_sum_to_one() -> None:
    cfg = get_config()
    assert abs(cfg.split.train + cfg.split.validation + cfg.split.test - 1.0) < 1e-9


def test_risk_tier_and_decision_are_consistent() -> None:
    cfg = get_config()
    approve = 0.2
    # below approve -> LOW / APPROVE
    assert cfg.risk.tier(0.05, approve) == "LOW"
    assert cfg.risk.decision(0.05, approve) == "APPROVE"
    # between approve and decline -> MEDIUM / REVIEW
    assert cfg.risk.tier(0.4, approve) == "MEDIUM"
    assert cfg.risk.decision(0.4, approve) == "REVIEW"
    # at/above decline -> HIGH / DECLINE
    assert cfg.risk.tier(0.9, approve) == "HIGH"
    assert cfg.risk.decision(0.9, approve) == "DECLINE"
