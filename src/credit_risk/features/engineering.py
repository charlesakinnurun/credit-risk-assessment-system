"""Credit-risk feature engineering.

Every engineered feature is **row-wise and deterministic** — no statistic is
computed across rows. This is deliberate: engineering can therefore run inside
the serialised preprocessing pipeline on unseen records with zero risk of
target leakage or train/test contamination.

All amounts are New Taiwan dollars, monthly, for the six months preceding the
label month.

Feature rationale
-----------------
* ``payment_delay_rate`` / ``serious_delay_rate`` — delinquency frequency. A
  client who repeatedly pays late is a stronger default signal than one who
  misses a single payment (Yeh & Lien, 2009).
* ``max_delay_months`` — severity of the worst delay observed.
* ``pay_duly_rate`` / ``revolving_use_rate`` — share of months paid in full vs.
  carrying a revolving balance.
* ``total_bill_amount`` / ``avg_bill_amount`` — absolute exposure level.
* ``total_payment_amount`` / ``avg_payment_amount`` — absolute repayment level.
* ``bill_volatility`` / ``payment_volatility`` — instability of balances/payments.
* ``net_cash_flow`` — total repayments minus total billed amount.
* ``payment_to_bill_ratio`` — share of billed amount repaid; the core repayment
  capability ratio.
* ``credit_utilization`` — statement balance relative to the credit limit, the
  classic bureau utilisation ratio.
* ``payment_to_limit_ratio`` — repayment relative to available credit.
* ``bill_trend`` — whether spending grew over the observation window.
* ``recent_delay_flag`` — most recent month's repayment status, which carries
  outsized signal immediately before the decision month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import get_config

ENGINEERED_FEATURES: tuple[str, ...] = (
    "payment_delay_rate",
    "serious_delay_rate",
    "max_delay_months",
    "pay_duly_rate",
    "revolving_use_rate",
    "total_bill_amount",
    "avg_bill_amount",
    "bill_volatility",
    "total_payment_amount",
    "avg_payment_amount",
    "payment_volatility",
    "net_cash_flow",
    "payment_to_bill_ratio",
    "credit_utilization",
    "payment_to_limit_ratio",
    "bill_trend",
    "recent_delay_flag",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ``numerator / denominator`` with NaN-safe division."""
    result = numerator / denominator.replace(0, np.nan)
    return result.where(np.isfinite(result), 0.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append engineered credit-risk features to ``df``.

    Args:
        df: DataFrame containing the configured raw feature columns.

    Returns:
        A new DataFrame with the original columns plus
        :data:`ENGINEERED_FEATURES`. The input is never mutated.
    """
    cfg = get_config()
    pay_cols = list(cfg.engineering.pay_columns)
    bill_cols = list(cfg.engineering.bill_columns)
    pay_amt_cols = list(cfg.engineering.payment_columns)
    limit_col = cfg.engineering.limit_column

    out = df.copy()
    pay = out[pay_cols].astype(float)
    bills = out[bill_cols].astype(float)
    payments = out[pay_amt_cols].astype(float)
    limit = out[limit_col].astype(float)

    # Delinquency -----------------------------------------------------------------
    out["payment_delay_rate"] = (pay > 0).mean(axis=1)
    out["serious_delay_rate"] = (pay >= 2).mean(axis=1)
    out["max_delay_months"] = np.maximum(pay.max(axis=1), 0.0)
    out["pay_duly_rate"] = pay.isin([-1.0, -2.0]).mean(axis=1)
    out["revolving_use_rate"] = (pay == 0).mean(axis=1)

    # Exposure and repayment ------------------------------------------------------
    out["total_bill_amount"] = bills.sum(axis=1)
    out["avg_bill_amount"] = bills.mean(axis=1)
    out["bill_volatility"] = bills.std(axis=1).fillna(0.0)
    out["total_payment_amount"] = payments.sum(axis=1)
    out["avg_payment_amount"] = payments.mean(axis=1)
    out["payment_volatility"] = payments.std(axis=1).fillna(0.0)
    out["net_cash_flow"] = out["total_payment_amount"] - out["total_bill_amount"]

    # Ratios ----------------------------------------------------------------------
    out["payment_to_bill_ratio"] = _safe_ratio(out["total_payment_amount"], out["total_bill_amount"])
    out["credit_utilization"] = _safe_ratio(out["avg_bill_amount"], limit)
    out["payment_to_limit_ratio"] = _safe_ratio(out["avg_payment_amount"], limit)
    out["bill_trend"] = _safe_ratio(
        bills[bill_cols[-1]] - bills[bill_cols[0]], bills[bill_cols[0]].abs()
    )
    out["recent_delay_flag"] = (pay[pay_cols[0]] > 0).astype(int)
    return out
