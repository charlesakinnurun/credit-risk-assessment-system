"""Probability calibration for the credit-risk model.

Calibration maps a classifier's raw scores onto probabilities that can be read
literally ("of clients given 0.30, roughly 30% default"). It is fitted on the
**validation split only** — never on training or test data.

Two methods are supported and compared by validation Brier score:

* **sigmoid (Platt scaling)** — a one-dimensional logistic regression on the
  raw score; robust with small calibration sets.
* **isotonic** — a monotonic, non-parametric step function; more flexible but
  prone to overfitting when the calibration set is small.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

VALID_METHODS = ("sigmoid", "isotonic")


class ProbabilityCalibrator(BaseEstimator):
    """Recalibrate the probabilities of an already-fitted estimator.

    Args:
        estimator: A fitted estimator exposing ``predict_proba`` (typically the
            full preprocessing + classifier pipeline).
        method: ``"sigmoid"`` or ``"isotonic"``.
    """

    def __init__(self, estimator: Any = None, method: str = "sigmoid") -> None:
        self.estimator = estimator
        self.method = method

    def _raw_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X))[:, 1]

    def fit(self, X: Any, y: np.ndarray) -> ProbabilityCalibrator:
        if self.method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got {self.method!r}")
        proba = self._raw_proba(X)
        if self.method == "sigmoid":
            calibrator: Any = LogisticRegression(max_iter=2000)
            calibrator.fit(proba.reshape(-1, 1), y)
        else:
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(proba, y)
        self.calibrator_ = calibrator
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return calibrated default probabilities for the positive class."""
        proba = self._raw_proba(X)
        if self.method == "sigmoid":
            calibrated = self.calibrator_.predict_proba(proba.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator_.predict(proba)
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
