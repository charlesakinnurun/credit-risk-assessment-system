"""The persisted inference artifact and the batch/record scoring interface.

The saved object is :class:`CreditRiskModel`, which bundles **everything**
inference needs:

* the fitted preprocessing pipeline (engineering → impute → winsorise → scale →
  one-hot),
* the fitted classifier,
* the probability calibrator (or ``None``),
* the learned approve threshold and the policy decline threshold,
* metadata (version, metrics, dataset fingerprint).

Because it is a single object, training and serving cannot drift: the same
transform path that produced the training matrix produces the inference matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_risk.config import get_config
from credit_risk.models.calibration import ProbabilityCalibrator
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)

TIER_TO_DECISION = {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "DECLINE"}


def to_risk_score(probability: float) -> int:
    """Map a default probability to an interpretable 0-1000 score (higher = safer).

    ``score = round(1000 * (1 - p))``. This is a project-defined analytical
    rescaling and is **not** equivalent to any bureau score such as FICO.
    """
    return int(round(1000 * (1 - float(np.clip(probability, 0.0, 1.0)))))


class CreditRiskModel:
    """End-to-end credit-risk model: preprocessing + classifier + calibration.

    Args:
        preprocessor: Fitted preprocessing ``Pipeline``.
        classifier: Classifier fitted on the transformed training matrix.
        calibrator: Fitted :class:`ProbabilityCalibrator`, or ``None``.
        approve_threshold: Learned operating threshold (LOW/MEDIUM boundary).
        decline_threshold: Policy threshold (MEDIUM/HIGH boundary).
        feature_columns: Ordered raw feature column names the model expects.
        metadata: Free-form metadata (persisted alongside the model).
    """

    def __init__(
        self,
        preprocessor: Any,
        classifier: Any,
        *,
        calibrator: ProbabilityCalibrator | None = None,
        approve_threshold: float = 0.5,
        decline_threshold: float = 0.6,
        feature_columns: list[str] | None = None,
        reference_values: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.calibrator = calibrator
        self.approve_threshold = float(approve_threshold)
        self.decline_threshold = float(decline_threshold)
        cfg = get_config()
        self.feature_columns = list(feature_columns or cfg.columns.features)
        # Training-set medians per feature, used as the ablation reference for
        # local explanations (falls back to empty if unavailable).
        self.reference_values = dict(reference_values or {})
        self.metadata = dict(metadata or {})

    # -- prediction ---------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated default probabilities for raw feature records."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_columns)
        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Input is missing required feature columns: {sorted(missing)}")
        X = X[self.feature_columns]
        if self.calibrator is not None:
            return np.asarray(self.calibrator.predict_proba(X), dtype=float)
        transformed = self.preprocessor.transform(X)
        return np.asarray(self.classifier.predict_proba(transformed))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard labels using the learned approve threshold."""
        return (self.predict_proba(X) >= self.approve_threshold).astype(int)

    def tier(self, probability: float) -> str:
        """Return ``LOW`` / ``MEDIUM`` / ``HIGH`` for a probability."""
        cfg = get_config()
        return cfg.risk.tier(probability, self.approve_threshold)

    def decision(self, probability: float) -> str:
        """Return ``APPROVE`` / ``REVIEW`` / ``DECLINE`` for a probability."""
        return TIER_TO_DECISION[self.tier(probability)]

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame, returning probability, tier, decision, and score."""
        probabilities = self.predict_proba(X)
        return pd.DataFrame(
            {
                "default_probability": probabilities,
                "risk_tier": [self.tier(p) for p in probabilities],
                "decision": [self.decision(p) for p in probabilities],
                "risk_score": [to_risk_score(p) for p in probabilities],
            },
            index=X.index,
        )

    def score_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Score a single record dict and return a JSON-safe payload."""
        frame = pd.DataFrame([record], columns=self.feature_columns)
        row = self.score(frame).iloc[0]
        return {
            "default_probability": round(float(row["default_probability"]), 6),
            "risk_tier": str(row["risk_tier"]),
            "decision": str(row["decision"]),
            "risk_score": int(row["risk_score"]),
            "model_version": self.metadata.get("model_version", "unknown"),
        }

    def explain_record(self, record: dict[str, Any], top_k: int = 8) -> dict[str, list[dict[str, Any]]]:
        """Return ablation-based local risk/protective factors for one record.

        Uses the training medians stored on the artifact as the reference; if
        they are unavailable an empty result is returned rather than a guess.
        """
        from credit_risk.explainability.explain import ablation_contributions

        reference_values = getattr(self, "reference_values", {})
        if not reference_values:
            return {"risk_increasing": [], "protective": []}
        frame = pd.DataFrame([record], columns=self.feature_columns)
        reference = pd.DataFrame([reference_values], columns=self.feature_columns)
        return ablation_contributions(self, frame, reference, top_k=top_k)

    # -- persistence --------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        """Persist the complete inference artifact and return its path."""
        cfg = get_config()
        path = path or cfg.serving.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved inference artifact to %s", path)
        return path


def load_model(path: Path | None = None) -> CreditRiskModel:
    """Load the persisted :class:`CreditRiskModel`.

    Raises:
        FileNotFoundError: If the artifact does not exist yet.
    """
    cfg = get_config()
    path = path or cfg.serving.model_path
    if not path.exists():
        raise FileNotFoundError(
            f"No model artifact at {path}. Train one first: `python -m credit_risk.models.train`."
        )
    model = joblib.load(path)
    # Duck-typed check: the artifact may be unpickled under a different module
    # name (e.g. ``__main__`` when running the module directly), so we verify the
    # required interface rather than the exact class identity.
    required = ("predict_proba", "score_record", "approve_threshold", "feature_columns")
    if not all(hasattr(model, attr) for attr in required):
        raise TypeError(f"Artifact at {path} does not implement the CreditRiskModel interface.")
    return model


def main() -> None:
    """CLI demo: score one plausible low-risk applicant record."""
    import json

    sample = {
        "limit_bal": 120000,
        "age": 33,
        "pay_0": -1, "pay_2": -1, "pay_3": 0, "pay_4": 0, "pay_5": -2, "pay_6": -2,
        "bill_amt1": 5250, "bill_amt2": 4305, "bill_amt3": 4953, "bill_amt4": 4690,
        "bill_amt5": 4000, "bill_amt6": 3972,
        "pay_amt1": 5250, "pay_amt2": 4305, "pay_amt3": 4953, "pay_amt4": 4690,
        "pay_amt5": 4000, "pay_amt6": 3972,
        "education": 2,
    }
    model = load_model()
    print(json.dumps(model.score_record(sample), indent=2))


if __name__ == "__main__":
    main()
