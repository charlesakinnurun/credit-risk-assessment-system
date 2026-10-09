"""FastAPI inference service for the credit-risk model.

Endpoints
---------
``GET  /health``      liveness + whether the model artifact is loaded
``GET  /model-info``  model version, dataset fingerprint, threshold, calibration
``POST /predict``     calibrated default probability, risk tier, and decision
``POST /explain``     ablation-based local risk / protective factors

Run locally::

    uvicorn app.api:app --reload --port 8000

The request schema uses the **actual** features of the UCI *Default of Credit
Card Clients* dataset. Generic credit fields such as ``annual_income`` or
``credit_score`` do **not** exist in this dataset and are therefore not accepted
(fabricating them would silently produce meaningless predictions).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from credit_risk.config import get_config
from credit_risk.models.predict import CreditRiskModel, load_model
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)


class CreditApplication(BaseModel):
    """A single credit-card applicant as observed over the six-month window."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "limit_bal": 120000,
                "age": 33,
                "pay_0": -1, "pay_2": -1, "pay_3": 0, "pay_4": 0, "pay_5": -2, "pay_6": -2,
                "bill_amt1": 5250, "bill_amt2": 4305, "bill_amt3": 4953,
                "bill_amt4": 4690, "bill_amt5": 4000, "bill_amt6": 3972,
                "pay_amt1": 5250, "pay_amt2": 4305, "pay_amt3": 4953,
                "pay_amt4": 4690, "pay_amt5": 4000, "pay_amt6": 3972,
                "education": 2,
            }
        },
    )

    limit_bal: float = Field(..., ge=0, le=1_000_000, description="Credit limit (NT$)")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    pay_0: int = Field(..., ge=-2, le=9, description="Repayment status, most recent month (-2..9)")
    pay_2: int = Field(..., ge=-2, le=9, description="Repayment status, 2 months ago")
    pay_3: int = Field(..., ge=-2, le=9, description="Repayment status, 3 months ago")
    pay_4: int = Field(..., ge=-2, le=9, description="Repayment status, 4 months ago")
    pay_5: int = Field(..., ge=-2, le=9, description="Repayment status, 5 months ago")
    pay_6: int = Field(..., ge=-2, le=9, description="Repayment status, 6 months ago")
    bill_amt1: float = Field(..., description="Statement balance, month 1 (NT$)")
    bill_amt2: float = Field(..., description="Statement balance, month 2 (NT$)")
    bill_amt3: float = Field(..., description="Statement balance, month 3 (NT$)")
    bill_amt4: float = Field(..., description="Statement balance, month 4 (NT$)")
    bill_amt5: float = Field(..., description="Statement balance, month 5 (NT$)")
    bill_amt6: float = Field(..., description="Statement balance, month 6 (NT$)")
    pay_amt1: float = Field(..., ge=0, description="Amount repaid, month 1 (NT$)")
    pay_amt2: float = Field(..., ge=0, description="Amount repaid, month 2 (NT$)")
    pay_amt3: float = Field(..., ge=0, description="Amount repaid, month 3 (NT$)")
    pay_amt4: float = Field(..., ge=0, description="Amount repaid, month 4 (NT$)")
    pay_amt5: float = Field(..., ge=0, description="Amount repaid, month 5 (NT$)")
    pay_amt6: float = Field(..., ge=0, description="Amount repaid, month 6 (NT$)")
    education: int = Field(..., ge=1, le=6, description="Education code (1=graduate, 2=university, 3=high school, 4+=other)")

    def as_record(self) -> dict[str, float | int]:
        return self.model_dump()


class PredictionResponse(BaseModel):
    """Scoring result."""

    default_probability: float = Field(..., ge=0, le=1)
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    risk_score: int = Field(..., ge=0, le=1000)
    model_version: str


class Factor(BaseModel):
    feature: str
    contribution: float


class ExplanationResponse(BaseModel):
    default_probability: float
    risk_tier: str
    decision: str
    risk_increasing: list[Factor]
    protective: list[Factor]


@lru_cache(maxsize=1)
def get_model() -> CreditRiskModel:
    """Load and cache the inference artifact (one load per process)."""
    return load_model()


def _clear_model_cache() -> None:
    """Clear the cached model (used in tests when swapping artifacts)."""
    get_model.cache_clear()


app = FastAPI(
    title="Credit Risk Scoring API",
    version=get_config().version,
    description="Probability-of-default scoring with calibration, explainability, and a human-review decision policy.",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a structured 422 payload instead of leaking the raw schema."""
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_request", "details": exc.errors()},
    )


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe reporting whether the model artifact is available."""
    try:
        model = get_model()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": model.metadata.get("model_version", "unknown"),
        }
    except FileNotFoundError as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Return model metadata: version, dataset, threshold, calibration, metrics."""
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    meta = model.metadata
    return {
        "model_version": meta.get("model_version"),
        "model_name": meta.get("model_name"),
        "trained_at": meta.get("trained_at"),
        "git_commit": meta.get("git_commit"),
        "dataset_fingerprint": meta.get("dataset", {}).get("fingerprint"),
        "feature_columns": meta.get("feature_columns"),
        "sensitive_columns_excluded": meta.get("sensitive_columns_excluded"),
        "operating_threshold": model.approve_threshold,
        "decline_threshold": model.decline_threshold,
        "calibration": meta.get("calibration"),
        "test_metrics": meta.get("test_metrics"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    """Score one applicant and return probability, tier, and decision."""
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result = model.score_record(application.as_record())
    logger.info(
        "predict: probability=%.4f tier=%s decision=%s",
        result["default_probability"], result["risk_tier"], result["decision"],
    )
    return PredictionResponse(**result)


@app.post("/explain", response_model=ExplanationResponse)
def explain(application: CreditApplication) -> ExplanationResponse:
    """Explain one applicant's score via ablation against training medians."""
    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    record = application.as_record()
    base = model.score_record(record)
    factors = model.explain_record(record)
    return ExplanationResponse(
        default_probability=base["default_probability"],
        risk_tier=base["risk_tier"],
        decision=base["decision"],
        risk_increasing=[Factor(**f) for f in factors["risk_increasing"]],
        protective=[Factor(**f) for f in factors["protective"]],
    )


def main() -> None:
    """Entry point for ``python -m app.api`` (development server)."""
    import uvicorn

    cfg = get_config()
    uvicorn.run("app.api:app", host=cfg.serving.host, port=cfg.serving.api_port, reload=False)


if __name__ == "__main__":
    main()
