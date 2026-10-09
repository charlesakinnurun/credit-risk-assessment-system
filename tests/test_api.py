"""API contract: health, model-info, prediction schema, and input validation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api as api
from credit_risk.config import get_config


@pytest.fixture()
def client(monkeypatch, tiny_model):
    # Swap the module-level loader for the in-memory test model.
    monkeypatch.setattr(api, "get_model", lambda: tiny_model)
    return TestClient(api.app)


def _valid_payload() -> dict[str, float | int]:
    cfg = get_config()
    payload: dict[str, float | int] = dict.fromkeys(cfg.columns.features, 0)
    payload.update({"limit_bal": 120000, "age": 33, "education": 2, "pay_0": -1, "pay_2": -1})
    return payload


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body
    assert "operating_threshold" in body


def test_predict_returns_contract(client: TestClient) -> None:
    response = client.post("/predict", json=_valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) >= {"default_probability", "risk_tier", "decision", "risk_score", "model_version"}
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["decision"] in {"APPROVE", "REVIEW", "DECLINE"}


def test_predict_rejects_out_of_range(client: TestClient) -> None:
    payload = _valid_payload()
    payload["age"] = 5  # below the allowed minimum
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_request"


def test_predict_rejects_unknown_field(client: TestClient) -> None:
    payload = _valid_payload()
    payload["annual_income"] = 65000  # not part of the dataset schema
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = _valid_payload()
    del payload["limit_bal"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_explain_returns_factors(client: TestClient) -> None:
    response = client.post("/explain", json=_valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert "risk_increasing" in body and "protective" in body
