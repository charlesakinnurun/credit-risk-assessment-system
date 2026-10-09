"""Lightweight, dependency-free experiment tracking.

Each training run is written as a self-contained JSON record (parameters,
metrics, dataset fingerprint, feature configuration, artifact paths, git commit)
under ``reports/experiments/`` and appended to a flat ``runs.jsonl`` index. This
makes the best run reproducibly identifiable without requiring a tracking
server.

If the optional ``mlflow`` package is installed and ``CREDIT_RISK_MLFLOW=1`` is
set, the same payload is additionally logged to MLflow.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)


def get_git_commit() -> str:
    """Return the current git commit SHA, or ``"unknown"`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return "unknown"


@dataclass
class RunRecord:
    """A single reproducible training run."""

    run_id: str
    created_at: str
    git_commit: str
    model_name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    dataset_fingerprint: str = ""
    feature_config: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    """Log runs to a local JSON store, optionally mirrored to MLflow."""

    def __init__(self, run_name: str) -> None:
        cfg = get_config()
        self.cfg = cfg
        self.run_name = run_name
        self.run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
        self.cfg.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._mlflow = self._init_mlflow()

    def _init_mlflow(self):
        if os.environ.get("CREDIT_RISK_MLFLOW", "0") != "1":
            return None
        try:
            import mlflow  # noqa: PLC0415 - optional dependency

            mlflow.set_experiment(self.cfg.project_name)
            mlflow.start_run(run_name=self.run_name)
            return mlflow
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("MLflow requested but unavailable: %s", exc)
            return None

    def log(
        self,
        *,
        model_name: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        dataset_fingerprint: str,
        feature_config: dict[str, Any],
        artifacts: dict[str, str] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> RunRecord:
        """Persist one run and return its record."""
        record = RunRecord(
            run_id=self.run_id,
            created_at=datetime.now(UTC).isoformat(),
            git_commit=get_git_commit(),
            model_name=model_name,
            params=params,
            metrics=metrics,
            dataset_fingerprint=dataset_fingerprint,
            feature_config=feature_config,
            artifacts=artifacts or {},
            tags=tags or {},
        )
        record_path = self.cfg.experiments_dir / f"{self.run_id}.json"
        record_path.write_text(json.dumps(record.as_dict(), indent=2, default=str), encoding="utf-8")
        with (self.cfg.experiments_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.as_dict(), default=str) + "\n")
        logger.info("Logged experiment run %s -> %s", self.run_id, record_path)

        if self._mlflow is not None:  # pragma: no cover - optional
            flat_params = {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in params.items()}
            self._mlflow.log_params(flat_params)
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            self._mlflow.set_tags({"model_name": model_name, **(tags or {})})
            self._mlflow.end_run()
        return record
