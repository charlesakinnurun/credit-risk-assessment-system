"""Central, typed configuration for the credit-risk system.

Configuration lives in ``configs/config.yaml`` and is loaded once into a set of
frozen dataclasses. Environment variables may override the config path, the
random seed, the model path, and the log level, which keeps deployment-specific
values out of source code::

    CREDIT_RISK_CONFIG   path to an alternative YAML config
    CREDIT_RISK_SEED     integer random seed
    CREDIT_RISK_MODEL_PATH  path to the persisted inference artifact
    CREDIT_RISK_LOG_LEVEL   DEBUG | INFO | WARNING | ERROR

No secrets are stored here. See ``.env.example`` for serving-time variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# ``configs/`` sits at the repository root; ``src/credit_risk/config.py`` is two
# levels below it.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass(frozen=True)
class DataConfig:
    raw_path: Path
    raw_url: str
    source_url: str
    expected_rows: int
    expected_columns: int
    min_default_rate: float
    max_default_rate: float


@dataclass(frozen=True)
class ColumnConfig:
    id: str
    target: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    sensitive: tuple[str, ...]

    @property
    def features(self) -> tuple[str, ...]:
        """Model input columns, in a stable order."""
        return self.numeric + self.categorical


@dataclass(frozen=True)
class EngineeringConfig:
    pay_columns: tuple[str, ...]
    bill_columns: tuple[str, ...]
    payment_columns: tuple[str, ...]
    limit_column: str


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class TrainingConfig:
    cv_folds: int
    search_iterations: dict[str, int]
    scoring: str
    model_selection_tolerance: float
    class_weight_options: tuple[Any, ...]
    permutation_repeats: int
    n_jobs_search: int


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int
    max_iter: int
    n_jobs: int


@dataclass(frozen=True)
class ThresholdConfig:
    min: float
    max: float
    step: float
    cost_false_negative: float
    cost_false_positive: float


@dataclass(frozen=True)
class RiskConfig:
    """Decision policy applied to calibrated default probabilities."""

    decline_threshold: float

    def tier(self, probability: float, approve_threshold: float) -> str:
        """Map a calibrated probability to a risk tier.

        The LOW/MEDIUM boundary is the learned approve threshold so the tier is
        always consistent with the decision; the MEDIUM/HIGH boundary is the
        policy decline threshold.
        """
        if probability < approve_threshold:
            return "LOW"
        if probability < self.decline_threshold:
            return "MEDIUM"
        return "HIGH"

    def decision(self, probability: float, approve_threshold: float) -> str:
        """Map a calibrated probability to a lending decision."""
        tier = self.tier(probability, approve_threshold)
        return {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "DECLINE"}[tier]


@dataclass(frozen=True)
class ServingConfig:
    model_path: Path
    metadata_path: Path
    host: str
    api_port: int
    dashboard_port: int


@dataclass(frozen=True)
class Config:
    project_name: str
    version: str
    seed: int
    data: DataConfig
    columns: ColumnConfig
    engineering: EngineeringConfig
    split: SplitConfig
    training: TrainingConfig
    model: ModelConfig
    calibration_methods: tuple[str, ...]
    threshold: ThresholdConfig
    risk: RiskConfig
    serving: ServingConfig
    log_level: str
    processed_dir: Path
    figures_dir: Path
    reports_dir: Path
    experiments_dir: Path
    project_root: Path = field(default=PROJECT_ROOT)


def _resolve(path_like: str, root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_config(raw: dict[str, Any]) -> Config:
    data = raw["data"]
    columns = raw["columns"]
    engineering = raw["engineering"]
    split = raw["split"]
    training = raw["training"]
    model = raw["model"]
    threshold = raw["threshold"]
    risk = raw["risk"]
    serving = raw["serving"]
    paths = raw["paths"]

    log_level = os.environ.get("CREDIT_RISK_LOG_LEVEL", raw.get("logging", {}).get("level", "INFO"))
    if log_level not in VALID_LOG_LEVELS:
        raise ValueError(f"Invalid log level {log_level!r}; expected one of {sorted(VALID_LOG_LEVELS)}")

    model_path = os.environ.get("CREDIT_RISK_MODEL_PATH", serving["model_path"])

    return Config(
        project_name=raw["project"]["name"],
        version=str(raw["project"]["version"]),
        seed=int(os.environ.get("CREDIT_RISK_SEED", raw.get("seed", 42))),
        data=DataConfig(
            raw_path=_resolve(data["raw_path"]),
            raw_url=data["raw_url"],
            source_url=data["source_url"],
            expected_rows=int(data["expected_rows"]),
            expected_columns=int(data["expected_columns"]),
            min_default_rate=float(data["min_default_rate"]),
            max_default_rate=float(data["max_default_rate"]),
        ),
        columns=ColumnConfig(
            id=columns["id"],
            target=columns["target"],
            numeric=tuple(columns["numeric"]),
            categorical=tuple(columns["categorical"]),
            sensitive=tuple(columns["sensitive"]),
        ),
        engineering=EngineeringConfig(
            pay_columns=tuple(engineering["pay_columns"]),
            bill_columns=tuple(engineering["bill_columns"]),
            payment_columns=tuple(engineering["payment_columns"]),
            limit_column=engineering["limit_column"],
        ),
        split=SplitConfig(
            train=float(split["train"]),
            validation=float(split["validation"]),
            test=float(split["test"]),
        ),
        training=TrainingConfig(
            cv_folds=int(training["cv_folds"]),
            search_iterations=dict(training["search_iterations"]),
            scoring=training["scoring"],
            model_selection_tolerance=float(training["model_selection_tolerance"]),
            class_weight_options=tuple(training["class_weight_options"]),
            permutation_repeats=int(training.get("permutation_repeats", 5)),
            n_jobs_search=int(training.get("n_jobs_search", 1)),
        ),
        model=ModelConfig(
            n_estimators=int(model["n_estimators"]),
            max_iter=int(model["max_iter"]),
            n_jobs=int(model["n_jobs"]),
        ),
        calibration_methods=tuple(raw["calibration"]["methods"]),
        threshold=ThresholdConfig(
            min=float(threshold["min"]),
            max=float(threshold["max"]),
            step=float(threshold["step"]),
            cost_false_negative=float(threshold["cost_false_negative"]),
            cost_false_positive=float(threshold["cost_false_positive"]),
        ),
        risk=RiskConfig(decline_threshold=float(risk["decline_threshold"])),
        serving=ServingConfig(
            model_path=_resolve(model_path),
            metadata_path=_resolve(serving["metadata_path"]),
            host=serving["host"],
            api_port=int(serving["api_port"]),
            dashboard_port=int(serving["dashboard_port"]),
        ),
        log_level=log_level,
        processed_dir=_resolve(paths["processed_dir"]),
        figures_dir=_resolve(paths["figures_dir"]),
        reports_dir=_resolve(paths["reports_dir"]),
        experiments_dir=_resolve(paths["experiments_dir"]),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Load (and cache) the project configuration."""
    config_path = Path(os.environ.get("CREDIT_RISK_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_absolute() and not config_path.exists():
        config_path = PROJECT_ROOT / config_path
    return _build_config(_load_yaml(config_path))


def reload_config() -> Config:
    """Clear the cache and reload configuration (useful in tests)."""
    get_config.cache_clear()
    return get_config()


def ensure_directories() -> None:
    """Create the runtime directories the pipeline writes to."""
    cfg = get_config()
    for directory in (
        cfg.data.raw_path.parent,
        cfg.processed_dir,
        cfg.serving.model_path.parent,
        cfg.figures_dir,
        cfg.reports_dir,
        cfg.experiments_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
