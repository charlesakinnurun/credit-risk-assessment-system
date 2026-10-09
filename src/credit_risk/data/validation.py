"""Schema and data-quality validation.

Validation is deliberately explicit and cheap to run in CI. It answers a narrow
set of questions:

* does the frame contain the exact columns the pipeline requires?
* are there missing values, duplicate rows, or duplicate client identifiers?
* are categorical codes inside the published codebook?
* is the row count and target prevalence plausible for this dataset?

Any structural failure raises :class:`DataValidationError` so a corrupt batch
fails loudly instead of silently training a degraded model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)

# Published codebook values for the categorical columns.
VALID_SEX_CODES: tuple[int, ...] = (1, 2)
VALID_EDUCATION_CODES: tuple[int, ...] = (1, 2, 3, 4)
VALID_MARRIAGE_CODES: tuple[int, ...] = (1, 2, 3)


class DataValidationError(ValueError):
    """Raised when a dataset violates a structural invariant."""


@dataclass
class DataQualityReport:
    """Summary of structural data-quality checks."""

    n_rows: int
    n_columns: int
    n_features: int
    n_missing: int
    missing_by_column: dict[str, int]
    n_duplicate_rows: int
    n_duplicate_ids: int
    n_defaults: int
    default_rate: float
    out_of_codebook: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_features": self.n_features,
            "n_missing": self.n_missing,
            "missing_by_column": self.missing_by_column,
            "n_duplicate_rows": self.n_duplicate_rows,
            "n_duplicate_ids": self.n_duplicate_ids,
            "n_defaults": self.n_defaults,
            "default_rate": round(self.default_rate, 6),
            "out_of_codebook": self.out_of_codebook,
        }


def require_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    """Raise :class:`DataValidationError` if any required column is absent."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise DataValidationError(f"DataFrame is missing required columns: {sorted(missing)}")


def validate_data(df: pd.DataFrame, strict_prevalence: bool = True) -> DataQualityReport:
    """Run structural data-quality checks and return a summary report.

    Args:
        df: Raw (renamed) credit-card DataFrame.
        strict_prevalence: If ``True`` (training), raise when the default rate
            falls outside the plausible band configured for this dataset.

    Returns:
        A :class:`DataQualityReport`.

    Raises:
        DataValidationError: On a structural violation.
    """
    cfg = get_config()
    required = [cfg.columns.id, *cfg.columns.features, *cfg.columns.sensitive, cfg.columns.target]
    require_columns(df, required)

    target = df[cfg.columns.target]
    if not set(target.dropna().unique()).issubset({0, 1}):
        raise DataValidationError("Target column must be binary {0, 1}.")

    missing = df.isna().sum()
    out_of_codebook: dict[str, dict[str, int]] = {}
    for column, valid_codes in (
        ("sex", VALID_SEX_CODES),
        ("education", VALID_EDUCATION_CODES),
        ("marriage", VALID_MARRIAGE_CODES),
    ):
        if column not in df.columns:
            continue
        counts = df[column].value_counts()
        unexpected = counts[~counts.index.isin(valid_codes)]
        if not unexpected.empty:
            out_of_codebook[column] = {str(k): int(v) for k, v in unexpected.items()}

    default_rate = float(target.mean())
    if strict_prevalence and not (cfg.data.min_default_rate <= default_rate <= cfg.data.max_default_rate):
        raise DataValidationError(
            f"Default rate {default_rate:.3f} is outside the plausible range "
            f"[{cfg.data.min_default_rate}, {cfg.data.max_default_rate}]."
        )

    report = DataQualityReport(
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        n_features=len(cfg.columns.features),
        n_missing=int(missing.sum()),
        missing_by_column={c: int(v) for c, v in missing.items() if v > 0},
        n_duplicate_rows=int(df.duplicated().sum()),
        n_duplicate_ids=int(df[cfg.columns.id].duplicated().sum()),
        n_defaults=int(target.sum()),
        default_rate=default_rate,
        out_of_codebook=out_of_codebook,
    )
    return report


def validate_inference_frame(df: pd.DataFrame) -> None:
    """Validate a batch of raw records destined for inference.

    Unlike training validation this does not require the target or identifier,
    does not enforce prevalence, and tolerates out-of-codebook categorical codes
    (they are mapped to ``"other"`` downstream).
    """
    cfg = get_config()
    require_columns(df, list(cfg.columns.features))
    if df.empty:
        raise DataValidationError("Inference frame is empty.")
    if df[list(cfg.columns.features)].isna().all(axis=1).any():
        raise DataValidationError("Inference frame contains rows with all features missing.")
