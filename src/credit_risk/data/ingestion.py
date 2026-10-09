"""Data ingestion for the UCI *Default of Credit Card Clients* dataset.

The raw spreadsheet ships with a human-readable banner row followed by the real
column names, so we read with ``header=1`` and rename the columns to clean
snake_case names. Ingestion never mutates modelling inputs beyond renaming;
validation and cleaning are separate, explicit steps.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests

from credit_risk.config import get_config
from credit_risk.utils.logging import get_logger

logger = get_logger(__name__)

# Raw spreadsheet columns -> canonical project column names.
COLUMN_RENAME: dict[str, str] = {
    "ID": "id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_0",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "bill_amt1",
    "BILL_AMT2": "bill_amt2",
    "BILL_AMT3": "bill_amt3",
    "BILL_AMT4": "bill_amt4",
    "BILL_AMT5": "bill_amt5",
    "BILL_AMT6": "bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default payment next month": "default_payment_next_month",
}


def _ordered_columns() -> list[str]:
    cfg = get_config()
    return [cfg.columns.id, *cfg.columns.numeric, *cfg.columns.categorical,
            *cfg.columns.sensitive, cfg.columns.target]


def download_raw_data(destination: Path | None = None, url: str | None = None) -> Path:
    """Download the raw dataset if it does not already exist locally.

    Args:
        destination: Local path to write the spreadsheet to. Defaults to config.
        url: Remote source. Defaults to config.

    Returns:
        Path to the (existing or newly downloaded) file.

    Raises:
        RuntimeError: If the download fails.
    """
    cfg = get_config()
    destination = destination or cfg.data.raw_path
    url = url or cfg.data.raw_url

    if destination.exists():
        logger.info("Raw data already present at %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading raw dataset from %s", url)
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:  # network/HTTP failure
        raise RuntimeError(
            f"Failed to download the dataset from {url}. "
            "Place cc_default.xls in data/raw/ manually and retry."
        ) from exc
    destination.write_bytes(response.content)
    logger.info("Saved raw dataset to %s (%d bytes)", destination, len(response.content))
    return destination


def load_raw_data(path: Path | None = None, download: bool = True) -> pd.DataFrame:
    """Load and rename the raw credit-card dataset.

    Args:
        path: Path to the raw ``.xls`` spreadsheet. Defaults to config.
        download: Download the file first if it is missing.

    Returns:
        DataFrame with clean snake_case column names, the ``id`` column retained
        for duplicate detection, and the target as the final column.

    Raises:
        FileNotFoundError: If the file is absent and ``download`` is ``False``.
    """
    cfg = get_config()
    path = path or cfg.data.raw_path
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Raw data not found at {path}")
        download_raw_data(path, cfg.data.raw_url)

    df = pd.read_excel(path, header=1)
    df = df.rename(columns=COLUMN_RENAME)
    present = [c for c in _ordered_columns() if c in df.columns]
    return df[present]


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Return a stable content hash that versions the dataset.

    The fingerprint is the SHA-256 of the column names and the target column's
    values, which is sufficient to detect a changed dataset without hashing the
    entire (sensitive) feature matrix.
    """
    hasher = hashlib.sha256()
    hasher.update("|".join(map(str, df.columns)).encode("utf-8"))
    target = get_config().columns.target
    hasher.update(pd.util.hash_pandas_object(df[target], index=False).values.tobytes())
    hasher.update(str(len(df)).encode("utf-8"))
    return hasher.hexdigest()[:16]


def main() -> None:
    """CLI entry point: download (if needed) and preview the raw dataset."""
    df = load_raw_data()
    cfg = get_config()
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"Target rate: {df[cfg.columns.target].mean():.1%}")
    print(f"Fingerprint: {dataset_fingerprint(df)}")


if __name__ == "__main__":
    main()
