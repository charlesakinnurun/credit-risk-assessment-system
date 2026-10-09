"""Structured logging helpers used across the package."""

from __future__ import annotations

import logging
import sys

from credit_risk.config import get_config

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root = logging.getLogger("credit_risk")
    root.setLevel(get_config().log_level)
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger that writes structured, timestamped lines.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` under the ``credit_risk`` root.
    """
    _configure_root()
    if not name.startswith("credit_risk"):
        name = f"credit_risk.{name}"
    return logging.getLogger(name)
