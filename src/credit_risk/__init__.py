"""Credit risk scoring — probability-of-default prediction system.

A production-oriented machine learning package for credit-default risk:

* leakage-safe data ingestion, validation, and preprocessing,
* interpretable and gradient-boosted models with hyper-parameter search,
* probability calibration and business-cost threshold optimisation,
* explainability and responsible-ML auditing,
* a persisted end-to-end inference artifact plus FastAPI/Streamlit surfaces.
"""

from credit_risk.config import Config, get_config

__all__ = ["Config", "get_config", "__version__"]

__version__ = "1.0.0"
