"""Responsible-ML / fairness auditing that never feeds sensitive data to a model."""

from credit_risk.responsible.fairness import audit, disparity_summary

__all__ = ["audit", "disparity_summary"]
