"""Modeling subpackage: training, evaluation, calibration, and inference.

Modules are imported explicitly by their full path (``credit_risk.models.train``
etc.). This package intentionally does not re-export the inference classes, so
that running a submodule with ``python -m credit_risk.models.predict`` does not
import the module twice (once as a package member and once as ``__main__``).
"""
