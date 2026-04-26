"""
Covariance Estimators
---------------------
Strictly-no-lookahead covariance estimators for the portfolio pipeline.

Classes
-------
BaseCovarianceEstimator
    Abstract ``estimate(history, as_of) -> Σ`` contract.
RollingCovariance
    Sample covariance over the last ``window_size`` dates ≤ ``as_of``
    with a small diagonal ridge for numerical stability.

Functions
---------
build_covariance_estimator
    Dispatch from a ``PanelRiskModelConfig`` to a concrete estimator.
"""

from __future__ import annotations

from density_model.portfolio.risk_model.base import BaseCovarianceEstimator
from density_model.portfolio.risk_model.build import build_covariance_estimator
from density_model.portfolio.risk_model.rolling import RollingCovariance

__all__ = [
    "BaseCovarianceEstimator",
    "RollingCovariance",
    "build_covariance_estimator",
]
