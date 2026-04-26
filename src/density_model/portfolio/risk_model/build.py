"""
Covariance Estimator Dispatch
-----------------------------
Construct a concrete :class:`BaseCovarianceEstimator` from a
:class:`PanelRiskModelConfig`.

Functions
---------
build_covariance_estimator
    Dispatch by ``cfg.type``.
"""

from __future__ import annotations

from density_model.portfolio.config import PanelRiskModelConfig
from density_model.portfolio.risk_model.base import BaseCovarianceEstimator
from density_model.portfolio.risk_model.rolling import RollingCovariance

__all__ = ["build_covariance_estimator"]


def build_covariance_estimator(cfg: PanelRiskModelConfig) -> BaseCovarianceEstimator:
    if cfg.type == "rolling":
        return RollingCovariance(
            window_size=cfg.window_size,
            ridge=cfg.ridge,
            ridge_mode=cfg.ridge_mode,
        )
    raise ValueError(f"Unsupported risk_model type {cfg.type!r}.")
