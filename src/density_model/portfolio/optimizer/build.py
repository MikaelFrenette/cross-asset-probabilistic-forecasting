"""
Optimizer Dispatch
------------------
Construct a concrete :class:`BaseOptimizer` from a
:class:`PanelOptimizerConfig`.

Functions
---------
build_optimizer
    Dispatch by ``cfg.type``.
"""

from __future__ import annotations

from density_model.portfolio.config import PanelOptimizerConfig
from density_model.portfolio.optimizer.base import BaseOptimizer
from density_model.portfolio.optimizer.mean_variance import MeanVarianceOptimizer

__all__ = ["build_optimizer"]


def build_optimizer(cfg: PanelOptimizerConfig) -> BaseOptimizer:
    if cfg.type == "mean_variance":
        return MeanVarianceOptimizer(
            risk_aversion=cfg.risk_aversion,
            long_only=cfg.long_only,
            full_investment=cfg.full_investment,
        )
    raise ValueError(f"Unsupported optimizer type {cfg.type!r}.")
