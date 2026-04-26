"""
Portfolio Optimizers
--------------------
Solve for portfolio weights from ``(α, Σ, λ)``.

Classes
-------
BaseOptimizer
    Abstract ``optimize(alpha, covariance) -> weights`` contract.
MeanVarianceOptimizer
    ``max α'w − (λ/2) w'Σw`` subject to ``1'w = 1`` (optionally ``w ≥ 0``).
    Uses the closed-form Lagrangian solution for the unconstrained-
    sign case; falls back to SLSQP (scipy) when ``long_only: true``.

Functions
---------
build_optimizer
    Dispatch from a ``PanelOptimizerConfig`` to a concrete optimizer.
"""

from __future__ import annotations

from density_model.portfolio.optimizer.base import BaseOptimizer
from density_model.portfolio.optimizer.build import build_optimizer
from density_model.portfolio.optimizer.mean_variance import MeanVarianceOptimizer

__all__ = ["BaseOptimizer", "MeanVarianceOptimizer", "build_optimizer"]
