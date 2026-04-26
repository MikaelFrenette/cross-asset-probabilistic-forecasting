"""
Portfolio Construction
----------------------
Takes density-model outputs (μ, σ per asset per forecast date) plus
realized returns and produces dynamic portfolio weights via a
mean-variance optimizer. Full-investment constraint (∑w = 1); shorting
is allowed in the baseline (toggleable via ``long_only: true``).

Pipeline
--------
At each forecast date t+1:

    1. α_raw = μ / σ  (cross-section at t+1)
    2. α     = zscore(α_raw)       across assets at t+1
    3. Σ     = rolling covariance  using realized returns strictly
                                   before t+1 (window_size knob)
    4. w     = argmax_w α'w − (λ/2) w'Σw   subject to 1'w = 1
    5. return = w · r_{t+1}        realized portfolio return

Every component (alpha builder, covariance estimator, optimizer) has an
abstract base + dispatch builder so adding a new variant (EWMA covariance,
long-only optimizer, turnover penalty, etc.) is a 3-line branch.

Functions
---------
Re-exported from submodules for convenience.
"""

from __future__ import annotations

from density_model.portfolio.alpha import (
    BaseAlphaBuilder,
    MuSigmaZScoreAlpha,
    build_alpha_builder,
)
from density_model.portfolio.backtest import BacktestOutput, run_backtest
from density_model.portfolio.config import (
    PanelPortfolioConfig,
    load_panel_portfolio_config,
)
from density_model.portfolio.metrics import compute_portfolio_metrics
from density_model.portfolio.optimizer import (
    BaseOptimizer,
    MeanVarianceOptimizer,
    build_optimizer,
)
from density_model.portfolio.plotting import plot_portfolio_vs_benchmarks
from density_model.portfolio.risk_model import (
    BaseCovarianceEstimator,
    RollingCovariance,
    build_covariance_estimator,
)
from density_model.portfolio.run import run_portfolio_construction

__all__ = [
    "BacktestOutput",
    "BaseAlphaBuilder",
    "BaseCovarianceEstimator",
    "BaseOptimizer",
    "MeanVarianceOptimizer",
    "MuSigmaZScoreAlpha",
    "PanelPortfolioConfig",
    "RollingCovariance",
    "build_alpha_builder",
    "build_covariance_estimator",
    "build_optimizer",
    "compute_portfolio_metrics",
    "load_panel_portfolio_config",
    "plot_portfolio_vs_benchmarks",
    "run_backtest",
    "run_portfolio_construction",
]
