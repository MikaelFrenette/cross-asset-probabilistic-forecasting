"""
Evaluation Package
------------------
Plotting and evaluation helpers for panel forecasting outputs.

Functions
---------
generate_panel_plots
    Per-asset three-panel forecast report (conditional mean, conditional
    volatility, risk-adjusted signal).

generate_panel_metrics
    Density-forecast metrics table (NLL, CRPS, MAE, bias, calibration
    coverage, mean sigma) at overall and per-asset level; writes
    metrics.json + metrics_per_asset.csv + metrics.md alongside the
    predictions CSV.

generate_calibration_plot
    Two-panel σ calibration showcase: conditional volatility vs rolling
    realized vol for a focus asset, plus a panel-wide reliability
    diagram (empirical vs nominal coverage of standardized residuals).

generate_signal_plot
    Two-panel cross-sectional signal showcase: pooled per-bucket
    annualized realized return + cumulative top-minus-bottom long/short
    spread vs an optional benchmark.
"""

from __future__ import annotations

from density_model.shared.evaluation.calibration_plot import generate_calibration_plot
from density_model.shared.evaluation.panel_metrics import generate_panel_metrics
from density_model.shared.evaluation.panel_plot import generate_panel_plots
from density_model.shared.evaluation.signal_plot import generate_signal_plot

__all__ = [
    "generate_calibration_plot",
    "generate_panel_metrics",
    "generate_panel_plots",
    "generate_signal_plot",
]
