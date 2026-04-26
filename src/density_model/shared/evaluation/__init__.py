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
"""

from __future__ import annotations

from density_model.shared.evaluation.panel_metrics import generate_panel_metrics
from density_model.shared.evaluation.panel_plot import generate_panel_plots

__all__ = ["generate_panel_metrics", "generate_panel_plots"]
