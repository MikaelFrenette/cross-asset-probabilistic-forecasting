"""
Evaluation Package
------------------
Plotting and evaluation helpers for panel forecasting outputs.

Functions
---------
generate_panel_plots
    Per-asset three-panel forecast report (conditional mean, conditional
    volatility, risk-adjusted signal).
"""

from __future__ import annotations

from density_model.shared.evaluation.panel_plot import generate_panel_plots

__all__ = ["generate_panel_plots"]
