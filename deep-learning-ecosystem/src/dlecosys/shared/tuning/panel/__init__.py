"""
Panel Tuning Package
--------------------
Optuna-driven hyperparameter search for panel forecasting pipelines with
time-series cross-validation (no row shuffling). Composes a per-trial
objective that re-fits preprocessing + vectorizes + trains per fold,
aggregates fold validation losses, and emits ``best_config.yaml`` with the
winning hyperparameters applied and ``tuning: null``.

Functions
---------
Re-exported from submodules for convenience.
"""

from __future__ import annotations

from dlecosys.shared.tuning.panel.best_config import emit_best_config
from dlecosys.shared.tuning.panel.objective import PanelObjective
from dlecosys.shared.tuning.panel.run import run_panel_tuning
from dlecosys.shared.tuning.panel.splitter_builder import build_panel_splitter
from dlecosys.shared.tuning.panel.study import build_panel_study

__all__ = [
    "PanelObjective",
    "build_panel_splitter",
    "build_panel_study",
    "emit_best_config",
    "run_panel_tuning",
]
