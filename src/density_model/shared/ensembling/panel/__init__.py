"""
Panel Ensembling Package
------------------------
Bagging ensemble for panel forecasting with per-estimator preprocessing,
forecast-date bootstrapping, out-of-bag-as-val training, asset-subspace
bootstrapping, top-N pruning, and Gaussian-aware aggregation (mean mu,
law-of-total-variance sigma).

Functions
---------
Re-exported from submodules for convenience.
"""

from __future__ import annotations

from density_model.shared.ensembling.panel.aggregation import aggregate_gaussian_predictions
from density_model.shared.ensembling.panel.bootstrappers import (
    AssetSubspaceBootstrapper,
    BaseFeatureBootstrapper,
    BaseSampleBootstrapper,
    DateBootstrapper,
    NoFeatureBootstrapper,
    NoSampleBootstrapper,
)
from density_model.shared.ensembling.panel.display import (
    clear_console,
    print_estimator_result,
    render_header,
    render_leaderboard,
)
from density_model.shared.ensembling.panel.manifest import (
    PanelEnsembleEstimatorEntry,
    PanelEnsembleManifest,
    load_panel_ensemble_manifest,
)
from density_model.shared.ensembling.panel.pruning import prune_top_n
from density_model.shared.ensembling.panel.run import run_panel_ensemble

__all__ = [
    "AssetSubspaceBootstrapper",
    "BaseFeatureBootstrapper",
    "BaseSampleBootstrapper",
    "DateBootstrapper",
    "NoFeatureBootstrapper",
    "NoSampleBootstrapper",
    "PanelEnsembleEstimatorEntry",
    "PanelEnsembleManifest",
    "aggregate_gaussian_predictions",
    "clear_console",
    "load_panel_ensemble_manifest",
    "print_estimator_result",
    "prune_top_n",
    "render_header",
    "render_leaderboard",
    "run_panel_ensemble",
]
