"""
Ensemble Factory
----------------
Construct a BaseEnsemble from an EnsembleSection config.

Functions
---------
build_ensemble
    Dispatch on ``cfg.type`` to the concrete ensemble class; build the
    composed sample + feature bootstrappers from their sub-sections.
"""

from __future__ import annotations

from dlecosys.shared.config.schema import EnsembleSection
from dlecosys.shared.ensembling.bagging import BaggingEnsemble
from dlecosys.shared.ensembling.base import BaseEnsemble
from dlecosys.shared.ensembling.bootstrappers import (
    build_feature_bootstrapper,
    build_sample_bootstrapper,
)

__all__ = ["build_ensemble"]


def build_ensemble(cfg: EnsembleSection) -> BaseEnsemble:
    if cfg.type == "bagging":
        return BaggingEnsemble(
            n_estimators=cfg.n_estimators,
            sample_bootstrapper=build_sample_bootstrapper(cfg.sample_bootstrapper),
            feature_bootstrapper=build_feature_bootstrapper(cfg.feature_bootstrapper),
            seed=cfg.seed,
        )
    raise ValueError(f"Unknown ensemble type: {cfg.type!r}")
