"""
Study Builders
--------------
Construct Optuna samplers, pruners, and studies from a TuningSection.

Functions
---------
build_sampler
    Construct a GridSampler or RandomSampler.
build_pruner
    Construct a MedianPruner or NopPruner.
build_study
    Construct an Optuna study with the configured sampler, pruner, and storage.
"""

from __future__ import annotations

from typing import Optional

import optuna
from optuna.pruners import BasePruner, MedianPruner, NopPruner
from optuna.samplers import BaseSampler, GridSampler, RandomSampler

from dlecosys.shared.config.schema import TuningSection
from dlecosys.shared.tuning.search_space import to_hashable

__all__ = ["build_sampler", "build_pruner", "build_study"]


def build_sampler(cfg: TuningSection) -> BaseSampler:
    if cfg.sampler == "grid":
        normalized_space = {
            path: [to_hashable(c) for c in choices]
            for path, choices in cfg.search_space.items()
        }
        return GridSampler(normalized_space)
    if cfg.sampler == "random":
        return RandomSampler()
    raise ValueError(f"Unknown sampler: {cfg.sampler!r}")


def build_pruner(cfg: TuningSection) -> BasePruner:
    if not cfg.pruner.enabled or cfg.pruner.type == "none":
        return NopPruner()
    if cfg.pruner.type == "median":
        return MedianPruner(
            n_warmup_steps=cfg.pruner.n_warmup_steps,
            n_startup_trials=cfg.pruner.n_startup_trials,
        )
    raise ValueError(f"Unknown pruner type: {cfg.pruner.type!r}")


def build_study(cfg: TuningSection, storage: Optional[str] = None) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=cfg.study_name,
        direction=cfg.direction,
        sampler=build_sampler(cfg),
        pruner=build_pruner(cfg),
        storage=storage,
        load_if_exists=True,
    )
