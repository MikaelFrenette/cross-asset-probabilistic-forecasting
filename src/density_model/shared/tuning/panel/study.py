"""
Panel Study Builder
-------------------
Construct Optuna samplers, pruners, and studies from a
:class:`PanelTuningConfig`.

Functions
---------
build_panel_sampler
    Grid or random sampler matching ``cfg.sampler``.

build_panel_pruner
    Median or no-op pruner matching ``cfg.pruner.type``.

build_panel_study
    Optuna study wired with the configured sampler + pruner.
"""

from __future__ import annotations

import optuna
from optuna.pruners import BasePruner, MedianPruner, NopPruner
from optuna.samplers import BaseSampler, GridSampler, RandomSampler

from density_model.shared.config.panel_schema import PanelTuningConfig
from density_model.shared.tuning.search_space import to_hashable

__all__ = ["build_panel_pruner", "build_panel_sampler", "build_panel_study"]


def build_panel_sampler(cfg: PanelTuningConfig) -> BaseSampler:
    if cfg.sampler == "grid":
        normalized_space = {
            path: [to_hashable(c) for c in choices] for path, choices in cfg.search_space.items()
        }
        return GridSampler(normalized_space)
    if cfg.sampler == "random":
        return RandomSampler()
    raise ValueError(f"Unknown sampler: {cfg.sampler!r}")


def build_panel_pruner(cfg: PanelTuningConfig) -> BasePruner:
    if not cfg.pruner.enabled or cfg.pruner.type == "none":
        return NopPruner()
    if cfg.pruner.type == "median":
        return MedianPruner(
            n_warmup_steps=cfg.pruner.n_warmup_steps,
            n_startup_trials=cfg.pruner.n_startup_trials,
        )
    raise ValueError(f"Unknown pruner type: {cfg.pruner.type!r}")


def build_panel_study(cfg: PanelTuningConfig, *, storage: str | None = None) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        study_name=cfg.study_name,
        direction=cfg.direction,
        sampler=build_panel_sampler(cfg),
        pruner=build_panel_pruner(cfg),
        storage=storage,
        load_if_exists=True,
    )
