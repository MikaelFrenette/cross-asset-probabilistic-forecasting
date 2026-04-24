"""
Panel Tuning Runner
-------------------
End-to-end Optuna hyperparameter search entry point. Loads the raw
concatenated train + val CSV (so the time-series splitter sees the full
history), constructs the study, runs every trial through
:class:`PanelObjective`, and emits ``best_params.yaml`` + ``best_config.yaml``
+ ``trials.csv`` once the study finishes.

Functions
---------
run_panel_tuning
    Top-level tuning entry point called by ``scripts/tune.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from density_model.shared.config.panel_schema import PanelPipelineConfig
from density_model.shared.tuning.panel.best_config import emit_best_config
from density_model.shared.tuning.panel.display import after_trial_callback, render_leaderboard
from density_model.shared.tuning.panel.objective import PanelObjective
from density_model.shared.tuning.panel.splitter_builder import build_panel_splitter
from density_model.shared.tuning.panel.study import build_panel_study

__all__ = ["run_panel_tuning"]

logger = logging.getLogger(__name__)


def run_panel_tuning(cfg: PanelPipelineConfig) -> Path:
    """
    Run an Optuna hyperparameter study over a panel pipeline config.

    Parameters
    ----------
    cfg : PanelPipelineConfig
        Validated pipeline config. ``cfg.tuning`` must be set.

    Returns
    -------
    pathlib.Path
        Output directory containing ``best_params.yaml`` + ``best_config.yaml``
        + ``trials.csv``.
    """

    if cfg.tuning is None:
        raise ValueError("run_panel_tuning requires cfg.tuning to be set.")

    logger.info("Loading combined train + val panel for tuning")
    raw_panel = _load_combined_panel(cfg)

    splitter = build_panel_splitter(cfg.tuning.splitter)
    study = build_panel_study(cfg.tuning)
    objective = PanelObjective(base_cfg=cfg, raw_panel=raw_panel, splitter=splitter)

    logger.info("Starting study %r", cfg.tuning.study_name)
    study.optimize(
        objective,
        n_trials=cfg.tuning.n_trials,
        callbacks=[after_trial_callback],
        show_progress_bar=False,
    )

    output_dir = (cfg.artifacts.output_path() / f"studies/{cfg.tuning.study_name}").resolve()
    best_params_path, best_config_path, trials_path = emit_best_config(
        study=study, base_cfg=cfg, output_dir=output_dir
    )
    render_leaderboard(
        study=study,
        study_name=cfg.tuning.study_name,
        metric=cfg.tuning.metric,
        direction=cfg.tuning.direction,
    )
    logger.info(
        "Study complete. Best value=%s; wrote %s, %s, %s",
        study.best_value,
        best_params_path,
        best_config_path,
        trials_path,
    )
    return output_dir


def _load_combined_panel(cfg: PanelPipelineConfig) -> pd.DataFrame:
    """Concatenate the train + val CSVs so the time-series splitter sees one history."""

    required = {
        cfg.data.date_column,
        cfg.data.id_column,
        cfg.features.target_column,
        *cfg.features.continuous_columns,
        *cfg.features.dynamic_categorical_columns,
        *cfg.features.static_categorical_columns,
    }
    frames = []
    for path in (cfg.data.train_csv_path(), cfg.data.val_csv_path()):
        frame = pd.read_csv(path)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"Panel CSV {path} is missing required columns: {', '.join(missing)}"
            )
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined[cfg.data.date_column] = pd.to_datetime(combined[cfg.data.date_column])
    combined = combined.drop_duplicates(
        subset=[cfg.data.id_column, cfg.data.date_column]
    )
    return combined.sort_values([cfg.data.id_column, cfg.data.date_column]).reset_index(
        drop=True
    )
