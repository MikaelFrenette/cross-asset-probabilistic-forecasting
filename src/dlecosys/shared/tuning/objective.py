"""
Trial Objective
---------------
Build the objective function that runs a single tuning trial: apply suggested
hyperparameters to the base config and train via the shared run_training
entrypoint.

Functions
---------
build_objective
    Return an Optuna-compatible objective closure bound to a base config and
    a study layout.
"""

from __future__ import annotations

import logging
import os
import statistics
import sys
from typing import Callable

import optuna
import yaml

from dlecosys.shared.config.schema import PipelineConfig
from dlecosys.shared.data import build_splitter, preprocess_fold
from dlecosys.shared.run import StudyLayout
from dlecosys.shared.training import run_training
from dlecosys.shared.tuning.pruning import PruningCallback
from dlecosys.shared.tuning.search_space import apply_suggestion, suggest_values

__all__ = ["build_objective"]

logger = logging.getLogger(__name__)


def _clear_console() -> None:
    if not sys.stdout.isatty():
        return
    os.system("cls" if os.name == "nt" else "clear")


def _format_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.5f}"
    return str(v)


def _render_trial_header(
    trial: optuna.Trial,
    suggestions: dict,
    tcfg,
    *,
    fold_idx: int = 0,
    n_folds: int = 1,
) -> None:
    _clear_console()
    study = trial.study
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]

    title = f"Study: {study.study_name}  |  Trial {trial.number + 1}"
    if n_folds > 1:
        title += f"  |  Fold {fold_idx + 1}/{n_folds}"

    print("=" * 72)
    print(title)
    print("-" * 72)
    print(f"Metric     : {tcfg.metric} ({tcfg.direction})")
    if completed:
        best = study.best_trial
        print(f"Best so far: {_format_value(float(best.value))} (trial {best.number})")
        print("Best params:")
        for k, v in best.params.items():
            print(f"    {k}: {_format_value(v)}")
    else:
        print("Best so far: — (no completed trials yet)")
    print("-" * 72)
    print("Current hyperparameters:")
    for k, v in suggestions.items():
        print(f"    {k}: {_format_value(v)}")
    print("=" * 72)
    sys.stdout.flush()


def build_objective(
    base_cfg: PipelineConfig,
    study_layout: StudyLayout,
) -> Callable[[optuna.Trial], float]:
    """
    Return an Optuna objective bound to a base config and study layout.

    Each call suggests values, applies them to a copy of the base config,
    creates a per-trial RunLayout sharing the study's data dir, and delegates
    to the shared ``run_training`` entrypoint. The final value of the
    configured metric is returned to Optuna.
    """
    if base_cfg.tuning is None:
        raise ValueError("build_objective requires a config with a non-null tuning section")

    tcfg_tuning = base_cfg.tuning

    def objective(trial: optuna.Trial) -> float:
        suggestions = suggest_values(trial, tcfg_tuning.search_space)
        cfg_dict = base_cfg.model_dump()
        for path, value in suggestions.items():
            apply_suggestion(cfg_dict, path, value)
        cfg = PipelineConfig(**cfg_dict)

        trial_layout = study_layout.trial_layout(trial.number)
        trial_layout.create(overwrite=True)
        with open(trial_layout.config_path, "w") as f:
            yaml.dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)

        splitter = build_splitter(tcfg_tuning.splitter, trial_layout)

        fold_values: list[float] = []
        multi_fold = splitter.n_splits > 1
        for fold_idx, (train_ds, val_ds) in enumerate(splitter.iterate_folds()):
            _render_trial_header(
                trial,
                suggestions,
                tcfg_tuning,
                fold_idx=fold_idx,
                n_folds=splitter.n_splits,
            )

            if tcfg_tuning.scale_splits:
                train_ds, val_ds = preprocess_fold(cfg.data, train_ds, val_ds)

            extra_callbacks = []
            if tcfg_tuning.pruner.enabled:
                extra_callbacks.append(PruningCallback(trial, monitor=tcfg_tuning.metric))

            fold_layout = trial_layout.fold_layout(fold_idx) if multi_fold else trial_layout
            if multi_fold:
                fold_layout.create(overwrite=True)

            trainer = run_training(
                cfg,
                fold_layout,
                extra_callbacks=extra_callbacks,
                summary_extras={"trial_number": trial.number, "fold": fold_idx},
                train_dataset=train_ds,
                val_dataset=val_ds,
                render_summary=False,
            )

            final_logs = trainer.logger.last_log()
            metric_val = final_logs.get(tcfg_tuning.metric)
            if metric_val is None:
                raise ValueError(
                    f"Metric '{tcfg_tuning.metric}' not in final logs: {list(final_logs.keys())}"
                )
            fold_values.append(float(metric_val))

        return statistics.mean(fold_values)

    return objective
