"""
Best-Config Emitter
-------------------
Apply an Optuna study's winning hyperparameters to the base pipeline config,
strip the ``tuning`` section, and emit a ``best_config.yaml`` ready to run
through ``scripts/preprocess.py`` + ``scripts/train.py``.

Functions
---------
emit_best_config
    Persist ``best_params.yaml``, ``best_config.yaml``, and ``trials.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import optuna
import yaml

from density_model.shared.config.panel_schema import PanelPipelineConfig
from density_model.shared.tuning.search_space import apply_suggestion

__all__ = ["emit_best_config"]


def _deep_merge(base: dict, overrides: dict) -> dict:
    """In-place deep merge ``overrides`` into ``base``; leaf values overwrite."""

    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def emit_best_config(
    *,
    study: optuna.Study,
    base_cfg: PanelPipelineConfig,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """
    Write ``best_params.yaml`` + ``best_config.yaml`` + ``trials.csv``.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study.
    base_cfg : PanelPipelineConfig
        Base pipeline config the study was driven from.
    output_dir : pathlib.Path
        Destination directory for study artifacts.

    Returns
    -------
    tuple of (pathlib.Path, pathlib.Path, pathlib.Path)
        Paths to ``best_params.yaml``, ``best_config.yaml``, and ``trials.csv``.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    best_params = dict(study.best_params)
    best_value = study.best_value

    best_params_path = output_dir / "best_params.yaml"
    best_params_path.write_text(
        yaml.safe_dump(
            {"best_value": float(best_value), "best_params": best_params},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg_dict = base_cfg.model_dump()
    for path, value in best_params.items():
        apply_suggestion(cfg_dict, path, value)

    # Apply any post_tuning overrides BEFORE stripping the tuning section.
    # This lets users separate the tuning-time budget (small max_epochs,
    # EarlyStopping off, ModelCheckpoint off) from the winner's full-budget
    # training settings (large max_epochs, ES/MC on).
    if base_cfg.tuning is not None and base_cfg.tuning.post_tuning:
        _deep_merge(cfg_dict, base_cfg.tuning.post_tuning)

    cfg_dict["tuning"] = None
    best_config_path = output_dir / "best_config.yaml"
    best_config_path.write_text(
        yaml.safe_dump(cfg_dict, sort_keys=False),
        encoding="utf-8",
    )

    trials_path = output_dir / "trials.csv"
    with trials_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        param_names = sorted(best_params.keys())
        writer.writerow(["trial_number", "state", "value", *param_names])
        for trial in study.trials:
            writer.writerow(
                [
                    trial.number,
                    trial.state.name,
                    trial.value if trial.value is not None else "",
                    *(trial.params.get(name, "") for name in param_names),
                ]
            )

    return best_params_path, best_config_path, trials_path
