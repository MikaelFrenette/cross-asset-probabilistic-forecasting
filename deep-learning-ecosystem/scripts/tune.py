"""
tune.py — Run a hyperparameter tuning study.

Expects preprocess.py to have been run first on the same config so that
preprocessed splits already exist under the study's data directory.

Usage
-----
    python scripts/tune.py --config configs/demo/regression_tuning.yaml
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dlecosys.models  # noqa: F401 — registers bundled models with the factory

from dlecosys.shared.config import configure_logging, load_config
from dlecosys.shared.run import StudyLayout
from dlecosys.shared.tuning.objective import build_objective
from dlecosys.shared.tuning.search_space import apply_suggestion, from_hashable
from dlecosys.shared.tuning.study import build_study


def _format_param(v):
    return f"{v:.5f}" if isinstance(v, float) else repr(v)


def _print_trial_result(study, trial):
    state = trial.state.name
    value_str = f"{trial.value:.5f}" if trial.value is not None else "—"
    params_str = ", ".join(f"{k}: {_format_param(v)}" for k, v in trial.params.items())
    line = f"Trial {trial.number} {state} — value: {value_str} | {params_str}"
    if trial.state.name == "COMPLETE" and study.best_trial is not None:
        best = study.best_trial
        line += f" | best: {best.value:.5f} (trial {best.number})"
    print(line)

logger = logging.getLogger(__name__)


def _grid_size(search_space: dict) -> int:
    total = 1
    for choices in search_space.values():
        total *= len(choices)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a hyperparameter tuning study.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config with a tuning section.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)

    if cfg.tuning is None:
        logger.error("config has no 'tuning' section — nothing to do")
        sys.exit(1)

    study_layout = StudyLayout(cfg.experiment.output_dir, cfg.tuning.study_name)

    if not study_layout.data_dir.exists():
        logger.error(
            "preprocessed data not found at %s — run preprocess.py first",
            study_layout.data_dir,
        )
        sys.exit(1)

    if cfg.tuning.sampler == "grid":
        n_trials = cfg.tuning.n_trials or _grid_size(cfg.tuning.search_space)
    else:
        n_trials = cfg.tuning.n_trials or 50

    study = build_study(cfg.tuning, storage=cfg.tuning.storage)
    objective = build_objective(cfg, study_layout)

    logger.info(
        "starting study '%s' — sampler=%s, pruner=%s, n_trials=%d",
        cfg.tuning.study_name,
        cfg.tuning.sampler,
        "on" if cfg.tuning.pruner.enabled else "off",
        n_trials,
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_print_trial_result])

    best_trial = study.best_trial
    with open(study_layout.best_params_path, "w") as f:
        yaml.dump(
            {
                "metric": cfg.tuning.metric,
                "direction": cfg.tuning.direction,
                "best_value": float(best_trial.value),
                "best_trial_number": best_trial.number,
                "params": best_trial.params,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )

    best_cfg_dict = cfg.model_dump()
    best_cfg_dict["tuning"] = None
    for path, value in best_trial.params.items():
        apply_suggestion(best_cfg_dict, path, from_hashable(value))
    best_cfg_dict["experiment"]["name"] = f"{cfg.tuning.study_name}_best"
    with open(study_layout.best_config_path, "w") as f:
        yaml.dump(best_cfg_dict, f, default_flow_style=False, sort_keys=False)

    with open(study_layout.trials_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["trial_number", "state", "value"] + list(cfg.tuning.search_space.keys())
        writer.writerow(header)
        for t in study.trials:
            row = [t.number, t.state.name, t.value]
            for key in cfg.tuning.search_space.keys():
                row.append(t.params.get(key))
            writer.writerow(row)

    logger.info(
        "study complete — best %s = %.6f at trial %d",
        cfg.tuning.metric,
        float(best_trial.value),
        best_trial.number,
    )
    logger.info("best params → %s", study_layout.best_params_path)
    logger.info("best config → %s", study_layout.best_config_path)
    logger.info("trials → %s", study_layout.trials_csv)
    logger.info(
        "next step: python scripts/preprocess.py --config %s && "
        "python scripts/train.py --config %s",
        study_layout.best_config_path,
        study_layout.best_config_path,
    )


if __name__ == "__main__":
    main()
