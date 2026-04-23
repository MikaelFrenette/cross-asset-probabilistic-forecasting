"""
tune.py — Run an Optuna hyperparameter tuning study for a panel pipeline.

Reads a panel pipeline config with a ``tuning:`` section, concatenates the
train + val CSVs into a single canonical panel, iterates Optuna trials
through a time-series splitter that re-fits preprocessing per fold (leak-
free), and emits ``best_params.yaml`` + ``best_config.yaml`` + ``trials.csv``
into ``<artifacts.output_dir>/studies/<study_name>/``.

Usage
-----
    python scripts/tune.py --config configs/density_model/yahoo_volatility_tuning.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dlecosys.shared.config.logging import configure_logging
from dlecosys.shared.config.panel_schema import load_panel_pipeline_config
from dlecosys.shared.tuning.panel.run import run_panel_tuning

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an Optuna hyperparameter tuning study for a panel pipeline."
    )
    parser.add_argument("--config", required=True, help="Path to panel pipeline YAML config with a tuning: section.")
    args = parser.parse_args()

    cfg = load_panel_pipeline_config(args.config)
    if cfg.tuning is None:
        raise SystemExit(
            f"Config {args.config!r} has no 'tuning:' section; "
            "cannot run hyperparameter search."
        )
    configure_logging(cfg.logging)

    study_dir = run_panel_tuning(cfg)
    print(f"study_dir={study_dir}")


if __name__ == "__main__":
    main()
