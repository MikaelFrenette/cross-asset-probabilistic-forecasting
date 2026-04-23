"""
ensemble.py — Train every estimator in a bagging ensemble.

Each estimator is an independent ``run_training`` call with a bootstrap sample
as its train set and the complementary out-of-bag (OOB) samples as its
validation set. EarlyStopping and ModelCheckpoint therefore target OOB loss.

Expects preprocess.py to have been run first with the same config; preprocess
writes raw (unscaled) train/val/test when the config has an ``ensemble`` section.

Usage
-----
    python scripts/ensemble.py --config configs/demo/regression_ensemble.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dlecosys.models  # noqa: F401 — registers bundled models with the factory

from dlecosys.shared.config import configure_logging, load_config
from dlecosys.shared.ensembling.runner import run_ensemble
from dlecosys.shared.run import EnsembleLayout

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bagging ensemble.")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config with an ensemble section.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_logging(cfg.logging)

    if cfg.ensemble is None:
        logger.error("config has no 'ensemble' section — nothing to do")
        sys.exit(1)

    layout = EnsembleLayout(cfg.experiment.output_dir, cfg.experiment.name)

    if not layout.data_path("train").exists():
        logger.error(
            "preprocessed pool not found at %s — run preprocess.py first",
            layout.data_path("train"),
        )
        sys.exit(1)

    run_ensemble(cfg, layout)

    logger.info("ensemble complete — run predict.py to aggregate test predictions")


if __name__ == "__main__":
    main()
