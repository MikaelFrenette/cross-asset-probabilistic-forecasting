"""
ensemble.py — Train every estimator in a panel bagging ensemble.

Each estimator bootstraps its own subset of forecast dates from the training
CSV, optionally subsamples assets, re-fits its own preprocessing bundle on
the in-bag slice (leak-free), trains with the out-of-bag dates as its
validation set, and writes its own ``preprocessing.json`` + ``model.pt`` +
``training_manifest.json`` under
``<artifacts.output_dir>/ensemble/estimator_k/``. After all estimators
finish, optional top-N pruning keeps only the best survivors and emits
``ensemble_manifest.json`` consumed by ``scripts/predict.py``.

Usage
-----
    python scripts/ensemble.py --config configs/density_model/yahoo_volatility_ensemble.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dlecosys.shared.config.logging import configure_logging
from dlecosys.shared.config.panel_schema import load_panel_pipeline_config
from dlecosys.shared.ensembling.panel.run import run_panel_ensemble

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train every estimator in a panel bagging ensemble."
    )
    parser.add_argument("--config", required=True, help="Path to panel pipeline YAML config with an ensemble: section.")
    args = parser.parse_args()

    cfg = load_panel_pipeline_config(args.config)
    if cfg.ensemble is None:
        raise SystemExit(
            f"Config {args.config!r} has no 'ensemble:' section; "
            "cannot train a bagging ensemble."
        )
    configure_logging(cfg.logging)

    ensemble_dir = run_panel_ensemble(cfg)
    print(f"ensemble_dir={ensemble_dir}")


if __name__ == "__main__":
    main()
