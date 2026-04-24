"""
preprocess.py — Fit preprocessing and vectorize panel forecasting splits.

Reads the train / val feature CSVs referenced by a panel pipeline config,
fits a :class:`PreprocessingBundle` on the training panel, transforms both
splits, vectorizes them into ``(B, ID, T, K)`` tensors, and writes the
preprocessing bundle + vectorized panels into the configured artifacts
directory. Called before ``train.py``.

Usage
-----
    python scripts/preprocess.py --config configs/density_model/yahoo_volatility.yaml
    python scripts/preprocess.py --config <cfg> --overwrite
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dlecosys.shared.config.logging import configure_logging
from dlecosys.shared.config.panel_schema import load_panel_pipeline_config
from dlecosys.shared.training.panel_preprocess import run_panel_preprocess

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit preprocessing and vectorize panel forecasting splits."
    )
    parser.add_argument("--config", required=True, help="Path to panel pipeline YAML config.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing preprocessing / panel artifacts.",
    )
    args = parser.parse_args()

    cfg = load_panel_pipeline_config(args.config)
    configure_logging(cfg.logging)

    output_dir = run_panel_preprocess(cfg, overwrite=args.overwrite)
    print(f"artifacts_dir={output_dir}")


if __name__ == "__main__":
    main()
