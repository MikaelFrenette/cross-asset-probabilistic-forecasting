"""
train.py — Train a panel forecasting model for a pipeline run.

Expects that ``scripts/preprocess.py`` has already written the preprocessing
bundle and vectorized panels into the configured artifacts directory. Loads
those, builds the registered model via :class:`ModelFactory`, runs
:class:`PanelTrainer` with mask-aware Gaussian NLL, and writes the trained
model weights + training manifest.

Usage
-----
    python scripts/train.py --config configs/density_model/yahoo_volatility.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dlecosys.shared.config.logging import configure_logging
from dlecosys.shared.config.panel_schema import load_panel_pipeline_config
from dlecosys.shared.training.panel_run import run_panel_training

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a panel forecasting model.")
    parser.add_argument("--config", required=True, help="Path to panel pipeline YAML config.")
    args = parser.parse_args()

    cfg = load_panel_pipeline_config(args.config)
    configure_logging(cfg.logging)

    output_dir = run_panel_training(cfg)
    print(f"artifacts_dir={output_dir}")


if __name__ == "__main__":
    main()
