"""
predict.py — Run panel forecasting inference from a frozen training manifest.

Reads a panel prediction config, loads the manifest + preprocessing bundle +
model weights produced by ``scripts/train.py``, calendar-normalizes the raw
input CSV, runs batched rolling inference, inverse-transforms Gaussian
``(mu, sigma)`` back to return space, and writes per-asset per-date forecasts
to the configured output CSV.

Usage
-----
    python scripts/predict.py --config configs/density_model/yahoo_predict.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dlecosys.shared.config.logging import configure_logging
from dlecosys.shared.config.panel_schema import load_panel_predict_config
from dlecosys.shared.inference.panel_predictor import predict_from_config

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run panel forecasting inference.")
    parser.add_argument("--config", required=True, help="Path to panel prediction YAML config.")
    args = parser.parse_args()

    cfg = load_panel_predict_config(args.config)
    configure_logging(cfg.logging)

    predictions_path = predict_from_config(cfg)
    print(f"predictions_path={predictions_path}")


if __name__ == "__main__":
    main()
