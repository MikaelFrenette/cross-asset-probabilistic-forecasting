"""
plot_calibration.py — Render the σ calibration showcase figure.

Two-panel figure demonstrating the model's probabilistic output:

1. Predicted ``sigma`` for one focus asset (default ``SPY``) overlaid
   with a rolling-window realized-volatility reference.
2. Reliability diagram — empirical coverage of the standardized
   residual ``(y - mu) / sigma`` against nominal Gaussian coverage at
   ±1σ, ±2σ, and a panel of other prediction-interval levels.

Reads the same panel prediction YAML as ``scripts/predict.py`` /
``scripts/plot.py``.

Usage
-----
    python scripts/plot_calibration.py --config configs/density_model/yahoo_predict_next_step_ensemble.yaml
    python scripts/plot_calibration.py --config configs/density_model/yahoo_predict_next_step_ensemble.yaml \
        --focus-asset AAPL --output-path docs/images/sigma_calibration.png
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from density_model.shared.config.logging import configure_logging
from density_model.shared.config.panel_schema import load_panel_predict_config
from density_model.shared.evaluation import generate_calibration_plot

logger = logging.getLogger(__name__)

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the σ calibration showcase figure.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the same panel prediction YAML used by predict.py.",
    )
    parser.add_argument(
        "--focus-asset",
        default="SPY",
        help="Ticker whose σ time series populates the left panel (default: SPY).",
    )
    parser.add_argument(
        "--realized-vol-window",
        type=int,
        default=21,
        help="Rolling-window size (trading days) for the realized-vol reference line.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Destination PNG (default: <predictions-dir>/sigma_calibration.png).",
    )
    args = parser.parse_args()

    cfg = load_panel_predict_config(args.config)
    configure_logging(cfg.logging)

    output_path = generate_calibration_plot(
        cfg,
        output_path=args.output_path,
        focus_asset=args.focus_asset,
        realized_vol_window=args.realized_vol_window,
    )
    print(f"calibration_plot={output_path}")


if __name__ == "__main__":
    main()
