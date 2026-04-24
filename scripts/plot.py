"""
plot.py — Generate per-asset forecast report plots from panel predictions.

Reads the same panel prediction YAML config as ``scripts/predict.py``. Loads
the manifest (single-model or ensemble) for the target column, joins the
predictions CSV with the realized data CSV on ``(asset_id,
forecast_start_date)``, and renders one three-panel report per asset
(conditional mean, conditional volatility, risk-adjusted signal).

Usage
-----
    python scripts/plot.py --config configs/density_model/yahoo_predict.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from density_model.shared.config.logging import configure_logging
from density_model.shared.config.panel_schema import load_panel_predict_config
from density_model.shared.evaluation import generate_panel_plots

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-asset forecast report plots from panel predictions."
    )
    parser.add_argument("--config", required=True, help="Path to the same panel prediction YAML used by predict.py.")
    args = parser.parse_args()

    cfg = load_panel_predict_config(args.config)
    configure_logging(cfg.logging)

    plots_dir = generate_panel_plots(cfg)
    print(f"plots_dir={plots_dir}")


if __name__ == "__main__":
    main()
