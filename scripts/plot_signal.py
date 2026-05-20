"""
plot_signal.py — Render the cross-sectional signal showcase figure.

Two-panel figure demonstrating the model's mu as a tradable
cross-sectional ranking signal:

1. Bar chart — pooled mean realized return per cross-sectional
   ``mu`` quantile bucket, annualized.
2. Cumulative return — long the top bucket, short the bottom bucket
   (equal-weight, daily rebalance), with an optional SPY benchmark
   overlay.

Reads the same panel prediction YAML as ``scripts/predict.py`` /
``scripts/plot.py``.

Usage
-----
    python scripts/plot_signal.py --config configs/density_model/yahoo_predict_next_step_ensemble.yaml
    python scripts/plot_signal.py --config configs/density_model/yahoo_predict_next_step_ensemble.yaml \
        --signal-column mu_over_sigma --n-buckets 5 \
        --output-path docs/images/signal_decile.png
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from density_model.shared.config.logging import configure_logging
from density_model.shared.config.panel_schema import load_panel_predict_config
from density_model.shared.evaluation import generate_signal_plot

logger = logging.getLogger(__name__)

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the cross-sectional signal showcase figure.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the same panel prediction YAML used by predict.py.",
    )
    parser.add_argument(
        "--signal-column",
        choices=["mu", "mu_over_sigma"],
        default="mu",
        help="Which signal to bucket each day (default: mu).",
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=5,
        help="Number of cross-sectional quantile buckets per day (default: 5).",
    )
    parser.add_argument(
        "--benchmark-asset",
        default="SPY",
        help="Ticker to overlay on the cumulative spread panel (use '' to omit).",
    )
    parser.add_argument(
        "--periods-per-year",
        type=int,
        default=252,
        help="Annualization factor (default: 252).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Destination PNG (default: <predictions-dir>/signal_decile.png).",
    )
    args = parser.parse_args()

    cfg = load_panel_predict_config(args.config)
    configure_logging(cfg.logging)

    benchmark = args.benchmark_asset or None
    output_path = generate_signal_plot(
        cfg,
        output_path=args.output_path,
        signal_column=args.signal_column,
        n_buckets=args.n_buckets,
        benchmark_asset=benchmark,
        periods_per_year=args.periods_per_year,
    )
    print(f"signal_plot={output_path}")


if __name__ == "__main__":
    main()
