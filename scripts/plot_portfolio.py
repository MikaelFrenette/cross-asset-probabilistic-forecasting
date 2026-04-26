"""
plot_portfolio.py — Cumulative-return plot: portfolio vs benchmark assets.

Reads the same portfolio config used by ``scripts/construct_portfolio.py``
(so the output dir + realized-returns CSV are already fixed), loads
``returns.csv`` from the run's output directory, joins benchmark return
series from the realized-returns CSV, and plots the cumulative-return
curves with all series indexed at 1.0 on the trading day before the
first rebalance.

Usage
-----
    # Default: plot vs SPY
    python scripts/plot_portfolio.py --config configs/density_model/portfolio/baseline.yaml

    # Plot vs multiple tickers
    python scripts/plot_portfolio.py --config <cfg> --benchmarks SPY AAPL GOOGL

    # Override output path (default: <output_dir>/cumulative_returns.png)
    python scripts/plot_portfolio.py --config <cfg> --output reports/portfolio_vs_spy.png
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from density_model.portfolio import (
    load_panel_portfolio_config,
    plot_portfolio_vs_benchmarks,
)
from density_model.shared.config.logging import configure_logging

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot portfolio cumulative returns vs benchmark assets."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the portfolio config used by construct_portfolio.py.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=["SPY"],
        help="Asset IDs from the realized-returns CSV to overlay (default: SPY).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="PNG output path; default: <output_dir>/cumulative_returns.png.",
    )
    args = parser.parse_args()

    cfg = load_panel_portfolio_config(args.config)
    configure_logging(cfg.logging)

    portfolio_returns_csv = cfg.output_path() / "returns.csv"
    if not portfolio_returns_csv.exists():
        raise SystemExit(
            f"Portfolio returns CSV not found at {portfolio_returns_csv}; "
            "run scripts/construct_portfolio.py first."
        )

    output_path = (
        Path(args.output)
        if args.output
        else cfg.output_path() / "cumulative_returns.png"
    )

    plot_portfolio_vs_benchmarks(
        portfolio_returns_csv=portfolio_returns_csv,
        realized_returns_csv=cfg.realized_returns_path(),
        date_column=cfg.date_column,
        id_column=cfg.id_column,
        return_column=cfg.return_column,
        benchmark_asset_ids=tuple(args.benchmarks),
        output_path=output_path,
    )
    print(f"plot_path={output_path}")


if __name__ == "__main__":
    main()
