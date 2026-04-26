"""
construct_portfolio.py — Build dynamic portfolio weights from density-model forecasts.

Given a ``predictions.csv`` (any density model or ensemble) plus a
realized-returns CSV, run a strict-no-lookahead mean-variance backtest:

    α = zscore(μ/σ)                              (configurable builder)
    Σ = rolling-window sample covariance          (configurable estimator)
    w = argmax_w α'w − (λ/2) w'Σw  s.t. 1'w = 1   (configurable optimizer)
    portfolio_return_{t+1} = w_t · r_{t+1}

Artifacts land at ``cfg.output_dir``:

    portfolio_manifest.json   # config snapshot + run summary + metrics
    weights.csv               # long format (date, asset_id, weight, α, realized_return)
    returns.csv               # per date (portfolio_return, leverage, HHI, effective N)
    metrics.{json,md}         # annual return/vol, Sharpe, MDD, hit rate, turnover, ...

Usage
-----
    python scripts/construct_portfolio.py --config configs/density_model/portfolio/baseline.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from density_model.portfolio import load_panel_portfolio_config, run_portfolio_construction
from density_model.shared.config.logging import configure_logging

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dynamic portfolio weights from density-model forecasts."
    )
    parser.add_argument("--config", required=True, help="Path to panel portfolio YAML config.")
    args = parser.parse_args()

    cfg = load_panel_portfolio_config(args.config)
    configure_logging(cfg.logging)

    output_dir = run_portfolio_construction(cfg)
    print(f"portfolio_dir={output_dir}")


if __name__ == "__main__":
    main()
