"""
benchmark.py — Fit a panel forecasting benchmark and write predictions.csv.

Supported benchmark types:
  - unconditional_mean  : per-asset historical mean                (point)
  - ar1                 : per-asset AR(1) closed-form OLS           (point)
  - garch               : per-asset Constant-mean + GARCH(1,1)      (density)

Predictions land at ``cfg.output_csv`` in the same four-column schema used
by ``scripts/predict.py`` so they can be passed to ``scripts/metrics.py
--benchmarks`` / ``--reference`` and compared side-by-side with the
density model.

Usage
-----
    python scripts/benchmark.py --config configs/density_model/benchmarks/unconditional_mean.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from density_model.shared.benchmarks import run_panel_benchmark
from density_model.shared.config.logging import configure_logging
from density_model.shared.config.panel_schema import load_panel_benchmark_config

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a panel forecasting benchmark and write predictions.csv."
    )
    parser.add_argument("--config", required=True, help="Path to panel benchmark YAML config.")
    args = parser.parse_args()

    cfg = load_panel_benchmark_config(args.config)
    configure_logging(cfg.logging)

    predictions_path = run_panel_benchmark(cfg)
    print(f"predictions_path={predictions_path}")


if __name__ == "__main__":
    main()
