"""
metrics.py — Compute density-forecast metrics; compare against benchmarks.

Reads the same panel prediction YAML config as ``scripts/predict.py`` /
``scripts/plot.py``. Joins the primary predictions CSV with the realized
target series from the raw data CSV on ``(asset_id, forecast_start_date)``
and computes:

  - Gaussian NLL (training convention)
  - CRPS (closed form for Gaussian)
  - MAE / RMSE / bias on the predicted mean
  - Coverage at ±1σ and ±2σ (target 0.683 / 0.955)
  - Mean predicted σ

Benchmark comparison
--------------------
``--benchmarks PATH [PATH ...]`` loads each benchmark's predictions.csv
(written by ``scripts/benchmark.py``) and reports its metrics on the same
row in the overall comparison table. Density metrics are silently skipped
for point-only benchmarks whose ``sigma`` column is NaN.

``--reference PATH`` designates one of the benchmarks as the RMSE
skill-score denominator. For every model the report includes

    rmse_skill = rmse_model / rmse_reference

at both the overall (scalar) and per-asset levels. Lower is better;
< 1 means the model beats the reference on RMSE.

Output files (alongside the primary predictions.csv):
  - metrics.json
  - metrics_per_asset.csv   (long format; one row per (model, asset))
  - metrics.md

Usage
-----
    python scripts/metrics.py --config configs/density_model/yahoo_predict_next_step_ensemble.yaml

    # Compare against three benchmarks; use unconditional_mean as the
    # skill-score reference denominator:
    python scripts/metrics.py \
        --config configs/density_model/yahoo_predict_next_step_ensemble.yaml \
        --benchmarks outputs/benchmarks/unconditional_mean/predictions.csv \
                     outputs/benchmarks/ar1/predictions.csv \
                     outputs/benchmarks/garch/predictions.csv \
        --reference  outputs/benchmarks/unconditional_mean/predictions.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from density_model.shared.config.logging import configure_logging
from density_model.shared.config.panel_schema import load_panel_predict_config
from density_model.shared.evaluation import generate_panel_metrics

logger = logging.getLogger(__name__)

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute density-forecast metrics; compare against benchmarks."
    )
    parser.add_argument("--config", required=True, help="Path to the same panel prediction YAML used by predict.py.")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional paths to benchmark predictions.csv files to compare side-by-side.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional path to the benchmark predictions.csv used as the rmse_skill denominator.",
    )
    args = parser.parse_args()

    cfg = load_panel_predict_config(args.config)
    configure_logging(cfg.logging)

    benchmark_paths = [Path(p) for p in (args.benchmarks or [])]
    reference_path = Path(args.reference) if args.reference else None

    output_dir = generate_panel_metrics(
        cfg,
        benchmark_paths=benchmark_paths,
        reference_path=reference_path,
    )
    print(f"metrics_dir={output_dir}")


if __name__ == "__main__":
    main()
