"""
Benchmark Runner
----------------
End-to-end orchestration for a panel benchmark: load train + predict CSVs,
fit on train, predict on test, and write ``predictions.csv`` to the
configured output path. The emitted CSV has the same column schema as
``scripts/predict.py`` so ``scripts/metrics.py`` and ``scripts/plot.py``
can consume it identically.

Functions
---------
run_panel_benchmark
    Entry point called by ``scripts/benchmark.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from density_model.shared.benchmarks.build import build_panel_benchmark
from density_model.shared.config.panel_schema import PanelBenchmarkConfig

__all__ = ["run_panel_benchmark"]

logger = logging.getLogger(__name__)


def run_panel_benchmark(cfg: PanelBenchmarkConfig) -> Path:
    """Fit the configured benchmark on train, predict on test, write predictions.csv."""

    train = _load_panel(
        csv_path=cfg.train_csv_path(),
        date_column=cfg.date_column,
        id_column=cfg.id_column,
        target_column=cfg.target_column,
    )
    predict_frame = _load_panel(
        csv_path=cfg.predict_csv_path(),
        date_column=cfg.date_column,
        id_column=cfg.id_column,
        target_column=cfg.target_column,
    )

    benchmark = build_panel_benchmark(cfg)
    logger.info("Fitting %s on %s", cfg.type, cfg.train_csv)
    benchmark.fit(
        train,
        date_column=cfg.date_column,
        id_column=cfg.id_column,
        target_column=cfg.target_column,
    )
    logger.info("Predicting on %s", cfg.predict_csv)
    predictions = benchmark.predict(
        predict_frame,
        date_column=cfg.date_column,
        id_column=cfg.id_column,
        target_column=cfg.target_column,
    )
    if predictions.empty:
        raise ValueError(
            f"Benchmark {cfg.type!r} produced no predictions — check the predict CSV."
        )

    output_path = cfg.output_path().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logger.info("Wrote %d predictions to %s", len(predictions), output_path)
    return output_path


def _load_panel(
    *,
    csv_path: Path,
    date_column: str,
    id_column: str,
    target_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {date_column, id_column, target_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(
            f"Benchmark panel {csv_path!s} is missing required columns: "
            f"{', '.join(missing)}"
        )
    frame = frame.loc[:, [id_column, date_column, target_column]].copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    return frame.sort_values([id_column, date_column]).reset_index(drop=True)
