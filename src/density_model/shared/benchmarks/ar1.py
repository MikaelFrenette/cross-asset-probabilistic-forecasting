"""
AR(1) Benchmark
---------------
Per-asset AR(1) regression fit by closed-form OLS on
``y_t = const + phi * y_{t-1}``. Prediction is the one-step-ahead rolling
forecast: for each test date ``t``, predict ``const + phi * y_{t-1}``
where ``y_{t-1}`` is the most recently observed target (the last training
value carries over into the first test step).

Point forecast only (``sigma = NaN``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from density_model.shared.benchmarks.base import BaseBenchmark

__all__ = ["AR1Benchmark"]


class AR1Benchmark(BaseBenchmark):
    """Per-asset AR(1) one-step-ahead forecaster."""

    def __init__(self) -> None:
        self.coefs_: dict[str, tuple[float, float]] = {}
        self.last_train_value_: dict[str, float] = {}

    def fit(
        self,
        train_panel: pd.DataFrame,
        *,
        date_column: str,
        id_column: str,
        target_column: str,
    ) -> None:
        for asset, group in train_panel.groupby(id_column, sort=False):
            series = (
                group.sort_values(date_column)[target_column]
                .dropna()
                .to_numpy(dtype=float)
            )
            if series.size < 3:
                # Not enough history — fall back to unconditional mean with phi = 0.
                mean_value = float(series.mean()) if series.size else 0.0
                self.coefs_[str(asset)] = (mean_value, 0.0)
                self.last_train_value_[str(asset)] = (
                    float(series[-1]) if series.size else float("nan")
                )
                continue
            y_t = series[1:]
            y_tm1 = series[:-1]
            design = np.column_stack([np.ones_like(y_tm1), y_tm1])
            coefs, *_ = np.linalg.lstsq(design, y_t, rcond=None)
            const, phi = float(coefs[0]), float(coefs[1])
            self.coefs_[str(asset)] = (const, phi)
            self.last_train_value_[str(asset)] = float(series[-1])

    def predict(
        self,
        predict_panel: pd.DataFrame,
        *,
        date_column: str,
        id_column: str,
        target_column: str,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for asset, group in predict_panel.groupby(id_column, sort=False):
            key = str(asset)
            if key not in self.coefs_:
                continue
            const, phi = self.coefs_[key]
            prev = self.last_train_value_.get(key, float("nan"))
            group_sorted = group.sort_values(date_column).reset_index(drop=True)
            realized = group_sorted[target_column].to_numpy(dtype=float)
            dates = group_sorted[date_column].to_list()
            for i, date in enumerate(dates):
                mu = const + phi * prev if not np.isnan(prev) else float("nan")
                rows.append(
                    {
                        "forecast_start_date": date,
                        "forecast_end_date": date,
                        "asset_id": asset,
                        "mu": mu,
                        "sigma": float("nan"),
                    }
                )
                if not np.isnan(realized[i]):
                    prev = float(realized[i])
        return pd.DataFrame(rows).loc[
            :, ["forecast_start_date", "forecast_end_date", "asset_id", "mu", "sigma"]
        ]
