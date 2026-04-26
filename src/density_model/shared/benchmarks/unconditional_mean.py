"""
Unconditional-Mean Benchmark
----------------------------
Per-asset historical mean: the fit takes the mean of the target over each
asset's training window; prediction returns that constant for every test
date. ``sigma = NaN`` because this is a point forecast only — density
metrics (NLL, CRPS, coverage) should be skipped for it.

Typical use: reference denominator for the RMSE skill score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from density_model.shared.benchmarks.base import BaseBenchmark

__all__ = ["UnconditionalMeanBenchmark"]


class UnconditionalMeanBenchmark(BaseBenchmark):
    """Predict each asset's historical training mean for every test date."""

    def __init__(self) -> None:
        self.means_: dict[str, float] = {}

    def fit(
        self,
        train_panel: pd.DataFrame,
        *,
        date_column: str,
        id_column: str,
        target_column: str,
    ) -> None:
        grouped = train_panel.groupby(id_column)[target_column]
        self.means_ = {str(asset): float(value) for asset, value in grouped.mean().items()}

    def predict(
        self,
        predict_panel: pd.DataFrame,
        *,
        date_column: str,
        id_column: str,
        target_column: str,
    ) -> pd.DataFrame:
        out = predict_panel.loc[:, [date_column, id_column]].copy()
        out["mu"] = out[id_column].map(self.means_).astype(float)
        out["sigma"] = np.nan
        out = out.rename(
            columns={date_column: "forecast_start_date", id_column: "asset_id"}
        )
        out["forecast_end_date"] = out["forecast_start_date"]
        return out.loc[
            :, ["forecast_start_date", "forecast_end_date", "asset_id", "mu", "sigma"]
        ]
