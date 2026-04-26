"""
Benchmark Base
--------------
Abstract base class for panel forecasting benchmarks.

Every benchmark fits on a training long-format panel and emits predictions
for a test long-format panel in the same four-column CSV schema used by
``scripts/predict.py`` (``forecast_start_date``, ``forecast_end_date``,
``asset_id``, ``mu``, ``sigma``). Point-forecast benchmarks set
``sigma = NaN`` to signal that density metrics (NLL, CRPS, coverage)
should be skipped for them by the metrics renderer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

__all__ = ["BaseBenchmark"]


class BaseBenchmark(ABC):
    """Fit-then-predict interface for panel forecasting benchmarks."""

    @abstractmethod
    def fit(
        self,
        train_panel: pd.DataFrame,
        *,
        date_column: str,
        id_column: str,
        target_column: str,
    ) -> None:
        """Fit the benchmark on the training long-format panel."""

    @abstractmethod
    def predict(
        self,
        predict_panel: pd.DataFrame,
        *,
        date_column: str,
        id_column: str,
        target_column: str,
    ) -> pd.DataFrame:
        """
        Produce predictions for every (asset, date) in the prediction panel.

        Returns
        -------
        pandas.DataFrame
            Columns ``(forecast_start_date, forecast_end_date, asset_id,
            mu, sigma)``. ``sigma`` may be NaN for point-only benchmarks.
        """
