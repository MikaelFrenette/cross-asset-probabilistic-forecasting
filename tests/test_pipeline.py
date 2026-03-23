"""
Pipeline Tests
--------------
Unit tests for the chronological forecasting data pipeline.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from density_model.config.config_schema import FeatureConfig
from density_model.data import YahooDailyReturnsRequest
from density_model.pipeline import ChronologicalSplitConfig, ForecastingPipeline

__all__ = []


class FakeYahooLoader:
    """
    Test double for the Yahoo returns loader.

    Parameters
    ----------
    returns : pandas.DataFrame
        Calendar-normalized return matrix returned by the fake loader.
    """

    def __init__(self, returns: pd.DataFrame) -> None:
        self.returns = returns

    def load_returns(self, request: YahooDailyReturnsRequest) -> pd.DataFrame:
        """
        Return a predetermined return matrix.

        Parameters
        ----------
        request : YahooDailyReturnsRequest
            Yahoo Finance request.

        Returns
        -------
        pandas.DataFrame
            Predetermined return matrix.
        """

        return self.returns.copy()


def test_forecasting_pipeline_builds_chronological_vpd_splits() -> None:
    """
    Build train and validation vectorized panels with a chronological split.

    Returns
    -------
    None
        This test asserts chronological pipeline behavior.
    """

    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "QQQ": [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
        },
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
            ]
        ),
    )
    pipeline = ForecastingPipeline(loader=FakeYahooLoader(returns))
    output = pipeline.run(
        request=YahooDailyReturnsRequest(
            tickers=("SPY", "QQQ"),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 9),
        ),
        feature_config=FeatureConfig(
            sequence_length=2,
            forecast_horizon=1,
        ),
        split_config=ChronologicalSplitConfig(validation_start_date="2024-01-08"),
    )

    assert "return" in output.panel.columns
    assert "ticker" in output.panel.columns
    assert np.all(output.train_panel_data["forecast_start_dates"] < np.datetime64("2024-01-08"))
    assert np.all(output.validation_panel_data["forecast_start_dates"] >= np.datetime64("2024-01-08"))


def test_forecasting_pipeline_fits_preprocessing_on_training_panel_only() -> None:
    """
    Fit preprocessing artifacts on the training panel without validation leakage.

    Returns
    -------
    None
        This test asserts chronological preprocessing behavior.
    """

    returns = pd.DataFrame(
        {
            "SPY": [1.0, 3.0, 100.0, 200.0],
        },
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ]
        ),
    )
    pipeline = ForecastingPipeline(loader=FakeYahooLoader(returns))
    output = pipeline.run(
        request=YahooDailyReturnsRequest(
            tickers=("SPY",),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 5),
        ),
        feature_config=FeatureConfig(
            sequence_length=1,
            forecast_horizon=1,
        ),
        split_config=ChronologicalSplitConfig(validation_start_date="2024-01-04"),
    )

    transformed_returns = output.panel.loc[output.panel["asset_id"] == "SPY", "return"].to_numpy()
    expected = np.array([-1.0, 1.0, 98.0, 198.0])
    np.testing.assert_allclose(transformed_returns, expected, rtol=1e-6, atol=1e-6)


def test_forecasting_pipeline_keeps_validation_context_across_split_boundary() -> None:
    """
    Keep validation forecast samples that require pre-boundary lookback history.

    Returns
    -------
    None
        This test asserts VPD splitting occurs after sequence construction.
    """

    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.02, 0.03, 0.04, 0.05],
            "QQQ": [0.11, 0.12, 0.13, 0.14, 0.15],
        },
        index=pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ]
        ),
    )
    pipeline = ForecastingPipeline(loader=FakeYahooLoader(returns))
    output = pipeline.run(
        request=YahooDailyReturnsRequest(
            tickers=("SPY", "QQQ"),
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 8),
        ),
        feature_config=FeatureConfig(
            sequence_length=2,
            forecast_horizon=1,
        ),
        split_config=ChronologicalSplitConfig(validation_start_date="2024-01-05"),
    )

    validation_forecast_dates = output.validation_panel_data["forecast_start_dates"]
    assert np.datetime64("2024-01-05") in validation_forecast_dates
    assert output.validation_panel_data["X_continuous"].shape[0] == 2


def test_forecasting_pipeline_rejects_empty_validation_split() -> None:
    """
    Reject a chronological split that yields an empty validation panel.

    Returns
    -------
    None
        This test asserts fail-fast split validation.
    """

    returns = pd.DataFrame(
        {"SPY": [0.01, 0.02, 0.03]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    pipeline = ForecastingPipeline(loader=FakeYahooLoader(returns))

    try:
        pipeline.run(
            request=YahooDailyReturnsRequest(
                tickers=("SPY",),
                start_date=date(2024, 1, 2),
                end_date=date(2024, 1, 4),
            ),
                feature_config=FeatureConfig(
                    sequence_length=2,
                    forecast_horizon=1,
                ),
                split_config=ChronologicalSplitConfig(validation_start_date="2024-01-10"),
            )
    except ValueError as error:
        assert "empty validation panel" in str(error)
    else:
        raise AssertionError("Expected an empty validation split to raise ValueError.")
