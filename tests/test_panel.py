"""
Panel Builder Tests
-------------------
Unit tests for Yahoo-return panel construction.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from density_model.config.config_schema import FeatureConfig
from density_model.data import YahooVolatilityPanelBuilder

__all__ = []


def test_yahoo_volatility_panel_builder_constructs_long_panel() -> None:
    """
    Convert a wide return matrix into a canonical long-format panel.

    Returns
    -------
    None
        This test asserts return panel construction.
    """

    returns = pd.DataFrame(
        {
            "SPY": [0.10, 0.20, np.nan, 0.40],
            "QQQ": [0.05, 0.15, 0.25, 0.35],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    )
    feature_config = FeatureConfig(
        sequence_length=3,
        forecast_horizon=1,
    )

    builder = YahooVolatilityPanelBuilder()
    panel = builder.build_from_returns(returns=returns, feature_config=feature_config)

    assert list(panel.columns) == ["date", "asset_id", "return", "ticker"]
    assert len(panel) == 8
    assert panel["ticker"].equals(panel["asset_id"])
    spy_rows = panel.loc[panel["asset_id"] == "SPY"].reset_index(drop=True)
    assert np.isnan(spy_rows.loc[2, "return"])
