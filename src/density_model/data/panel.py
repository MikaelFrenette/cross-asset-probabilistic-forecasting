"""
Yahoo Return Panel
------------------
Build canonical long-format panels from calendar-normalized Yahoo returns for
return forecasting workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from density_model.data.yahoo import YahooDailyReturnsLoader, YahooDailyReturnsRequest

if TYPE_CHECKING:
    from density_model.config.config_schema import FeatureConfig

__all__ = ["YahooVolatilityPanelBuilder"]


class YahooVolatilityPanelBuilder:
    """
    Build a canonical panel DataFrame for return forecasting.

    Parameters
    ----------
    id_column : str, default="asset_id"
        Identifier column name used in the output panel.
    date_column : str, default="date"
        Date column name used in the output panel.
    return_column : str, default="return"
        Return column name used in the output panel.
    ticker_column : str, default="ticker"
        Ticker column name used as a static categorical feature.
    """

    def __init__(
        self,
        id_column: str = "asset_id",
        date_column: str = "date",
        return_column: str = "return",
        ticker_column: str = "ticker",
    ) -> None:
        self.id_column = id_column
        self.date_column = date_column
        self.return_column = return_column
        self.ticker_column = ticker_column

    def build_from_loader(
        self,
        *,
        loader: YahooDailyReturnsLoader,
        request: YahooDailyReturnsRequest,
        feature_config: FeatureConfig,
    ) -> pd.DataFrame:
        """
        Download returns and build the forecasting panel.

        Parameters
        ----------
        loader : YahooDailyReturnsLoader
            Yahoo Finance loader used to fetch calendar-normalized returns.
        request : YahooDailyReturnsRequest
            Download request validated by the data layer.
        feature_config : FeatureConfig
            Feature configuration defining the forecasting layout.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format panel with transformed returns and ticker.
        """

        returns = loader.load_returns(request=request)
        return self.build_from_returns(returns=returns, feature_config=feature_config)

    def build_feature_panel_from_loader(
        self,
        *,
        loader: YahooDailyReturnsLoader,
        request: YahooDailyReturnsRequest,
    ) -> pd.DataFrame:
        """
        Download returns and build a raw feature panel without training targets.

        Parameters
        ----------
        loader : YahooDailyReturnsLoader
            Yahoo Finance loader used to fetch calendar-normalized returns.
        request : YahooDailyReturnsRequest
            Download request validated by the data layer.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format feature panel containing raw returns and ticker.
        """

        returns = loader.load_returns(request=request)
        return self.build_feature_panel_from_returns(returns=returns)

    def build_from_returns(self, *, returns: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
        """
        Convert wide returns into a canonical long-format forecasting panel.

        Parameters
        ----------
        returns : pandas.DataFrame
            Calendar-normalized wide return matrix indexed by date.
        feature_config : FeatureConfig
            Feature configuration defining the return transformation.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format panel with returns and ticker.
        """

        return_frame = returns.stack(future_stack=True).rename(self.return_column).reset_index()
        return_frame.columns = [self.date_column, self.id_column, self.return_column]
        return_frame[self.ticker_column] = return_frame[self.id_column]
        return return_frame.sort_values([self.id_column, self.date_column]).reset_index(drop=True)

    def build_feature_panel_from_returns(self, *, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide returns into a canonical long-format feature panel.

        Parameters
        ----------
        returns : pandas.DataFrame
            Calendar-normalized wide return matrix indexed by date.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format feature panel with returns and ticker.
        """

        return_frame = returns.stack(future_stack=True).rename(self.return_column).reset_index()
        return_frame.columns = [self.date_column, self.id_column, self.return_column]
        return_frame[self.ticker_column] = return_frame[self.id_column]
        return return_frame.sort_values([self.id_column, self.date_column]).reset_index(drop=True)

    def transform_feature_panel(self, *, panel: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
        """
        Validate and normalize a canonical feature panel without transforming returns.

        Parameters
        ----------
        panel : pandas.DataFrame
            Canonical long-format feature panel containing returns.
        feature_config : FeatureConfig
            Feature configuration defining the forecasting layout.

        Returns
        -------
        pandas.DataFrame
            Canonical long-format panel with standard returns.
        """

        required_columns = {self.date_column, self.id_column, self.return_column}
        missing_columns = sorted(required_columns.difference(panel.columns))
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Feature panel is missing required columns: {missing_text}")
        _ = feature_config
        return panel.sort_values([self.id_column, self.date_column]).reset_index(drop=True)
