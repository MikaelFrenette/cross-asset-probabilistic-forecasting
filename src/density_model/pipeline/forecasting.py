"""
Forecasting Pipeline
--------------------
Chronological data pipeline for Yahoo-based return forecasting through panel
construction, preprocessing, VPD assembly, and optional torch dataset wrapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from density_model.config.config_schema import FeatureConfig
from density_model.data import YahooDailyReturnsLoader, YahooDailyReturnsRequest, YahooVolatilityPanelBuilder
from density_model.datasets import VectorizedPanelDataset, VectorizedPanelConfig
from density_model.preprocessing.bundle import PreprocessingBundle

__all__ = ["ChronologicalSplitConfig", "ForecastingPipeline", "ForecastingPipelineOutput"]


class ChronologicalSplitConfig(BaseModel):
    """
    Chronological split configuration for forecasting datasets.

    Parameters
    ----------
    validation_start_date : str
        First forecast date assigned to the validation split.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    validation_start_date: str


@dataclass(frozen=True, slots=True)
class ForecastingPipelineOutput:
    """
    Output container for the forecasting data pipeline.

    Parameters
    ----------
    panel : pandas.DataFrame
        Full transformed long-format panel built with train-fitted preprocessing.
    preprocessing : PreprocessingBundle
        Preprocessing bundle fitted on the chronological training panel only.
    train_panel_data : dict of str to Any
        Training split vectorized panel output.
    validation_panel_data : dict of str to Any
        Validation split vectorized panel output.
    """

    panel: pd.DataFrame
    preprocessing: PreprocessingBundle
    train_panel_data: dict[str, Any]
    validation_panel_data: dict[str, Any]


class ForecastingPipeline:
    """
    Chronological forecasting pipeline for model-ready panel construction.

    Parameters
    ----------
    loader : YahooDailyReturnsLoader
        Loader used to fetch calendar-normalized Yahoo returns.
    panel_builder : YahooVolatilityPanelBuilder or None, default=None
        Long-format panel builder.
    """

    def __init__(
        self,
        loader: YahooDailyReturnsLoader,
        panel_builder: YahooVolatilityPanelBuilder | None = None,
    ) -> None:
        self.loader = loader
        self.panel_builder = panel_builder or YahooVolatilityPanelBuilder()

    def run(
        self,
        *,
        request: YahooDailyReturnsRequest,
        feature_config: FeatureConfig,
        split_config: ChronologicalSplitConfig,
        panel_config: VectorizedPanelConfig | None = None,
    ) -> ForecastingPipelineOutput:
        """
        Run the full chronological preprocessing pipeline.

        Parameters
        ----------
        request : YahooDailyReturnsRequest
            Yahoo Finance request.
        feature_config : FeatureConfig
            Feature configuration for return transformation and sequence lengths.
        split_config : ChronologicalSplitConfig
            Chronological validation split configuration.
        panel_config : VectorizedPanelConfig or None, default=None
            Vectorized panel assembly configuration.

        Returns
        -------
        ForecastingPipelineOutput
            Model-ready chronological pipeline outputs.
        """

        continuous_columns = feature_config.streams.resolve_continuous_columns()
        categorical_columns = feature_config.streams.resolve_dynamic_categorical_columns()
        static_categorical_columns = feature_config.streams.resolve_static_categorical_columns()
        raw_panel = self.panel_builder.build_from_loader(
            loader=self.loader,
            request=request,
            feature_config=feature_config,
        )
        train_panel, _ = self._split_panel(
            raw_panel,
            validation_start_date=split_config.validation_start_date,
        )
        preprocessing = PreprocessingBundle().fit(
            train_panel,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns + static_categorical_columns,
        )
        transformed_panel = preprocessing.transform(raw_panel)
        full_vpd = self._build_vpd(
            transformed_panel,
            feature_config=feature_config,
            panel_config=panel_config,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            static_categorical_columns=static_categorical_columns,
        )
        train_vpd, validation_vpd = self._split_panel_data(
            full_vpd,
            validation_start_date=split_config.validation_start_date,
        )
        return ForecastingPipelineOutput(
            panel=transformed_panel,
            preprocessing=preprocessing,
            train_panel_data=train_vpd,
            validation_panel_data=validation_vpd,
        )

    def build_torch_datasets(self, output: ForecastingPipelineOutput) -> tuple[Any, Any]:
        """
        Wrap vectorized panel outputs in torch dataset blueprints.

        Parameters
        ----------
        output : ForecastingPipelineOutput
            Pipeline output containing vectorized panel data.

        Returns
        -------
        tuple
            Training and validation torch dataset wrappers.
        """

        from density_model.torch_datasets import VectorizedPanelTorchDataset

        return (
            VectorizedPanelTorchDataset(output.train_panel_data),
            VectorizedPanelTorchDataset(output.validation_panel_data),
        )

    def _split_panel(self, panel: pd.DataFrame, *, validation_start_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a panel chronologically by forecast date.

        Parameters
        ----------
        panel : pandas.DataFrame
            Long-format panel.
        validation_start_date : str
            First validation date in ``YYYY-MM-DD`` format.

        Returns
        -------
        tuple of pandas.DataFrame and pandas.DataFrame
            Training and validation panel splits.
        """

        validation_boundary = pd.Timestamp(validation_start_date)
        train_panel = panel.loc[panel["date"] < validation_boundary].copy()
        validation_panel = panel.loc[panel["date"] >= validation_boundary].copy()
        if train_panel.empty:
            raise ValueError("Chronological split produced an empty training panel.")
        if validation_panel.empty:
            raise ValueError("Chronological split produced an empty validation panel.")
        return train_panel, validation_panel

    def _build_vpd(
        self,
        panel: pd.DataFrame,
        *,
        feature_config: FeatureConfig,
        panel_config: VectorizedPanelConfig | None,
        continuous_columns: tuple[str, ...],
        categorical_columns: tuple[str, ...] | None,
        static_categorical_columns: tuple[str, ...] | None,
    ) -> dict[str, Any]:
        """
        Build vectorized panel data from a transformed long-format panel.

        Parameters
        ----------
        panel : pandas.DataFrame
            Long-format panel.
        feature_config : FeatureConfig
            Sequence and target configuration.
        panel_config : VectorizedPanelConfig or None
            Vectorized panel assembly configuration.
        continuous_columns : tuple of str
            Continuous input feature columns.
        categorical_columns : tuple of str or None
            Dynamic categorical columns.
        static_categorical_columns : tuple of str or None
            Static categorical columns.

        Returns
        -------
        dict of str to Any
            Vectorized panel output.
        """

        dataset = VectorizedPanelDataset(
            data=panel,
            date_column="date",
            id_column="asset_id",
            targets=feature_config.target_column,
            in_steps=feature_config.sequence_length,
            horizon=feature_config.forecast_horizon,
            out_steps=1,
            dynamic_features=continuous_columns,
            dynamic_categorical_features=categorical_columns,
            static_categorical_features=static_categorical_columns,
            panel_config=panel_config,
        )
        return dataset.generate_sequences()

    def _split_panel_data(
        self,
        panel_data: dict[str, Any],
        *,
        validation_start_date: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Split vectorized panel samples by forecast start date.

        Parameters
        ----------
        panel_data : dict of str to Any
            Vectorized panel output for the full transformed panel.
        validation_start_date : str
            First validation forecast date in ``YYYY-MM-DD`` format.

        Returns
        -------
        tuple of dict of str to Any and dict of str to Any
            Training and validation vectorized panel outputs.
        """

        forecast_start_dates = pd.to_datetime(panel_data["forecast_start_dates"])
        validation_boundary = pd.Timestamp(validation_start_date)
        train_mask = forecast_start_dates < validation_boundary
        validation_mask = forecast_start_dates >= validation_boundary
        if not train_mask.any():
            raise ValueError("Chronological split produced an empty training panel dataset.")
        if not validation_mask.any():
            raise ValueError("Chronological split produced an empty validation panel dataset.")
        return (
            self._slice_panel_data(panel_data, train_mask),
            self._slice_panel_data(panel_data, validation_mask),
        )

    def _slice_panel_data(
        self,
        panel_data: dict[str, Any],
        sample_mask: pd.Series | Any,
    ) -> dict[str, Any]:
        """
        Slice vectorized panel outputs along the forecast-date axis.

        Parameters
        ----------
        panel_data : dict of str to Any
            Vectorized panel output to slice.
        sample_mask : array-like
            Boolean mask over the leading sample axis.

        Returns
        -------
        dict of str to Any
            Sliced vectorized panel output.
        """

        sample_mask = pd.Index(sample_mask).to_numpy(dtype=bool, copy=False)
        sliced: dict[str, Any] = {}
        for key, value in panel_data.items():
            if key == "id_index":
                sliced[key] = list(value)
                continue
            if value is None:
                sliced[key] = None
                continue
            sliced[key] = value[sample_mask]
        return sliced
