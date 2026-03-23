"""
Vectorized Panel Dataset
------------------------
Transform a panel time-series DataFrame into forecast-aligned tensors shaped
``(B, ID, Seq, Dim)`` for inputs and ``(B, ID, Out, TargetDim)`` for targets.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from density_model.datasets.base import BaseTimeSeriesDataset
from density_model.datasets.config import FeatureInputConfig, ForecastingConfig, VectorizedPanelConfig
from density_model.datasets.features import FeatureGroups, FeatureResolver
from density_model.datasets.inspectors import TimeSeriesInspector

__all__ = ["VectorizedPanelDataset"]


class VectorizedPanelDataset(BaseTimeSeriesDataset):
    """
    Build forecast-aligned panel tensors from a canonical panel DataFrame.

    Parameters
    ----------
    data : Any
        Panel DataFrame or Series.
    date_column : str
        Date column name.
    id_column : str
        Entity identifier column name.
    targets : str or sequence of str
        Target column names to predict.
    in_steps : int, default=1
        Number of historical input steps.
    horizon : int, default=1
        Positive forecasting horizon.
    out_steps : int, default=1
        Number of target steps to predict.
    coerce_dates : bool, default=False
        Whether to coerce the date column to datetime.
    sort : bool, default=True
        Whether to sort by identifier and date.
        dynamic_features : sequence of str or None, default=None
        Continuous input features. When omitted, numeric columns are derived automatically.
    dynamic_categorical_features : sequence of str or None, default=None
        Dynamic categorical input features.
    static_categorical_features : sequence of str or None, default=None
        Static categorical input features. These must be constant within each identifier.
    panel_config : VectorizedPanelConfig or None, default=None
        Panel assembly configuration.
    """

    __slots__ = ("feature_cfg", "forecasting_cfg", "panel_cfg", "resolver")

    def __init__(
        self,
        *,
        data: Any,
        date_column: str,
        id_column: str,
        targets: str | tuple[str, ...] | list[str],
        in_steps: int = 1,
        horizon: int = 1,
        out_steps: int = 1,
        coerce_dates: bool = False,
        sort: bool = True,
        dynamic_features: tuple[str, ...] | list[str] | str | None = None,
        dynamic_categorical_features: tuple[str, ...] | list[str] | str | None = None,
        static_categorical_features: tuple[str, ...] | list[str] | str | None = None,
        panel_config: VectorizedPanelConfig | None = None,
    ) -> None:
        super().__init__(
            data=data,
            date_column=date_column,
            id_column=id_column,
            coerce_dates=coerce_dates,
            sort=sort,
        )
        self.forecasting_cfg = ForecastingConfig(
            targets=targets,
            in_steps=in_steps,
            horizon=horizon,
            out_steps=out_steps,
        )
        self.feature_cfg = FeatureInputConfig(
            dynamic_features=dynamic_features,
            dynamic_categorical_features=dynamic_categorical_features,
            static_categorical_features=static_categorical_features,
        )
        self.panel_cfg = panel_config or VectorizedPanelConfig()
        self.resolver = FeatureResolver(
            date_column=self.date_column,
            id_column=self.id_column,
            dynamic_features=self.feature_cfg.dynamic_features,
            dynamic_categorical_features=self.feature_cfg.dynamic_categorical_features,
            static_categorical_features=self.feature_cfg.static_categorical_features,
            targets=self.forecasting_cfg.targets,
        )
        if self.feature_cfg.static_categorical_features:
            TimeSeriesInspector.validate_static_features(
                self.data,
                id_column=self.id_column,
                static_features=self.feature_cfg.static_categorical_features,
            )

    @property
    def in_steps(self) -> int:
        """
        Return the number of historical input steps.

        Returns
        -------
        int
            Number of historical input steps.
        """

        return self.forecasting_cfg.in_steps

    @property
    def horizon(self) -> int:
        """
        Return the forecasting horizon.

        Returns
        -------
        int
            Positive forecasting horizon.
        """

        return self.forecasting_cfg.horizon

    @property
    def out_steps(self) -> int:
        """
        Return the number of output target steps.

        Returns
        -------
        int
            Number of target steps to predict.
        """

        return self.forecasting_cfg.out_steps

    @property
    def targets(self) -> tuple[str, ...]:
        """
        Return the configured target columns.

        Returns
        -------
        tuple of str
            Target column names.
        """

        return self.forecasting_cfg.targets

    def generate_sequences(self) -> dict[str, Any]:
        """
        Generate forecast-aligned panel tensors.

        Returns
        -------
        dict of str to Any
            Mapping containing ``X_continuous``, ``X_cat_continuous``,
            ``X_cat_static``, their masks, ``y``, ``y_mask``, ``id_index``,
            ``forecast_start_dates``, and ``forecast_end_dates``.
        """

        features = self.resolver.resolve(self.data)
        self._validate_forecasting_contract(features)
        per_id_samples: "OrderedDict[Any, dict[pd.Timestamp, dict[str, Any]]]" = OrderedDict()
        id_order: list[Any] = []

        for entity_id, group in self.data.groupby(self.id_column, sort=False):
            samples = self._build_group_samples(group=group, features=features)
            if samples:
                per_id_samples[entity_id] = samples
                id_order.append(entity_id)

        if not per_id_samples:
            return {
                "X_continuous": None,
                "X_continuous_mask": None,
                "X_cat_continuous": None,
                "X_cat_continuous_mask": None,
                "X_cat_static": None,
                "X_cat_static_mask": None,
                "y": None,
                "y_mask": None,
                "id_index": [],
                "forecast_start_dates": np.array([], dtype="datetime64[ns]"),
                "forecast_end_dates": np.array([], dtype="datetime64[ns]"),
            }

        forecast_dates = self._aligned_forecast_dates(per_id_samples)
        forecast_end_dates = self._aligned_forecast_end_dates(per_id_samples, forecast_dates)
        x_continuous, x_continuous_mask = self._stack_feature(per_id_samples, id_order, forecast_dates, "X_continuous")
        x_cat_continuous, x_cat_continuous_mask = self._stack_feature(
            per_id_samples,
            id_order,
            forecast_dates,
            "X_cat_continuous",
        )
        x_cat_static, x_cat_static_mask = self._stack_feature(
            per_id_samples,
            id_order,
            forecast_dates,
            "X_cat_static",
        )
        y, y_mask = self._stack_feature(per_id_samples, id_order, forecast_dates, "y")
        result = {
            "X_continuous": x_continuous,
            "X_continuous_mask": x_continuous_mask,
            "X_cat_continuous": x_cat_continuous,
            "X_cat_continuous_mask": x_cat_continuous_mask,
            "X_cat_static": x_cat_static,
            "X_cat_static_mask": x_cat_static_mask,
            "y": y,
            "y_mask": y_mask,
            "id_index": id_order,
            "forecast_start_dates": np.array(forecast_dates, dtype="datetime64[ns]"),
            "forecast_end_dates": np.array(forecast_end_dates, dtype="datetime64[ns]"),
        }
        return self._drop_missing_dates(result)

    def _build_group_samples(
        self,
        *,
        group: pd.DataFrame,
        features: FeatureGroups,
    ) -> dict[pd.Timestamp, dict[str, Any]]:
        """
        Build per-date samples for a single identifier group.

        Parameters
        ----------
        group : pandas.DataFrame
            Identifier-specific subframe sorted by date.
        features : FeatureGroups
            Resolved feature groups.

        Returns
        -------
        dict
            Mapping from forecast date to feature and target arrays.
        """

        sample_count = len(group) - (self.in_steps + self.horizon + self.out_steps - 1) + 1
        if sample_count <= 0:
            return {}

        samples: dict[pd.Timestamp, dict[str, Any]] = {}
        target_start_offset = self.in_steps + self.horizon - 1
        static_vector = None
        if features.static_categorical:
            static_vector = group.loc[:, list(features.static_categorical)].iloc[0].to_numpy(copy=True)

        for start_index in range(sample_count):
            input_end = start_index + self.in_steps - 1
            target_start = start_index + target_start_offset
            target_end = target_start + self.out_steps
            forecast_start_date = pd.Timestamp(group.iloc[target_start][self.date_column])
            forecast_end_date = pd.Timestamp(group.iloc[target_end - 1][self.date_column])
            if target_start <= input_end:
                raise ValueError("Forecast leakage detected: target window overlaps the input window.")
            if target_start != input_end + self.horizon:
                raise ValueError("Forecast leakage detected: first target does not start at t + horizon.")

            sample: dict[str, Any] = {
                "forecast_start_date": forecast_start_date,
                "forecast_end_date": forecast_end_date,
            }

            if features.dynamic:
                sample["X_continuous"] = group.loc[
                    group.index[start_index : start_index + self.in_steps],
                    list(features.dynamic),
                ].to_numpy(copy=True)
            if features.dynamic_categorical and self.panel_cfg.include_dynamic_categorical:
                sample["X_cat_continuous"] = group.loc[
                    group.index[start_index : start_index + self.in_steps],
                    list(features.dynamic_categorical),
                ].to_numpy(copy=True)
            if static_vector is not None and self.panel_cfg.include_static_categorical:
                sample["X_cat_static"] = np.broadcast_to(
                    static_vector[None, :],
                    (self.in_steps, static_vector.shape[0]),
                ).copy()
            sample["y"] = group.loc[
                group.index[target_start:target_end],
                list(features.targets),
            ].to_numpy(copy=True)
            samples[forecast_start_date] = sample

        return samples

    def _aligned_forecast_dates(
        self,
        per_id_samples: OrderedDict[Any, dict[pd.Timestamp, dict[str, Any]]],
    ) -> list[pd.Timestamp]:
        """
        Align forecast dates across identifiers according to the panel configuration.

        Parameters
        ----------
        per_id_samples : OrderedDict
            Per-identifier sample mapping keyed by forecast date.

        Returns
        -------
        list of pandas.Timestamp
            Aligned forecast dates.
        """

        date_sets = [set(samples.keys()) for samples in per_id_samples.values()]
        if self.panel_cfg.align == "truncate":
            aligned = set.intersection(*date_sets)
        else:
            aligned = set.union(*date_sets)
        return sorted(aligned)

    def _aligned_forecast_end_dates(
        self,
        per_id_samples: OrderedDict[Any, dict[pd.Timestamp, dict[str, Any]]],
        forecast_dates: list[pd.Timestamp],
    ) -> list[pd.Timestamp]:
        """
        Resolve aligned forecast end dates for each forecast start date.

        Parameters
        ----------
        per_id_samples : OrderedDict
            Per-identifier sample mapping keyed by forecast start date.
        forecast_dates : list of pandas.Timestamp
            Aligned forecast start dates.

        Returns
        -------
        list of pandas.Timestamp
            Forecast end dates aligned to ``forecast_dates``.
        """

        end_dates: list[pd.Timestamp] = []
        for forecast_date in forecast_dates:
            resolved_end_date = None
            for samples in per_id_samples.values():
                sample = samples.get(forecast_date)
                if sample is not None:
                    resolved_end_date = sample["forecast_end_date"]
                    break
            if resolved_end_date is None:
                raise ValueError("Unable to resolve forecast end date for an aligned forecast start date.")
            end_dates.append(pd.Timestamp(resolved_end_date))
        return end_dates

    def _stack_feature(
        self,
        per_id_samples: OrderedDict[Any, dict[pd.Timestamp, dict[str, Any]]],
        id_order: list[Any],
        forecast_dates: list[pd.Timestamp],
        feature_name: str,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Stack a named feature across forecast dates and identifiers.

        Parameters
        ----------
        per_id_samples : OrderedDict
            Per-identifier sample mapping.
        id_order : list of str
            Identifier order along the panel axis.
        forecast_dates : list of pandas.Timestamp
            Forecast dates aligned across identifiers.
        feature_name : str
            Feature name to stack.

        Returns
        -------
        tuple of numpy.ndarray or None and numpy.ndarray or None
            Value tensor and feature-level observation mask.
        """

        prototype = self._prototype_array(per_id_samples, feature_name)
        if prototype is None:
            return None, None
        if not forecast_dates:
            empty_values = np.empty((0, len(id_order)) + prototype.shape, dtype=prototype.dtype)
            empty_mask = np.empty((0, len(id_order)) + prototype.shape, dtype=bool)
            return empty_values, empty_mask

        output = np.full(
            (len(forecast_dates), len(id_order)) + prototype.shape,
            fill_value=self._coerce_pad_value(self.panel_cfg.pad_value, prototype.dtype),
            dtype=prototype.dtype,
        )
        output_mask = np.zeros((len(forecast_dates), len(id_order)) + prototype.shape, dtype=bool)

        for date_index, forecast_date in enumerate(forecast_dates):
            for id_index, entity_id in enumerate(id_order):
                sample = per_id_samples[entity_id].get(forecast_date)
                if sample is None or feature_name not in sample:
                    if self.panel_cfg.align == "truncate":
                        raise ValueError("Truncate alignment produced a missing identifier-date sample.")
                    continue
                sample_array = sample[feature_name]
                output[date_index, id_index] = sample_array
                output_mask[date_index, id_index] = self._observed_mask(sample_array)
        return output, output_mask

    def _prototype_array(
        self,
        per_id_samples: OrderedDict[Any, dict[pd.Timestamp, dict[str, Any]]],
        feature_name: str,
    ) -> np.ndarray | None:
        """
        Find a prototype array for a named feature.

        Parameters
        ----------
        per_id_samples : OrderedDict
            Per-identifier sample mapping.
        feature_name : str
            Feature name to locate.

        Returns
        -------
        numpy.ndarray or None
            First matching feature array, or ``None`` when absent.
        """

        for samples in per_id_samples.values():
            for sample in samples.values():
                if feature_name in sample:
                    return sample[feature_name]
        return None

    def _validate_forecasting_contract(self, features: FeatureGroups) -> None:
        """
        Validate the forecasting contract enforced by the panel builder.

        Parameters
        ----------
        features : FeatureGroups
            Resolved feature groups used by the dataset.

        Returns
        -------
        None
            This method raises when the configuration risks leakage.

        Raises
        ------
        ValueError
            If the feature grouping violates the forecasting contract.
        """

        target_set = set(features.targets)
        if target_set.intersection(features.dynamic_categorical):
            raise ValueError("Target columns cannot be included in X_cat_continuous.")
        if target_set.intersection(features.static_categorical):
            raise ValueError("Target columns cannot be included in X_cat_static.")

    def _coerce_pad_value(self, pad_value: float | int, dtype: np.dtype) -> float | int | bool:
        """
        Coerce the configured pad value to a target dtype.

        Parameters
        ----------
        pad_value : float or int
            Configured pad value.
        dtype : numpy.dtype
            Target dtype.

        Returns
        -------
        float or int or bool
            Dtype-compatible pad value.
        """

        try:
            return np.array(pad_value, dtype=dtype).item()
        except Exception:
            if np.issubdtype(dtype, np.floating):
                return np.nan
            if np.issubdtype(dtype, np.integer):
                return -1
            if np.issubdtype(dtype, np.bool_):
                return False
            return 0

    def _observed_mask(self, array: np.ndarray) -> np.ndarray:
        """
        Build a feature-level observation mask for an array.

        Parameters
        ----------
        array : numpy.ndarray
            Feature or target array extracted from one sample.

        Returns
        -------
        numpy.ndarray
            Boolean mask with the same shape as ``array``.
        """

        if np.issubdtype(array.dtype, np.floating):
            return ~np.isnan(array)
        return np.ones(array.shape, dtype=bool)

    def _drop_missing_dates(self, result: dict[str, Any]) -> dict[str, Any]:
        """
        Drop forecast dates containing missing floating-point values when requested.

        Parameters
        ----------
        result : dict of str to Any
            Assembled dataset output.

        Returns
        -------
        dict of str to Any
            Filtered dataset output.
        """

        if not self.panel_cfg.dropna:
            return result

        mask = np.ones(len(result["forecast_start_dates"]), dtype=bool)
        for key in ("X_continuous_mask", "X_cat_continuous_mask", "X_cat_static_mask", "y_mask"):
            array_mask = result[key]
            if array_mask is not None:
                mask &= np.all(array_mask.reshape(array_mask.shape[0], -1), axis=1)

        result["forecast_start_dates"] = result["forecast_start_dates"][mask]
        result["forecast_end_dates"] = result["forecast_end_dates"][mask]
        for key in (
            "X_continuous",
            "X_continuous_mask",
            "X_cat_continuous",
            "X_cat_continuous_mask",
            "X_cat_static",
            "X_cat_static_mask",
            "y",
            "y_mask",
        ):
            if result[key] is not None:
                result[key] = result[key][mask]
        return result
