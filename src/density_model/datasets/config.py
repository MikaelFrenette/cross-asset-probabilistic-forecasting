"""
Dataset Configuration
---------------------
Pydantic configuration models for time-series metadata, forecasting windows,
feature selection, and vectorized panel construction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from density_model.datasets.inspectors import ForecastingInspector

__all__ = [
    "FeatureInputConfig",
    "ForecastingConfig",
    "TimeSeriesConfig",
    "VectorizedPanelConfig",
]


class TimeSeriesConfig(BaseModel):
    """
    Configuration for panel time-series metadata.

    Parameters
    ----------
    date_column : str
        Date column name.
    id_column : str
        Entity identifier column name.
    coerce_dates : bool, default=False
        Whether to coerce the date column to datetime.
    sort : bool, default=True
        Whether to sort the data by identifier and date.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    date_column: str
    id_column: str
    coerce_dates: bool = False
    sort: bool = True

    @field_validator("date_column", "id_column")
    @classmethod
    def validate_required_column_name(cls, value: str) -> str:
        """
        Validate a required column name.

        Parameters
        ----------
        value : str
            Candidate column name.

        Returns
        -------
        str
            Validated column name.
        """

        if not value or value.strip() != value:
            raise ValueError("Column names must be non-empty and must not include surrounding whitespace.")
        return value


class ForecastingConfig(BaseModel):
    """
    Configuration for forecasting-window construction.

    Parameters
    ----------
    targets : tuple of str
        Target column names predicted by the dataset.
    in_steps : int
        Number of historical input steps.
    horizon : int
        Positive forecasting horizon.
    out_steps : int
        Number of target steps to predict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    targets: tuple[str, ...]
    in_steps: int = Field(default=1, gt=0)
    horizon: int = Field(default=1, gt=0)
    out_steps: int = Field(default=1, gt=0)

    @field_validator("targets", mode="before")
    @classmethod
    def normalize_targets(cls, value: Any) -> tuple[str, ...]:
        """
        Normalize and validate target column names.

        Parameters
        ----------
        value : Any
            Raw target specification.

        Returns
        -------
        tuple of str
            Validated target column names.
        """

        if isinstance(value, str):
            value = (value,)
        if not isinstance(value, (tuple, list)):
            raise TypeError("targets must be a string or a sequence of strings.")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise TypeError("targets must contain only strings.")
            if not item or item.strip() != item:
                raise ValueError("Target names must be non-empty and must not include surrounding whitespace.")
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        if not normalized:
            raise ValueError("At least one target column is required.")
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_windows(self) -> ForecastingConfig:
        """
        Validate forecasting-window lengths.

        Returns
        -------
        ForecastingConfig
            Validated forecasting configuration.
        """

        ForecastingInspector.validate_window_lengths(
            in_steps=self.in_steps,
            horizon=self.horizon,
            out_steps=self.out_steps,
        )
        return self


class FeatureInputConfig(BaseModel):
    """
    Configuration for feature selection in vectorized panel datasets.

    Parameters
    ----------
    dynamic_features : tuple of str or None, default=None
        Dynamic continuous features.
    dynamic_categorical_features : tuple of str or None, default=None
        Dynamic categorical features.
    static_categorical_features : tuple of str or None, default=None
        Static categorical features.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dynamic_features: tuple[str, ...] | None = None
    dynamic_categorical_features: tuple[str, ...] | None = None
    static_categorical_features: tuple[str, ...] | None = None

    @field_validator("dynamic_features", "dynamic_categorical_features", "static_categorical_features", mode="before")
    @classmethod
    def normalize_feature_names(cls, value: Any) -> tuple[str, ...] | None:
        """
        Normalize and validate feature name sequences.

        Parameters
        ----------
        value : Any
            Raw feature-name specification.

        Returns
        -------
        tuple of str or None
            Validated feature names, or ``None`` when unspecified.
        """

        if value is None:
            return None
        if isinstance(value, str):
            value = (value,)
        if not isinstance(value, (tuple, list)):
            raise TypeError("Feature lists must be strings, sequences of strings, or None.")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Feature names must be strings.")
            if not item or item.strip() != item:
                raise ValueError("Feature names must be non-empty and must not include surrounding whitespace.")
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        return tuple(normalized) if normalized else None


class VectorizedPanelConfig(BaseModel):
    """
    Configuration for vectorized panel assembly.

    Parameters
    ----------
    align : {"truncate", "pad"}, default="truncate"
        Alignment mode across identifiers. ``truncate`` keeps only shared forecast
        dates. ``pad`` keeps the union of forecast dates and pads missing entries.
    pad_value : float or int, default=nan
        Padding value used when ``align="pad"``.
    include_dynamic_categorical : bool, default=True
        Whether to include dynamic categorical features.
    include_static_categorical : bool, default=True
        Whether to include static categorical features.
    dropna : bool, default=False
        Whether to drop forecast dates containing missing floating-point values
        after assembly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    align: str = Field(default="truncate", pattern="^(truncate|pad)$")
    pad_value: float | int = float("nan")
    include_dynamic_categorical: bool = True
    include_static_categorical: bool = True
    dropna: bool = False
