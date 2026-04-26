"""
Panel Dataset Configuration
---------------------------
Pydantic configuration models for panel time-series metadata, forecasting
windows, feature selection, and vectorized panel construction.

Classes
-------
TimeSeriesConfig
    Panel time-series metadata (date column, id column, coercion flags).

ForecastingConfig
    Forecasting-window configuration (targets, in_steps, horizon, out_steps).

FeatureInputConfig
    Feature selection for vectorized panel datasets (dynamic, dynamic-
    categorical, static-categorical).

VectorizedPanelConfig
    Vectorized panel assembly knobs (alignment mode, pad value, inclusion
    flags, dropna).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from density_model.shared.data.panel.inspectors import ForecastingInspector

__all__ = [
    "FeatureInputConfig",
    "ForecastingConfig",
    "TimeSeriesConfig",
    "VectorizedPanelConfig",
]


class TimeSeriesConfig(BaseModel):
    """Configuration for panel time-series metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    date_column: str
    id_column: str
    coerce_dates: bool = False
    sort: bool = True

    @field_validator("date_column", "id_column")
    @classmethod
    def validate_required_column_name(cls, value: str) -> str:
        """Validate a required column name."""

        if not value or value.strip() != value:
            raise ValueError(
                "Column names must be non-empty and must not include surrounding whitespace."
            )
        return value


class ForecastingConfig(BaseModel):
    """Configuration for forecasting-window construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    targets: tuple[str, ...]
    in_steps: int = Field(default=1, gt=0)
    horizon: int = Field(default=1, gt=0)
    out_steps: int = Field(default=1, gt=0)

    @field_validator("targets", mode="before")
    @classmethod
    def normalize_targets(cls, value: Any) -> tuple[str, ...]:
        """Normalize and validate target column names."""

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
                raise ValueError(
                    "Target names must be non-empty and must not include surrounding whitespace."
                )
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        if not normalized:
            raise ValueError("At least one target column is required.")
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_windows(self) -> ForecastingConfig:
        """Validate forecasting-window lengths."""

        ForecastingInspector.validate_window_lengths(
            in_steps=self.in_steps,
            horizon=self.horizon,
            out_steps=self.out_steps,
        )
        return self


class FeatureInputConfig(BaseModel):
    """Configuration for feature selection in vectorized panel datasets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dynamic_features: tuple[str, ...] | None = None
    dynamic_categorical_features: tuple[str, ...] | None = None
    static_categorical_features: tuple[str, ...] | None = None

    @field_validator(
        "dynamic_features",
        "dynamic_categorical_features",
        "static_categorical_features",
        mode="before",
    )
    @classmethod
    def normalize_feature_names(cls, value: Any) -> tuple[str, ...] | None:
        """Normalize and validate feature name sequences."""

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
                raise ValueError(
                    "Feature names must be non-empty and must not include surrounding whitespace."
                )
            if item not in seen:
                normalized.append(item)
                seen.add(item)
        return tuple(normalized) if normalized else None


class VectorizedPanelConfig(BaseModel):
    """Configuration for vectorized panel assembly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    align: str = Field(default="truncate", pattern="^(truncate|pad)$")
    pad_value: float | int = float("nan")
    include_dynamic_categorical: bool = True
    include_static_categorical: bool = True
    dropna: bool = False
    target_mode: str = Field(default="tail", pattern="^(tail|next_step)$")
