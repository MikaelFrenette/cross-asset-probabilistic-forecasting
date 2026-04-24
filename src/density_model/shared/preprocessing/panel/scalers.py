"""
Panel Scalers
-------------
Concrete scaling artifacts for continuous preprocessing in panel forecasting
pipelines. Statistics are fit per-identifier so each asset keeps its own mean
and scale, and leakage across the panel is avoided.

Classes
-------
StandardScaler
    Per-identifier, per-feature standard scaler for long-format panel data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from density_model.shared.preprocessing.panel.base import BaseScaler, decode_scalar, encode_scalar

__all__ = ["StandardScaler"]


class StandardScaler(BaseScaler):
    """Per-identifier, per-feature standard scaler for long-format panel data."""

    artifact_type = "standard_scaler"

    def __init__(
        self,
        *,
        id_column: str = "asset_id",
        feature_columns: tuple[str, ...] | None = None,
        mean_: dict[Any, dict[str, float]] | None = None,
        scale_: dict[Any, dict[str, float]] | None = None,
        variance_: dict[Any, dict[str, float]] | None = None,
    ) -> None:
        self.id_column = id_column
        self.feature_columns = feature_columns
        self.mean_ = mean_
        self.scale_ = scale_
        self.variance_ = variance_

    def fit(self, values: Any) -> StandardScaler:
        """Fit per-identifier scaling statistics on a long-format panel."""

        frame = self._to_frame(values)
        self.feature_columns = self._infer_feature_columns(frame)
        self.mean_ = {}
        self.variance_ = {}
        self.scale_ = {}
        for entity_id, group in frame.groupby(self.id_column, sort=False):
            self.mean_[entity_id] = {}
            self.variance_[entity_id] = {}
            self.scale_[entity_id] = {}
            for column in self.feature_columns:
                mean = float(group[column].mean(skipna=True))
                variance = float(group[column].var(skipna=True, ddof=0))
                scale = variance ** 0.5
                if scale == 0.0:
                    scale = 1.0
                self.mean_[entity_id][column] = mean
                self.variance_[entity_id][column] = variance
                self.scale_[entity_id][column] = scale
        return self

    def transform(self, values: Any) -> Any:
        """Standardize continuous features using per-identifier statistics."""

        self._check_is_fitted()
        frame = self._to_frame(values).copy()
        self._validate_frame(frame)
        for entity_id, group_index in frame.groupby(self.id_column, sort=False).groups.items():
            self._validate_entity_id(entity_id)
            for column in self.feature_columns or ():
                frame.loc[group_index, column] = (
                    frame.loc[group_index, column] - self.mean_[entity_id][column]
                ) / self.scale_[entity_id][column]
        return frame

    def inverse_transform(self, values: Any) -> Any:
        """Restore scaled features to the original per-identifier scale."""

        self._check_is_fitted()
        frame = self._to_frame(values).copy()
        self._validate_frame(frame)
        for entity_id, group_index in frame.groupby(self.id_column, sort=False).groups.items():
            self._validate_entity_id(entity_id)
            for column in self.feature_columns or ():
                frame.loc[group_index, column] = (
                    frame.loc[group_index, column] * self.scale_[entity_id][column]
                ) + self.mean_[entity_id][column]
        return frame

    def _serialize_state(self) -> dict[str, Any]:
        self._check_is_fitted()
        return {
            "id_column": self.id_column,
            "feature_columns": list(self.feature_columns or ()),
            "mean_": self._serialize_nested_stats(self.mean_),
            "scale_": self._serialize_nested_stats(self.scale_),
            "variance_": self._serialize_nested_stats(self.variance_),
        }

    @classmethod
    def _deserialize_state(cls, state: dict[str, Any]) -> StandardScaler:
        return cls(
            id_column=state["id_column"],
            feature_columns=tuple(state["feature_columns"]),
            mean_=cls._deserialize_nested_stats(state["mean_"]),
            scale_=cls._deserialize_nested_stats(state["scale_"]),
            variance_=cls._deserialize_nested_stats(state["variance_"]),
        )

    def _check_is_fitted(self) -> None:
        if (
            self.mean_ is None
            or self.scale_ is None
            or self.variance_ is None
            or self.feature_columns is None
        ):
            raise ValueError("StandardScaler must be fitted before use.")

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        if self.id_column not in frame.columns:
            raise ValueError(f"Input data must contain id column {self.id_column!r}.")
        for column in self.feature_columns or ():
            if column not in frame.columns:
                raise ValueError(f"Input data must contain fitted feature column {column!r}.")

    def _validate_entity_id(self, entity_id: Any) -> None:
        if entity_id not in (self.mean_ or {}):
            raise ValueError(f"Input data contains unseen identifier {entity_id!r}.")

    def _to_frame(self, values: Any) -> pd.DataFrame:
        if not isinstance(values, pd.DataFrame):
            raise TypeError("StandardScaler expects a pandas DataFrame in long panel format.")
        return values

    def _infer_feature_columns(self, frame: pd.DataFrame) -> tuple[str, ...]:
        excluded = {self.id_column, "date"}
        feature_columns = tuple(column for column in frame.columns if column not in excluded)
        if not feature_columns:
            raise ValueError("StandardScaler requires at least one continuous feature column.")
        return feature_columns

    def _serialize_nested_stats(
        self, stats: dict[Any, dict[str, float]] | None
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for entity_id, feature_stats in (stats or {}).items():
            serialized.append(
                {
                    "entity_id": encode_scalar(entity_id),
                    "stats": feature_stats,
                }
            )
        return serialized

    @staticmethod
    def _deserialize_nested_stats(
        payload: list[dict[str, Any]],
    ) -> dict[Any, dict[str, float]]:
        deserialized: dict[Any, dict[str, float]] = {}
        for item in payload:
            entity_id = decode_scalar(item["entity_id"])
            deserialized[entity_id] = {
                str(key): float(value) for key, value in item["stats"].items()
            }
        return deserialized
