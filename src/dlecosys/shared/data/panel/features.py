"""
Panel Features
--------------
Feature planning and resolution utilities for vectorized panel forecasting
datasets.

Classes
-------
FeatureGroups
    Frozen dataclass holding resolved dynamic, dynamic-categorical,
    static-categorical, and target feature name groups.

FeaturePlanner
    Derive default dynamic features from a canonical panel DataFrame.

FeatureResolver
    Resolve declared feature groups against a canonical panel DataFrame,
    raising on missing columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = ["FeatureGroups", "FeaturePlanner", "FeatureResolver"]


@dataclass(frozen=True, slots=True)
class FeatureGroups:
    """Resolved feature groups used during dataset assembly."""

    dynamic: tuple[str, ...]
    dynamic_categorical: tuple[str, ...]
    static_categorical: tuple[str, ...]
    targets: tuple[str, ...]


class FeaturePlanner:
    """Derive default dynamic features from a canonical panel DataFrame."""

    @staticmethod
    def plan_dynamic_features(
        df: pd.DataFrame,
        *,
        date_column: str,
        id_column: str,
        targets: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Derive default dynamic features from numeric columns."""

        excluded = {date_column, id_column}
        numeric_columns = [
            column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])
        ]
        target_columns = [column for column in targets if column in numeric_columns]
        remaining_columns = [
            column
            for column in numeric_columns
            if column not in excluded and column not in targets
        ]
        planned = tuple(target_columns + remaining_columns)
        if not planned:
            raise ValueError(
                "No numeric columns are available to build dynamic forecasting features."
            )
        return planned


class FeatureResolver:
    """Resolve declared feature groups against a canonical panel DataFrame."""

    def __init__(
        self,
        *,
        date_column: str,
        id_column: str,
        dynamic_features: tuple[str, ...] | None,
        dynamic_categorical_features: tuple[str, ...] | None,
        static_categorical_features: tuple[str, ...] | None,
        targets: tuple[str, ...],
    ) -> None:
        self.date_column = date_column
        self.id_column = id_column
        self.dynamic_features = dynamic_features
        self.dynamic_categorical_features = dynamic_categorical_features
        self.static_categorical_features = static_categorical_features
        self.targets = targets

    def resolve(self, df: pd.DataFrame) -> FeatureGroups:
        """Resolve declared feature groups against the provided DataFrame."""

        for target in self.targets:
            if target not in df.columns:
                raise ValueError(f"Target column {target!r} was not found in the input data.")

        dynamic = (
            self.dynamic_features
            if self.dynamic_features is not None
            else FeaturePlanner.plan_dynamic_features(
                df,
                date_column=self.date_column,
                id_column=self.id_column,
                targets=self.targets,
            )
        )
        dynamic_categorical = self.dynamic_categorical_features or tuple()
        static_categorical = self.static_categorical_features or tuple()

        for name, columns in {
            "dynamic_features": dynamic,
            "dynamic_categorical_features": dynamic_categorical,
            "static_categorical_features": static_categorical,
        }.items():
            missing = [column for column in columns if column not in df.columns]
            if missing:
                missing_text = ", ".join(missing)
                raise ValueError(
                    f"{name} contains columns not present in the input data: {missing_text}"
                )

        return FeatureGroups(
            dynamic=tuple(dynamic),
            dynamic_categorical=tuple(dynamic_categorical),
            static_categorical=tuple(static_categorical),
            targets=tuple(self.targets),
        )
