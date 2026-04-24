"""
Panel Preprocessing Bundle
--------------------------
Saveable container for fitted panel preprocessing artifacts applied together
in the forecasting pipeline.

Classes
-------
PreprocessingBundle
    Saveable bundle composing a per-id ``StandardScaler`` and a
    ``VocabularyTokenizer``, applied together against a long-format panel
    DataFrame.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from density_model.shared.preprocessing.panel.base import BasePreprocessorArtifact
from density_model.shared.preprocessing.panel.scalers import StandardScaler
from density_model.shared.preprocessing.panel.tokenizers import VocabularyTokenizer

__all__ = ["PreprocessingBundle"]


class PreprocessingBundle(BasePreprocessorArtifact):
    """Saveable bundle of fitted preprocessing artifacts."""

    artifact_type = "preprocessing_bundle"

    def __init__(
        self,
        *,
        continuous_scaler: StandardScaler | None = None,
        categorical_tokenizer: VocabularyTokenizer | None = None,
        continuous_columns: tuple[str, ...] | None = None,
        categorical_columns: tuple[str, ...] | None = None,
        id_column: str = "asset_id",
    ) -> None:
        self.continuous_scaler = continuous_scaler
        self.categorical_tokenizer = categorical_tokenizer
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.id_column = id_column

    def fit(
        self,
        panel: pd.DataFrame,
        *,
        continuous_columns: tuple[str, ...] | None = None,
        categorical_columns: tuple[str, ...] | None = None,
    ) -> PreprocessingBundle:
        """Fit all configured preprocessing artifacts on a panel DataFrame."""

        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        if continuous_columns:
            scaler_input = panel.loc[:, [self.id_column, *continuous_columns]].copy()
            self.continuous_scaler = StandardScaler(id_column=self.id_column).fit(scaler_input)
        if categorical_columns:
            tokenizer_input = panel.loc[:, list(categorical_columns)].copy()
            self.categorical_tokenizer = VocabularyTokenizer().fit(tokenizer_input)
        return self

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted preprocessing artifacts to a panel DataFrame."""

        transformed = panel.copy()
        if self.continuous_scaler is not None and self.continuous_columns:
            scaler_input = transformed.loc[:, [self.id_column, *self.continuous_columns]].copy()
            scaled = self.continuous_scaler.transform(scaler_input)
            transformed.loc[:, list(self.continuous_columns)] = scaled.loc[
                :, list(self.continuous_columns)
            ]
        if self.categorical_tokenizer is not None and self.categorical_columns:
            tokenized = self.categorical_tokenizer.transform(
                transformed.loc[:, list(self.categorical_columns)]
            )
            tokenized = tokenized.astype("int64")
            for column in self.categorical_columns:
                transformed[column] = tokenized[column]
        return transformed

    def _serialize_state(self) -> dict[str, Any]:
        return {
            "continuous_scaler": None
            if self.continuous_scaler is None
            else {
                "artifact_type": self.continuous_scaler.artifact_type,
                "state": self.continuous_scaler._serialize_state(),
            },
            "categorical_tokenizer": None
            if self.categorical_tokenizer is None
            else {
                "artifact_type": self.categorical_tokenizer.artifact_type,
                "state": self.categorical_tokenizer._serialize_state(),
            },
            "continuous_columns": list(self.continuous_columns or ()),
            "categorical_columns": list(self.categorical_columns or ()),
            "id_column": self.id_column,
        }

    @classmethod
    def _deserialize_state(cls, state: dict[str, Any]) -> PreprocessingBundle:
        continuous_scaler = None
        if state["continuous_scaler"] is not None:
            continuous_scaler = StandardScaler._deserialize_state(
                state["continuous_scaler"]["state"]
            )
        categorical_tokenizer = None
        if state["categorical_tokenizer"] is not None:
            categorical_tokenizer = VocabularyTokenizer._deserialize_state(
                state["categorical_tokenizer"]["state"]
            )
        return cls(
            continuous_scaler=continuous_scaler,
            categorical_tokenizer=categorical_tokenizer,
            continuous_columns=tuple(state["continuous_columns"]) or None,
            categorical_columns=tuple(state["categorical_columns"]) or None,
            id_column=state.get("id_column", "asset_id"),
        )
