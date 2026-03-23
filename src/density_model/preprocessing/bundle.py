"""
Preprocessing Bundle
--------------------
Saveable container for fitted preprocessing artifacts applied together in the
forecasting pipeline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from density_model.preprocessing.base import BasePreprocessorArtifact
from density_model.preprocessing.scalers import StandardScaler
from density_model.preprocessing.tokenizers import VocabularyTokenizer

__all__ = ["PreprocessingBundle"]


class PreprocessingBundle(BasePreprocessorArtifact):
    """
    Saveable bundle of fitted preprocessing artifacts.

    Parameters
    ----------
    continuous_scaler : StandardScaler or None, default=None
        Continuous-feature scaler.
    categorical_tokenizer : VocabularyTokenizer or None, default=None
        Categorical tokenizer.
    continuous_columns : tuple of str or None, default=None
        Continuous feature columns transformed by the scaler.
    categorical_columns : tuple of str or None, default=None
        Categorical feature columns transformed by the tokenizer.
    """

    artifact_type = "preprocessing_bundle"

    def __init__(
        self,
        *,
        continuous_scaler: StandardScaler | None = None,
        categorical_tokenizer: VocabularyTokenizer | None = None,
        continuous_columns: tuple[str, ...] | None = None,
        categorical_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.continuous_scaler = continuous_scaler
        self.categorical_tokenizer = categorical_tokenizer
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns

    def fit(
        self,
        panel: pd.DataFrame,
        *,
        continuous_columns: tuple[str, ...] | None = None,
        categorical_columns: tuple[str, ...] | None = None,
    ) -> PreprocessingBundle:
        """
        Fit all configured preprocessing artifacts on a panel DataFrame.

        Parameters
        ----------
        panel : pandas.DataFrame
            Long-format panel DataFrame.
        continuous_columns : tuple of str or None, default=None
            Continuous feature columns to scale.
        categorical_columns : tuple of str or None, default=None
            Categorical feature columns to tokenize.

        Returns
        -------
        PreprocessingBundle
            Fitted preprocessing bundle.
        """

        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        if continuous_columns:
            scaler_input = panel.loc[:, ["asset_id", *continuous_columns]].copy()
            self.continuous_scaler = StandardScaler().fit(scaler_input)
        if categorical_columns:
            tokenizer_input = panel.loc[:, list(categorical_columns)].copy()
            self.categorical_tokenizer = VocabularyTokenizer().fit(tokenizer_input)
        return self

    def transform(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing artifacts to a panel DataFrame.

        Parameters
        ----------
        panel : pandas.DataFrame
            Long-format panel DataFrame.

        Returns
        -------
        pandas.DataFrame
            Transformed panel DataFrame.
        """

        transformed = panel.copy()
        if self.continuous_scaler is not None and self.continuous_columns:
            scaler_input = transformed.loc[:, ["asset_id", *self.continuous_columns]].copy()
            scaled = self.continuous_scaler.transform(scaler_input)
            transformed.loc[:, list(self.continuous_columns)] = scaled.loc[:, list(self.continuous_columns)]
        if self.categorical_tokenizer is not None and self.categorical_columns:
            tokenized = self.categorical_tokenizer.transform(transformed.loc[:, list(self.categorical_columns)])
            tokenized = tokenized.astype("int64")
            for column in self.categorical_columns:
                transformed[column] = tokenized[column]
        return transformed

    def _serialize_state(self) -> dict[str, Any]:
        """
        Serialize the preprocessing bundle state.

        Returns
        -------
        dict of str to Any
            Serialized preprocessing bundle state.
        """

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
        }

    @classmethod
    def _deserialize_state(cls, state: dict[str, Any]) -> PreprocessingBundle:
        """
        Reconstruct a preprocessing bundle from serialized state.

        Parameters
        ----------
        state : dict of str to Any
            Serialized preprocessing bundle state.

        Returns
        -------
        PreprocessingBundle
            Reconstructed preprocessing bundle.
        """

        continuous_scaler = None
        if state["continuous_scaler"] is not None:
            continuous_scaler = StandardScaler._deserialize_state(state["continuous_scaler"]["state"])
        categorical_tokenizer = None
        if state["categorical_tokenizer"] is not None:
            categorical_tokenizer = VocabularyTokenizer._deserialize_state(state["categorical_tokenizer"]["state"])
        return cls(
            continuous_scaler=continuous_scaler,
            categorical_tokenizer=categorical_tokenizer,
            continuous_columns=tuple(state["continuous_columns"]) or None,
            categorical_columns=tuple(state["categorical_columns"]) or None,
        )
