"""
Tokenizers
----------
Concrete tokenization artifacts for categorical preprocessing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from density_model.preprocessing.base import BaseTokenizer, decode_scalar, encode_scalar

__all__ = ["VocabularyTokenizer"]


class VocabularyTokenizer(BaseTokenizer):
    """
    Per-column vocabulary tokenizer with reserved unknown and missing tokens.

    Parameters
    ----------
    column_names : tuple of str or None, default=None
        Categorical columns fitted by the tokenizer.
    vocabularies_ : dict or None, default=None
        Per-column token mapping.
    inverse_vocabularies_ : dict or None, default=None
        Per-column inverse token mapping.
    unknown_token : str, default="[UNK]"
        Reserved token for unseen values.
    missing_token : str, default="[MISSING]"
        Reserved token for missing values.
    """

    artifact_type = "vocabulary_tokenizer"

    def __init__(
        self,
        *,
        column_names: tuple[str, ...] | None = None,
        vocabularies_: dict[str, dict[str, int]] | None = None,
        inverse_vocabularies_: dict[str, dict[int, dict[str, Any]]] | None = None,
        unknown_token: str = "[UNK]",
        missing_token: str = "[MISSING]",
    ) -> None:
        self.column_names = column_names
        self.vocabularies_ = vocabularies_
        self.inverse_vocabularies_ = inverse_vocabularies_
        self.unknown_token = unknown_token
        self.missing_token = missing_token

    def fit(self, values: Any) -> VocabularyTokenizer:
        """
        Fit independent categorical vocabularies for each column.

        Parameters
        ----------
        values : Any
            Categorical pandas DataFrame.

        Returns
        -------
        VocabularyTokenizer
            Fitted tokenizer instance.
        """

        frame = self._to_frame(values)
        self.column_names = tuple(str(column) for column in frame.columns)
        self.vocabularies_ = {}
        self.inverse_vocabularies_ = {}
        for column in self.column_names:
            normalized = self._normalize_series(frame[column])
            vocabulary = {
                self.missing_token: 0,
                self.unknown_token: 1,
            }
            inverse_vocabulary = {
                0: encode_scalar(None),
                1: encode_scalar(self.unknown_token),
            }
            unique_tokens: dict[str, dict[str, Any]] = {}
            for token_key, token_payload in normalized:
                unique_tokens[token_key] = token_payload
            for token_key, token_payload in sorted(unique_tokens.items(), key=lambda item: item[0]):
                if token_key in vocabulary:
                    continue
                token_id = len(vocabulary)
                vocabulary[token_key] = token_id
                inverse_vocabulary[token_id] = token_payload
            self.vocabularies_[column] = vocabulary
            self.inverse_vocabularies_[column] = inverse_vocabulary
        return self

    def transform(self, values: Any) -> Any:
        """
        Transform categorical columns to per-column token ids.

        Parameters
        ----------
        values : Any
            Categorical pandas DataFrame.

        Returns
        -------
        Any
            Tokenized pandas DataFrame.
        """

        self._check_is_fitted()
        frame = self._to_frame(values)
        self._validate_columns(frame)
        encoded = pd.DataFrame(index=frame.index)
        for column in self.column_names or ():
            normalized = self._normalize_series(frame[column])
            encoded[column] = [
                self.vocabularies_[column].get(token_key, self.vocabularies_[column][self.unknown_token])
                for token_key, _ in normalized
            ]
        return encoded

    def inverse_transform(self, values: Any) -> Any:
        """
        Decode per-column token ids back to categorical values.

        Parameters
        ----------
        values : Any
            Tokenized pandas DataFrame.

        Returns
        -------
        Any
            Decoded pandas DataFrame.
        """

        self._check_is_fitted()
        frame = self._to_frame(values)
        self._validate_columns(frame)
        decoded = pd.DataFrame(index=frame.index)
        for column in self.column_names or ():
            decoded[column] = [
                decode_scalar(self.inverse_vocabularies_[column].get(int(value), encode_scalar(self.unknown_token)))
                for value in frame[column]
            ]
        return decoded

    def get_vocabulary_size(self, column_name: str) -> int:
        """
        Return the fitted vocabulary size for one categorical column.

        Parameters
        ----------
        column_name : str
            Fitted categorical column name.

        Returns
        -------
        int
            Number of token ids reserved for the column.
        """

        self._check_is_fitted()
        if column_name not in (self.vocabularies_ or {}):
            raise ValueError(f"VocabularyTokenizer has no fitted vocabulary for column {column_name!r}.")
        return len(self.vocabularies_[column_name])

    def get_vocabulary_sizes(self, column_names: tuple[str, ...]) -> tuple[int, ...]:
        """
        Return fitted vocabulary sizes for multiple categorical columns.

        Parameters
        ----------
        column_names : tuple of str
            Ordered categorical column names.

        Returns
        -------
        tuple of int
            Ordered vocabulary sizes aligned with ``column_names``.
        """

        return tuple(self.get_vocabulary_size(column_name) for column_name in column_names)

    def _serialize_state(self) -> dict[str, Any]:
        """
        Serialize fitted tokenizer state.

        Returns
        -------
        dict of str to Any
            Serialized tokenizer state.
        """

        self._check_is_fitted()
        return {
            "column_names": list(self.column_names or ()),
            "vocabularies_": self.vocabularies_,
            "inverse_vocabularies_": self.inverse_vocabularies_,
            "unknown_token": self.unknown_token,
            "missing_token": self.missing_token,
        }

    @classmethod
    def _deserialize_state(cls, state: dict[str, Any]) -> VocabularyTokenizer:
        """
        Reconstruct a tokenizer from serialized state.

        Parameters
        ----------
        state : dict of str to Any
            Serialized tokenizer state.

        Returns
        -------
        VocabularyTokenizer
            Reconstructed tokenizer instance.
        """

        return cls(
            column_names=tuple(state["column_names"]),
            vocabularies_={
                str(column): {str(key): int(value) for key, value in vocabulary.items()}
                for column, vocabulary in state["vocabularies_"].items()
            },
            inverse_vocabularies_={
                str(column): {int(key): value for key, value in vocabulary.items()}
                for column, vocabulary in state["inverse_vocabularies_"].items()
            },
            unknown_token=state["unknown_token"],
            missing_token=state["missing_token"],
        )

    def _normalize_series(self, values: pd.Series) -> list[tuple[str, dict[str, Any]]]:
        """
        Normalize a categorical series to typed token keys.

        Parameters
        ----------
        values : pandas.Series
            Input categorical values.

        Returns
        -------
        list of tuple
            Ordered typed token keys and payloads.
        """

        normalized: list[tuple[str, dict[str, Any]]] = []
        for value in values.tolist():
            if pd.isna(value):
                payload = encode_scalar(None)
                key = self.missing_token
            else:
                payload = encode_scalar(value)
                key = f"{payload['type']}::{payload['value']}"
            normalized.append((key, payload))
        return normalized

    def _to_frame(self, values: Any) -> pd.DataFrame:
        """
        Convert supported inputs to a categorical DataFrame.

        Parameters
        ----------
        values : Any
            Input values to convert.

        Returns
        -------
        pandas.DataFrame
            Categorical DataFrame.
        """

        if not isinstance(values, pd.DataFrame):
            raise TypeError("VocabularyTokenizer expects a pandas DataFrame with one or more categorical columns.")
        return values

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        """
        Validate that input columns match the fitted tokenizer schema.

        Parameters
        ----------
        frame : pandas.DataFrame
            Input frame to validate.

        Returns
        -------
        None
            This method raises when columns are incompatible.
        """

        if tuple(str(column) for column in frame.columns) != tuple(self.column_names or ()):
            raise ValueError("Input categorical columns do not match the fitted tokenizer state.")

    def _check_is_fitted(self) -> None:
        """
        Validate that the tokenizer has been fitted.

        Returns
        -------
        None
            This method raises when the tokenizer is unfitted.
        """

        if self.vocabularies_ is None or self.inverse_vocabularies_ is None or self.column_names is None:
            raise ValueError("VocabularyTokenizer must be fitted before use.")
