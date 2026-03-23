"""
Preprocessing Tests
-------------------
Unit tests for saveable scaler and tokenizer artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from density_model.preprocessing import BasePreprocessorArtifact, StandardScaler, VocabularyTokenizer
from density_model.preprocessing.bundle import PreprocessingBundle

__all__ = []


def test_standard_scaler_round_trip_save_load(tmp_path: Path) -> None:
    """
    Save and load a fitted standard scaler artifact.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test asserts deterministic scaler serialization.
    """

    frame = pd.DataFrame(
        {
            "asset_id": ["A", "A", "B", "B"],
            "return": [1.0, 3.0, 10.0, 14.0],
            "volatility": [2.0, 6.0, 20.0, 28.0],
        }
    )
    scaler = StandardScaler().fit(frame)
    transformed_before = scaler.transform(frame)

    artifact_path = tmp_path / "scaler.json"
    scaler.save(artifact_path)
    loaded = BasePreprocessorArtifact.load(artifact_path)

    assert isinstance(loaded, StandardScaler)
    transformed_after = loaded.transform(frame)
    pd.testing.assert_frame_equal(transformed_before, transformed_after)
    restored = loaded.inverse_transform(transformed_after)
    pd.testing.assert_frame_equal(restored, frame)
    assert transformed_after.loc[0, "return"] == -1.0
    assert transformed_after.loc[2, "return"] == -1.0


def test_vocabulary_tokenizer_round_trip_save_load(tmp_path: Path) -> None:
    """
    Save and load a fitted tokenizer artifact.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This test asserts deterministic tokenizer serialization.
    """

    values = pd.DataFrame(
        {
            "sector": ["tech", "energy", None, "tech"],
            "exchange_code": ["NYSE", 7, "NASDAQ", 7],
        }
    )
    tokenizer = VocabularyTokenizer().fit(values)
    encoded_before = tokenizer.transform(values)

    artifact_path = tmp_path / "tokenizer.json"
    tokenizer.save(artifact_path)
    loaded = BasePreprocessorArtifact.load(artifact_path)

    assert isinstance(loaded, VocabularyTokenizer)
    encoded_after = loaded.transform(values)
    pd.testing.assert_frame_equal(encoded_before, encoded_after)
    decoded = loaded.inverse_transform(encoded_after)
    assert list(decoded["sector"]) == ["tech", "energy", None, "tech"]
    assert list(decoded["exchange_code"]) == ["NYSE", 7, "NASDAQ", 7]


def test_vocabulary_tokenizer_maps_unseen_values_to_unknown_token() -> None:
    """
    Map unseen categorical values to the reserved unknown token id.

    Returns
    -------
    None
        This test asserts deterministic unknown-token behavior.
    """

    tokenizer = VocabularyTokenizer().fit(pd.DataFrame({"sector": ["A", "B"]}))
    encoded = tokenizer.transform(pd.DataFrame({"sector": ["A", "C"]}))
    assert encoded["sector"].tolist() == [
        tokenizer.vocabularies_["sector"]["str::A"],
        tokenizer.vocabularies_["sector"]["[UNK]"],
    ]


def test_vocabulary_tokenizer_reports_per_column_vocabulary_sizes() -> None:
    """
    Report fitted vocabulary sizes for ordered categorical columns.

    Returns
    -------
    None
        This test asserts vocabulary-size helpers.
    """

    values = pd.DataFrame(
        {
            "ticker": ["SPY", "QQQ", None],
            "exchange_code": ["NYSE", 7, "NASDAQ"],
        }
    )
    tokenizer = VocabularyTokenizer().fit(values)

    assert tokenizer.get_vocabulary_size("ticker") == 4
    assert tokenizer.get_vocabulary_sizes(("exchange_code", "ticker")) == (5, 4)

    with pytest.raises(ValueError, match="no fitted vocabulary"):
        tokenizer.get_vocabulary_size("sector")


def test_preprocessing_bundle_casts_tokenized_columns_to_integer_dtype() -> None:
    """
    Preserve integer dtype for tokenized categorical columns after transformation.

    Returns
    -------
    None
        This test asserts that tokenized categorical columns do not remain object-typed.
    """

    panel = pd.DataFrame(
        {
            "asset_id": ["SPY", "SPY", "QQQ", "QQQ"],
            "ticker": ["SPY", "SPY", "QQQ", "QQQ"],
            "return": [0.1, 0.2, -0.1, 0.0],
        }
    )
    bundle = PreprocessingBundle().fit(
        panel,
        continuous_columns=("return",),
        categorical_columns=("ticker",),
    )

    transformed = bundle.transform(panel)

    assert pd.api.types.is_integer_dtype(transformed["ticker"])
