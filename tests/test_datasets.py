"""
Dataset Tests
-------------
Unit tests for forecast-aligned vectorized panel dataset construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from density_model.datasets import VectorizedPanelConfig, VectorizedPanelDataset

__all__ = []


def test_vectorized_panel_dataset_generates_forecast_aligned_tensors() -> None:
    """
    Build balanced panel tensors with the expected forecasting shapes and values.

    Returns
    -------
    None
        This test asserts tensor shape and target alignment.
    """

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "asset_id": ["A"] * 5 + ["B"] * 5,
            "return": [0.1, 0.2, 0.3, 0.4, 0.5, 1.1, 1.2, 1.3, 1.4, 1.5],
            "volatility": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            "sector": [0] * 5 + [1] * 5,
        }
    )

    dataset = VectorizedPanelDataset(
        data=frame,
        date_column="date",
        id_column="asset_id",
        targets="volatility",
        in_steps=2,
        horizon=1,
        out_steps=1,
        dynamic_features=("return", "volatility"),
        static_categorical_features=("sector",),
    )
    result = dataset.generate_sequences()

    assert result["X_continuous"].shape == (3, 2, 2, 2)
    assert result["X_continuous_mask"].shape == (3, 2, 2, 2)
    assert result["X_cat_static"].shape == (3, 2, 2, 1)
    assert result["X_cat_static_mask"].shape == (3, 2, 2, 1)
    assert result["y"].shape == (3, 2, 1, 1)
    assert result["y_mask"].shape == (3, 2, 1, 1)
    assert result["id_index"] == ["A", "B"]
    np.testing.assert_array_equal(
        result["forecast_start_dates"],
        np.array(["2024-01-03", "2024-01-04", "2024-01-05"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(
        result["forecast_end_dates"],
        np.array(["2024-01-03", "2024-01-04", "2024-01-05"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(
        result["X_continuous"][0, 0],
        np.array([[0.1, 10.0], [0.2, 11.0]]),
    )
    assert np.all(result["X_continuous_mask"][0, 0])
    assert np.all(result["y_mask"][0, 0])
    np.testing.assert_allclose(result["y"][0, 0], np.array([[12.0]]))


def test_vectorized_panel_dataset_truncate_aligns_on_shared_forecast_dates() -> None:
    """
    Align panel samples on shared forecast dates rather than per-identifier sample count.

    Returns
    -------
    None
        This test asserts forecasting-date alignment under truncation.
    """

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                ]
            ),
            "asset_id": ["A"] * 5 + ["B"] * 5,
            "feature": [1, 2, 3, 4, 5, 11, 12, 14, 15, 16],
            "target": [101, 102, 103, 104, 105, 201, 202, 204, 205, 206],
        }
    )

    dataset = VectorizedPanelDataset(
        data=frame,
        date_column="date",
        id_column="asset_id",
        targets="target",
        in_steps=2,
        horizon=1,
        out_steps=1,
        dynamic_features=("feature",),
        panel_config=VectorizedPanelConfig(align="truncate"),
    )
    result = dataset.generate_sequences()

    np.testing.assert_array_equal(
        result["forecast_start_dates"],
        np.array(["2024-01-04", "2024-01-05"], dtype="datetime64[ns]"),
    )
    np.testing.assert_array_equal(
        result["forecast_end_dates"],
        np.array(["2024-01-04", "2024-01-05"], dtype="datetime64[ns]"),
    )
    np.testing.assert_allclose(result["y"][:, 0, 0, 0], np.array([104.0, 105.0]))
    np.testing.assert_allclose(result["y"][:, 1, 0, 0], np.array([204.0, 205.0]))


def test_vectorized_panel_dataset_rejects_invalid_forecasting_configuration() -> None:
    """
    Reject invalid forecasting windows through pydantic validation.

    Returns
    -------
    None
        This test asserts fail-fast dataset configuration validation.
    """

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "asset_id": ["A", "A"],
            "target": [1.0, 2.0],
        }
    )

    with pytest.raises(ValidationError):
        VectorizedPanelDataset(
            data=frame,
            date_column="date",
            id_column="asset_id",
            targets="target",
            in_steps=2,
            horizon=0,
            out_steps=1,
        )


def test_vectorized_panel_dataset_preserves_native_identifier_types() -> None:
    """
    Preserve native identifier types instead of coercing them to strings.

    Returns
    -------
    None
        This test asserts that panel identifiers remain typed keys.
    """

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "asset_id": [1, 1, 1, 2, 2, 2],
            "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        }
    )

    dataset = VectorizedPanelDataset(
        data=frame,
        date_column="date",
        id_column="asset_id",
        targets="target",
        in_steps=2,
        horizon=1,
        out_steps=1,
        dynamic_features=("feature",),
    )
    result = dataset.generate_sequences()

    assert result["id_index"] == [1, 2]


def test_vectorized_panel_dataset_rejects_target_leakage_in_categorical_blocks() -> None:
    """
    Reject target columns inside categorical input blocks.

    Returns
    -------
    None
        This test asserts explicit leakage protection.
    """

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset_id": ["A", "A", "A"],
            "target": [1.0, 2.0, 3.0],
        }
    )

    dataset = VectorizedPanelDataset(
        data=frame,
        date_column="date",
        id_column="asset_id",
        targets="target",
        in_steps=2,
        horizon=1,
        out_steps=1,
        dynamic_categorical_features=("target",),
    )

    with pytest.raises(ValueError, match="X_cat_continuous"):
        dataset.generate_sequences()


def test_vectorized_panel_dataset_emits_feature_level_masks_for_missing_values() -> None:
    """
    Emit feature-level masks while preserving missing values in the tensors.

    Returns
    -------
    None
        This test asserts explicit mask output for missing observations.
    """

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "asset_id": ["A", "A", "A", "B", "B", "B"],
            "feature": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "target": [10.0, 11.0, 12.0, 20.0, 21.0, np.nan],
        }
    )

    dataset = VectorizedPanelDataset(
        data=frame,
        date_column="date",
        id_column="asset_id",
        targets="target",
        in_steps=2,
        horizon=1,
        out_steps=1,
        dynamic_features=("feature",),
        panel_config=VectorizedPanelConfig(align="truncate"),
    )
    result = dataset.generate_sequences()

    assert np.isnan(result["X_continuous"][0, 0, 1, 0])
    assert bool(result["X_continuous_mask"][0, 0, 1, 0]) is False
    assert np.isnan(result["y"][0, 1, 0, 0])
    assert bool(result["y_mask"][0, 1, 0, 0]) is False
