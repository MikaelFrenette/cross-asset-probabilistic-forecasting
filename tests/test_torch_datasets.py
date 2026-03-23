"""
Torch Dataset Tests
-------------------
Unit tests for torch dataset wrappers around vectorized panel outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from density_model.models import PanelBatch
from density_model.torch_datasets import VectorizedPanelTorchDataset

__all__ = []


def test_vectorized_panel_torch_dataset_returns_model_ready_sample() -> None:
    """
    Return one forecast-date sample as tensors and metadata.

    Returns
    -------
    None
        This test asserts per-sample tensor conversion.
    """

    panel_data = {
        "X_continuous": np.array([[[[1.0], [2.0]], [[3.0], [4.0]]]], dtype=float),
        "X_continuous_mask": np.array([[[[True], [True]], [[True], [False]]]], dtype=bool),
        "X_cat_continuous": np.array([[[[1], [2]], [[3], [4]]]], dtype=np.int64),
        "X_cat_continuous_mask": np.array([[[[True], [True]], [[True], [True]]]], dtype=bool),
        "X_cat_static": np.array([[[[5], [5]], [[6], [6]]]], dtype=np.int64),
        "X_cat_static_mask": np.array([[[[True], [True]], [[True], [True]]]], dtype=bool),
        "y": np.array([[[[0.5]], [[1.5]]]], dtype=float),
        "y_mask": np.array([[[[True]], [[False]]]], dtype=bool),
        "id_index": ["A", "B"],
        "forecast_start_dates": np.array(["2024-01-03"], dtype="datetime64[ns]"),
        "forecast_end_dates": np.array(["2024-01-03"], dtype="datetime64[ns]"),
    }

    dataset = VectorizedPanelTorchDataset(panel_data)
    features, target, target_mask = dataset[0]
    sample = dataset.get_sample(0)
    batch = dataset.get_batch(0)

    assert len(dataset) == 1
    assert features["X_continuous"].dtype == torch.float32
    assert features["X_cat_continuous"].dtype == torch.long
    assert features["X_continuous_mask"].dtype == torch.bool
    assert target.dtype == torch.float32
    assert target_mask.dtype == torch.bool
    assert sample.metadata["id_index"] == ["A", "B"]
    assert isinstance(batch, PanelBatch)
    assert batch.X_continuous.shape == torch.Size([1, 2, 2, 1])
    training_tuple = batch.as_training_tuple()
    assert training_tuple[1].shape == torch.Size([1, 2, 1, 1])


def test_vectorized_panel_torch_dataset_rejects_missing_targets() -> None:
    """
    Reject panel data missing required target arrays.

    Returns
    -------
    None
        This test asserts fail-fast validation.
    """

    with pytest.raises(ValueError, match="non-empty `y` and `y_mask`"):
        VectorizedPanelTorchDataset(
            {
                "y": None,
                "y_mask": None,
                "id_index": [],
                "forecast_start_dates": np.array([], dtype="datetime64[ns]"),
                "forecast_end_dates": np.array([], dtype="datetime64[ns]"),
            }
        )


def test_vectorized_panel_torch_dataset_handles_absent_optional_streams() -> None:
    """
    Return samples when optional categorical streams are absent.

    Returns
    -------
    None
        This test asserts optional-stream dataset behavior.
    """

    panel_data = {
        "X_continuous": np.array([[[[1.0], [2.0]]]], dtype=float),
        "X_continuous_mask": np.array([[[[True], [True]]]], dtype=bool),
        "X_cat_continuous": None,
        "X_cat_continuous_mask": None,
        "X_cat_static": None,
        "X_cat_static_mask": None,
        "y": np.array([[[[0.5]]]], dtype=float),
        "y_mask": np.array([[[[True]]]], dtype=bool),
        "id_index": ["A"],
        "forecast_start_dates": np.array(["2024-01-03"], dtype="datetime64[ns]"),
        "forecast_end_dates": np.array(["2024-01-03"], dtype="datetime64[ns]"),
    }

    dataset = VectorizedPanelTorchDataset(panel_data)
    features, target, target_mask = dataset[0]

    assert "X_continuous" in features
    assert "X_cat_continuous" not in features
    assert "X_cat_static" not in features
    assert target.shape == torch.Size([1, 1, 1])
    assert target_mask.shape == torch.Size([1, 1, 1])


def test_vectorized_panel_torch_dataset_coerces_object_backed_integer_blocks() -> None:
    """
    Convert object-backed integer arrays to ``torch.long`` tensors.

    Returns
    -------
    None
        This test asserts defensive dtype coercion at the torch dataset boundary.
    """

    panel_data = {
        "X_continuous": np.array([[[[1.0], [2.0]]]], dtype=float),
        "X_continuous_mask": np.array([[[[True], [True]]]], dtype=bool),
        "X_cat_continuous": None,
        "X_cat_continuous_mask": None,
        "X_cat_static": np.array([[[[2], [2]]]], dtype=object),
        "X_cat_static_mask": np.array([[[[True], [True]]]], dtype=bool),
        "y": np.array([[[[0.5]]]], dtype=float),
        "y_mask": np.array([[[[True]]]], dtype=bool),
        "id_index": ["A"],
        "forecast_start_dates": np.array(["2024-01-03"], dtype="datetime64[ns]"),
        "forecast_end_dates": np.array(["2024-01-03"], dtype="datetime64[ns]"),
    }

    dataset = VectorizedPanelTorchDataset(panel_data)
    features, _, _ = dataset[0]

    assert features["X_cat_static"].dtype == torch.long


def test_panel_batch_rejects_shape_mismatch() -> None:
    """
    Reject feature blocks whose batch axes do not match the target tensor.

    Returns
    -------
    None
        This test asserts explicit batch-schema validation.
    """

    with pytest.raises(ValueError, match="must share the batch and ID axes of y"):
        PanelBatch(
            X_continuous=torch.zeros(2, 2, 2, 1, dtype=torch.float32),
            X_continuous_mask=torch.ones(2, 2, 2, 1, dtype=torch.bool),
            X_cat_continuous=None,
            X_cat_continuous_mask=None,
            X_cat_static=None,
            X_cat_static_mask=None,
            y=torch.zeros(1, 2, 1, 1, dtype=torch.float32),
            y_mask=torch.ones(1, 2, 1, 1, dtype=torch.bool),
            id_index=["A", "B"],
            forecast_start_dates=[np.datetime64("2024-01-03")],
            forecast_end_dates=[np.datetime64("2024-01-03")],
        )
