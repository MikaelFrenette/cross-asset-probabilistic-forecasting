"""
Panel Torch Datasets
--------------------
Torch dataset wrappers that expose vectorized panel outputs as PyTorch-ready
samples compatible with panel forecasting models.

Classes
-------
BasePanelTorchDataset
    Abstract base class for panel-aware torch datasets; emits the
    ``(features, target, target_mask)`` trainer-facing tuple and can assemble a
    validated ``PanelBatch`` via ``get_batch``.

VectorizedPanelTorchDataset
    Wrap vectorized panel arrays produced by ``VectorizedPanelDataset`` into a
    PyTorch dataset interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from density_model.shared.data.panel.schema import PanelBatch, PanelSample

__all__ = ["BasePanelTorchDataset", "VectorizedPanelTorchDataset"]


class BasePanelTorchDataset(
    Dataset[tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor]],
    ABC,
):
    """Abstract base class for panel-aware torch dataset wrappers."""

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor]:
        """Return the model-ready tensors for one sample."""

        sample = self.get_sample(index)
        return sample.features, sample.target, sample.target_mask

    def get_batch(self, index: int) -> PanelBatch:
        """Return one forecast-date sample as a validated model-facing batch object."""

        sample = self.get_sample(index)
        return PanelBatch(
            X_continuous=self._ensure_batch_axis(sample.features.get("X_continuous")),
            X_continuous_mask=self._ensure_batch_axis(sample.features.get("X_continuous_mask")),
            X_cat_continuous=self._ensure_batch_axis(sample.features.get("X_cat_continuous")),
            X_cat_continuous_mask=self._ensure_batch_axis(
                sample.features.get("X_cat_continuous_mask")
            ),
            X_cat_static=self._ensure_batch_axis(sample.features.get("X_cat_static")),
            X_cat_static_mask=self._ensure_batch_axis(sample.features.get("X_cat_static_mask")),
            y=self._ensure_batch_axis(sample.target),
            y_mask=self._ensure_batch_axis(sample.target_mask),
            id_index=list(sample.metadata["id_index"]),
            forecast_start_dates=[sample.metadata["forecast_start_date"]],
            forecast_end_dates=[sample.metadata["forecast_end_date"]],
        )

    def _ensure_batch_axis(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.unsqueeze(0)

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of forecast samples."""

    @abstractmethod
    def get_sample(self, index: int) -> PanelSample:
        """Return the full panel sample, including metadata."""


class VectorizedPanelTorchDataset(BasePanelTorchDataset):
    """Wrap vectorized panel arrays into a PyTorch dataset interface."""

    feature_keys = (
        "X_continuous",
        "X_continuous_mask",
        "X_cat_continuous",
        "X_cat_continuous_mask",
        "X_cat_static",
        "X_cat_static_mask",
    )

    def __init__(self, panel_data: dict[str, Any]) -> None:
        self._validate_panel_data(panel_data)
        self.panel_data = panel_data

    def __len__(self) -> int:
        return int(self.panel_data["y"].shape[0])

    def get_sample(self, index: int) -> PanelSample:
        """Return one forecast-date sample from the wrapped panel data."""

        self._validate_index(index)
        features = {
            key: self._to_tensor(self.panel_data[key][index])
            for key in self.feature_keys
            if key in self.panel_data and self.panel_data[key] is not None
        }
        target = self._to_tensor(self.panel_data["y"][index])
        target_mask = self._to_tensor(self.panel_data["y_mask"][index])
        metadata = {
            "forecast_start_date": self.panel_data["forecast_start_dates"][index],
            "forecast_end_date": self.panel_data["forecast_end_dates"][index],
            "id_index": list(self.panel_data["id_index"]),
        }
        return PanelSample(
            features=features, target=target, target_mask=target_mask, metadata=metadata
        )

    def _validate_panel_data(self, panel_data: dict[str, Any]) -> None:
        required_keys = {"y", "y_mask", "id_index", "forecast_start_dates", "forecast_end_dates"}
        missing_keys = required_keys.difference(panel_data)
        if missing_keys:
            missing_text = ", ".join(sorted(missing_keys))
            raise ValueError(f"VectorizedPanelTorchDataset requires panel keys: {missing_text}")
        if panel_data["y"] is None or panel_data["y_mask"] is None:
            raise ValueError(
                "VectorizedPanelTorchDataset requires non-empty `y` and `y_mask` arrays."
            )

    def _validate_index(self, index: int) -> None:
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Sample index {index} is out of bounds for dataset length {len(self)}."
            )

    def _to_tensor(self, value: Any) -> torch.Tensor | None:
        if value is None:
            return None
        array = np.asarray(value)
        if array.dtype == object:
            array = self._coerce_object_array(array)
        if array.dtype == np.bool_:
            return torch.as_tensor(array, dtype=torch.bool)
        if np.issubdtype(array.dtype, np.integer):
            return torch.as_tensor(array, dtype=torch.long)
        return torch.as_tensor(array, dtype=torch.float32)

    def _coerce_object_array(self, array: np.ndarray) -> np.ndarray:
        inferred = (
            pd.DataFrame(array.reshape(-1, 1))
            .infer_objects(copy=False)
            .to_numpy()
            .reshape(array.shape)
        )
        if inferred.dtype != object:
            return inferred

        flattened = inferred.reshape(-1).tolist()
        non_missing = [value for value in flattened if value is not None and not pd.isna(value)]
        if not non_missing:
            return np.asarray(inferred, dtype=np.float32)
        if all(isinstance(value, (bool, np.bool_)) for value in non_missing):
            return np.asarray(inferred, dtype=bool)
        if all(
            isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            for value in non_missing
        ):
            return np.asarray(inferred, dtype=np.int64)
        if all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in non_missing
        ):
            return np.asarray(inferred, dtype=np.float32)
        raise TypeError(
            "Object-backed vectorized panel arrays must contain only numeric or boolean values."
        )
