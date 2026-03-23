"""
Torch Vectorized Panel Dataset
------------------------------
Torch dataset wrapper for vectorized panel outputs produced by
``VectorizedPanelDataset``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from density_model.torch_datasets.base import BasePanelTorchDataset, PanelSample

__all__ = ["VectorizedPanelTorchDataset"]


class VectorizedPanelTorchDataset(BasePanelTorchDataset):
    """
    Wrap vectorized panel arrays into a PyTorch dataset interface.

    Parameters
    ----------
    panel_data : dict of str to Any
        Output mapping produced by ``VectorizedPanelDataset.generate_sequences()``.
    """

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
        """
        Return the number of forecast samples in the wrapped panel.

        Returns
        -------
        int
            Number of forecast dates represented by the panel data.
        """

        return int(self.panel_data["y"].shape[0])

    def get_sample(self, index: int) -> PanelSample:
        """
        Return one forecast-date sample from the wrapped panel data.

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        PanelSample
            Per-sample panel tensors and metadata.
        """

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
        return PanelSample(features=features, target=target, target_mask=target_mask, metadata=metadata)

    def _validate_panel_data(self, panel_data: dict[str, Any]) -> None:
        """
        Validate the required vectorized panel outputs.

        Parameters
        ----------
        panel_data : dict of str to Any
            Output mapping produced by ``VectorizedPanelDataset.generate_sequences()``.

        Returns
        -------
        None
            This method raises when required keys are missing or empty.
        """

        required_keys = {"y", "y_mask", "id_index", "forecast_start_dates", "forecast_end_dates"}
        missing_keys = required_keys.difference(panel_data)
        if missing_keys:
            missing_text = ", ".join(sorted(missing_keys))
            raise ValueError(f"VectorizedPanelTorchDataset requires panel keys: {missing_text}")
        if panel_data["y"] is None or panel_data["y_mask"] is None:
            raise ValueError("VectorizedPanelTorchDataset requires non-empty `y` and `y_mask` arrays.")

    def _validate_index(self, index: int) -> None:
        """
        Validate the requested sample index.

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        None
            This method raises when the index is out of bounds.
        """

        if index < 0 or index >= len(self):
            raise IndexError(f"Sample index {index} is out of bounds for dataset length {len(self)}.")

    def _to_tensor(self, value: Any) -> torch.Tensor | None:
        """
        Convert a NumPy-backed sample value to a torch tensor.

        Parameters
        ----------
        value : Any
            Sample value to convert.

        Returns
        -------
        torch.Tensor or None
            Converted tensor, or ``None`` when the sample component is absent.
        """

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
        """
        Convert an object-backed NumPy array to a supported numeric or boolean dtype.

        Parameters
        ----------
        array : numpy.ndarray
            Object-backed array extracted from vectorized panel data.

        Returns
        -------
        numpy.ndarray
            Array coerced to a supported dtype.

        Raises
        ------
        TypeError
            If the object array cannot be coerced to a supported tensor dtype.
        """

        inferred = pd.DataFrame(array.reshape(-1, 1)).infer_objects(copy=False).to_numpy().reshape(array.shape)
        if inferred.dtype != object:
            return inferred

        flattened = inferred.reshape(-1).tolist()
        non_missing = [value for value in flattened if value is not None and not pd.isna(value)]
        if not non_missing:
            return np.asarray(inferred, dtype=np.float32)
        if all(isinstance(value, (bool, np.bool_)) for value in non_missing):
            return np.asarray(inferred, dtype=bool)
        if all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) for value in non_missing):
            return np.asarray(inferred, dtype=np.int64)
        if all(isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) for value in non_missing):
            return np.asarray(inferred, dtype=np.float32)
        raise TypeError("Object-backed vectorized panel arrays must contain only numeric or boolean values.")
