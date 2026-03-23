"""
Torch Dataset Base
------------------
Abstract torch dataset blueprints for wrapping model-ready panel arrays into
PyTorch-compatible sample access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from density_model.models import PanelBatch

__all__ = ["BasePanelTorchDataset", "PanelSample"]


@dataclass(frozen=True, slots=True)
class PanelSample:
    """
    Per-sample panel payload returned by torch dataset wrappers.

    Parameters
    ----------
    features : dict of str to torch.Tensor or None
        Model input tensors for one forecast date.
    target : torch.Tensor
        Target tensor for one forecast date.
    target_mask : torch.Tensor
        Target-availability mask for one forecast date.
    metadata : dict of str to Any
        Sample-level metadata such as forecast dates.
    """

    features: dict[str, torch.Tensor | None]
    target: torch.Tensor
    target_mask: torch.Tensor
    metadata: dict[str, Any]


class BasePanelTorchDataset(Dataset[tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor]], ABC):
    """
    Abstract base class for panel-aware torch dataset wrappers.

    Parameters
    ----------
    None
        Subclasses manage array storage and sample conversion.
    """

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor | None], torch.Tensor, torch.Tensor]:
        """
        Return the model-ready tensors for one sample.

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        tuple
            ``(features, target, target_mask)`` for one sample.
        """

        sample = self.get_sample(index)
        return sample.features, sample.target, sample.target_mask

    def get_batch(self, index: int) -> PanelBatch:
        """
        Return one forecast-date sample as a validated model-facing batch object.

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        PanelBatch
            Validated model-facing batch schema.
        """

        sample = self.get_sample(index)
        return PanelBatch(
            X_continuous=self._ensure_batch_axis(sample.features.get("X_continuous")),
            X_continuous_mask=self._ensure_batch_axis(sample.features.get("X_continuous_mask")),
            X_cat_continuous=self._ensure_batch_axis(sample.features.get("X_cat_continuous")),
            X_cat_continuous_mask=self._ensure_batch_axis(sample.features.get("X_cat_continuous_mask")),
            X_cat_static=self._ensure_batch_axis(sample.features.get("X_cat_static")),
            X_cat_static_mask=self._ensure_batch_axis(sample.features.get("X_cat_static_mask")),
            y=self._ensure_batch_axis(sample.target),
            y_mask=self._ensure_batch_axis(sample.target_mask),
            id_index=list(sample.metadata["id_index"]),
            forecast_start_dates=[sample.metadata["forecast_start_date"]],
            forecast_end_dates=[sample.metadata["forecast_end_date"]],
        )

    def _ensure_batch_axis(self, value: torch.Tensor | None) -> torch.Tensor | None:
        """
        Add a leading batch axis to one sample tensor.

        Parameters
        ----------
        value : torch.Tensor or None
            Sample tensor without a batch axis.

        Returns
        -------
        torch.Tensor or None
            Tensor with a leading batch axis, or ``None`` when absent.
        """

        if value is None:
            return None
        return value.unsqueeze(0)

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of forecast samples.

        Returns
        -------
        int
            Number of forecast dates represented by the dataset.
        """

    @abstractmethod
    def get_sample(self, index: int) -> PanelSample:
        """
        Return the full panel sample, including metadata.

        Parameters
        ----------
        index : int
            Zero-based sample index.

        Returns
        -------
        PanelSample
            Panel sample for the requested forecast date.
        """
