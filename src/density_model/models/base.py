"""
Model Base
----------
Abstract base classes for panel forecasting models built around the
``PanelBatch`` schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from density_model.models.schema import PanelBatch

__all__ = ["BasePanelModel"]


class BasePanelModel(nn.Module, ABC):
    """
    Abstract base class for panel forecasting models.

    Parameters
    ----------
    None
        Subclasses define architecture-specific configuration and forward logic.
    """

    @abstractmethod
    def forward(self, batch: PanelBatch):  # type: ignore[override]
        """
        Compute model predictions from a validated panel batch.

        Parameters
        ----------
        batch : PanelBatch
            Validated model-facing batch schema.

        Returns
        -------
        torch.Tensor
            Predicted target tensor shaped like ``batch.y``.
        """
