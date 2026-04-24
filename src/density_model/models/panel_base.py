"""
Panel Model Base
----------------
Abstract base class for panel forecasting models that operate on a validated
``PanelBatch`` produced by the panel data pipeline.

Classes
-------
BasePanelModel
    Abstract ``nn.Module`` with a ``forward(batch: PanelBatch)`` contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn

from density_model.shared.data.panel.schema import PanelBatch

__all__ = ["BasePanelModel"]


class BasePanelModel(nn.Module, ABC):
    """Abstract base class for panel forecasting models."""

    @abstractmethod
    def forward(self, batch: PanelBatch):  # type: ignore[override]
        """Compute model predictions from a validated panel batch."""
