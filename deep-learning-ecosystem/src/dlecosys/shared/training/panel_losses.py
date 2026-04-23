"""
Panel Losses
------------
Masked regression losses for panel forecasting targets with partial target
availability, including Gaussian log-likelihood losses used by the density
model.

Classes
-------
GaussianLogLikelihood
    Gaussian negative log-likelihood over all positions, given ``mu`` and
    ``sigma`` concatenated on the last axis of the prediction.

MaskedGaussianLogLikelihood
    Gaussian negative log-likelihood restricted to valid target positions via
    a boolean mask. Callable signature ``(y_pred, y_true, y_mask)``.

MaskedMSELoss
    Masked mean squared error.

MaskedMAELoss
    Masked mean absolute error.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "GaussianLogLikelihood",
    "MaskedGaussianLogLikelihood",
    "MaskedMAELoss",
    "MaskedMSELoss",
]


class GaussianLogLikelihood(nn.Module):
    """Compute Gaussian negative log-likelihood from ``mu`` and ``sigma`` outputs."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mu, sigma = self._split_prediction(y_pred=y_pred, y_true=y_true)
        loss = ((y_true - mu) ** 2) / (sigma**2) + torch.log(sigma**2)
        return loss.mean()

    def _split_prediction(
        self, *, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_dim = y_true.shape[-1]
        expected_dim = target_dim * 2
        if y_pred.shape[:-1] != y_true.shape[:-1] or y_pred.shape[-1] != expected_dim:
            raise ValueError(
                "GaussianLogLikelihood expects y_pred with last dimension "
                "equal to 2 * y_true.shape[-1]."
            )
        return torch.split(y_pred, target_dim, dim=-1)


class MaskedGaussianLogLikelihood(GaussianLogLikelihood):
    """Compute Gaussian negative log-likelihood over valid target positions only."""

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, y_mask: torch.Tensor
    ) -> torch.Tensor:
        mu, sigma = self._split_prediction(y_pred=y_pred, y_true=y_true)
        valid = y_mask.to(dtype=torch.bool)
        if not torch.any(valid):
            raise ValueError(
                "MaskedGaussianLogLikelihood requires at least one valid target position."
            )
        loss = ((y_true - mu) ** 2) / (sigma**2) + torch.log(sigma**2)
        return loss.masked_select(valid).mean()


class MaskedMSELoss(nn.Module):
    """Compute mean squared error over valid target positions only."""

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, y_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = y_mask.to(dtype=torch.bool)
        if not torch.any(valid):
            raise ValueError("MaskedMSELoss requires at least one valid target position.")
        squared_error = (y_pred - y_true) ** 2
        return squared_error.masked_select(valid).mean()


class MaskedMAELoss(nn.Module):
    """Compute mean absolute error over valid target positions only."""

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, y_mask: torch.Tensor
    ) -> torch.Tensor:
        valid = y_mask.to(dtype=torch.bool)
        if not torch.any(valid):
            raise ValueError("MaskedMAELoss requires at least one valid target position.")
        absolute_error = torch.abs(y_pred - y_true)
        return absolute_error.masked_select(valid).mean()
