"""
Masked Losses
-------------
Masked regression losses for forecast targets with partial target availability,
including Gaussian log-likelihood losses.
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
    """
    Compute Gaussian negative log-likelihood from ``mu`` and ``sigma`` outputs.

    Parameters
    ----------
    None
        This loss has no configurable parameters.
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian negative log-likelihood.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted tensor whose last axis concatenates ``mu`` and ``sigma``.
        y_true : torch.Tensor
            Ground-truth target tensor.

        Returns
        -------
        torch.Tensor
            Scalar Gaussian negative log-likelihood.
        """

        mu, sigma = self._split_prediction(y_pred=y_pred, y_true=y_true)
        loss = ((y_true - mu) ** 2) / (sigma**2) + torch.log(sigma**2)
        return loss.mean()

    def _split_prediction(self, *, y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split probabilistic predictions into ``mu`` and ``sigma`` tensors.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted tensor.
        y_true : torch.Tensor
            Ground-truth target tensor.

        Returns
        -------
        tuple of torch.Tensor and torch.Tensor
            ``mu`` and ``sigma`` tensors.

        Raises
        ------
        ValueError
            If the prediction shape is incompatible with the target shape.
        """

        target_dim = y_true.shape[-1]
        expected_dim = target_dim * 2
        if y_pred.shape[:-1] != y_true.shape[:-1] or y_pred.shape[-1] != expected_dim:
            raise ValueError("GaussianLogLikelihood expects y_pred with last dimension equal to 2 * y_true.shape[-1].")
        return torch.split(y_pred, target_dim, dim=-1)


class MaskedGaussianLogLikelihood(GaussianLogLikelihood):
    """
    Compute Gaussian negative log-likelihood over valid target positions only.

    Parameters
    ----------
    None
        This loss has no configurable parameters.
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked Gaussian negative log-likelihood.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted tensor whose last axis concatenates ``mu`` and ``sigma``.
        y_true : torch.Tensor
            Ground-truth target tensor.
        y_mask : torch.Tensor
            Boolean tensor indicating valid target positions.

        Returns
        -------
        torch.Tensor
            Scalar masked Gaussian negative log-likelihood.

        Raises
        ------
        ValueError
            If no valid target positions are available.
        """

        mu, sigma = self._split_prediction(y_pred=y_pred, y_true=y_true)
        valid = y_mask.to(dtype=torch.bool)
        if not torch.any(valid):
            raise ValueError("MaskedGaussianLogLikelihood requires at least one valid target position.")
        loss = ((y_true - mu) ** 2) / (sigma**2) + torch.log(sigma**2)
        return loss.masked_select(valid).mean()


class MaskedMSELoss(nn.Module):
    """
    Compute mean squared error over valid target positions only.

    Parameters
    ----------
    None
        This loss has no configurable parameters.
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked mean squared error.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted target tensor.
        y_true : torch.Tensor
            Ground-truth target tensor.
        y_mask : torch.Tensor
            Boolean tensor indicating valid target positions.

        Returns
        -------
        torch.Tensor
            Scalar masked mean squared error.

        Raises
        ------
        ValueError
            If no valid target positions are available.
        """

        valid = y_mask.to(dtype=torch.bool)
        if not torch.any(valid):
            raise ValueError("MaskedMSELoss requires at least one valid target position.")
        squared_error = (y_pred - y_true) ** 2
        return squared_error.masked_select(valid).mean()


class MaskedMAELoss(nn.Module):
    """
    Compute mean absolute error over valid target positions only.

    Parameters
    ----------
    None
        This loss has no configurable parameters.
    """

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute masked mean absolute error.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted target tensor.
        y_true : torch.Tensor
            Ground-truth target tensor.
        y_mask : torch.Tensor
            Boolean tensor indicating valid target positions.

        Returns
        -------
        torch.Tensor
            Scalar masked mean absolute error.

        Raises
        ------
        ValueError
            If no valid target positions are available.
        """

        valid = y_mask.to(dtype=torch.bool)
        if not torch.any(valid):
            raise ValueError("MaskedMAELoss requires at least one valid target position.")
        absolute_error = torch.abs(y_pred - y_true)
        return absolute_error.masked_select(valid).mean()
