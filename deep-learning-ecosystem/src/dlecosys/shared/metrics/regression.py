"""
Regression Metrics
------------------
Scalar metric functions for regression tasks.

Functions
---------
mae
    Mean absolute error.
mse
    Mean squared error.
rmse
    Root mean squared error.
mape
    Mean absolute percentage error.
"""

from __future__ import annotations

import torch

__all__ = ["mae", "mse", "rmse", "mape"]


def mae(*, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth values.
    y_pred : torch.Tensor
        Predicted values, same shape as y_true.

    Returns
    -------
    torch.Tensor
        Scalar MAE.
    """
    return (y_pred - y_true).abs().mean()


def mse(*, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth values.
    y_pred : torch.Tensor
        Predicted values, same shape as y_true.

    Returns
    -------
    torch.Tensor
        Scalar MSE.
    """
    return ((y_pred - y_true) ** 2).mean()


def rmse(*, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Root mean squared error.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth values.
    y_pred : torch.Tensor
        Predicted values, same shape as y_true.

    Returns
    -------
    torch.Tensor
        Scalar RMSE.
    """
    return mse(y_true=y_true, y_pred=y_pred).sqrt()


def mape(
    *,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Mean absolute percentage error.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth values.
    y_pred : torch.Tensor
        Predicted values, same shape as y_true.
    eps : float, default 1e-8
        Floor applied to |y_true| before division to prevent instability on
        near-zero targets.

    Returns
    -------
    torch.Tensor
        Scalar MAPE (as a fraction, not a percentage).
    """
    return ((y_pred - y_true).abs() / (y_true.abs() + eps)).mean()
