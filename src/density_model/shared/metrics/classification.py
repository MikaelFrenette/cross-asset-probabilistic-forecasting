"""
Classification Metrics
-----------------------
Scalar metric functions for classification tasks.

Functions
---------
accuracy
    Top-1 accuracy for multi-class classification from class logits.
binary_accuracy
    Accuracy for binary classification from logits or probabilities.
"""

from __future__ import annotations

import torch

__all__ = ["accuracy", "binary_accuracy"]


def accuracy(*, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Top-1 accuracy for multi-class classification.

    Parameters
    ----------
    y_true : torch.Tensor of shape (N,)
        Ground-truth class indices.
    y_pred : torch.Tensor of shape (N, C)
        Raw class logits. Argmax is applied along the last dimension.

    Returns
    -------
    torch.Tensor
        Scalar accuracy in [0, 1].
    """
    preds = y_pred.argmax(dim=-1)
    return (preds == y_true).float().mean()


def binary_accuracy(
    *,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Accuracy for binary classification.

    Parameters
    ----------
    y_true : torch.Tensor of shape (N,) or (N, 1)
        Binary ground-truth labels (0 or 1).
    y_pred : torch.Tensor of shape (N,) or (N, 1)
        Logits or probabilities. Sigmoid is applied, then values >= threshold
        are classified as 1.
    threshold : float, default 0.5
        Decision boundary after sigmoid.

    Returns
    -------
    torch.Tensor
        Scalar accuracy in [0, 1].

    Notes
    -----
    Use ``functools.partial(binary_accuracy, threshold=0.3)`` to register a
    custom threshold as a named metric in BaseTrainer.
    """
    preds = (torch.sigmoid(y_pred) >= threshold).long()
    return (preds == y_true.long()).float().mean()
