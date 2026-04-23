"""
Aggregation
-----------
Reduce per-estimator predictions to a single ensemble prediction.

Functions
---------
aggregate
    Dispatch on aggregation mode ("mean" / "median" / "soft_vote" / "majority").
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["aggregate"]


def aggregate(predictions: Tensor, mode: str) -> Tensor:
    """
    Reduce per-estimator predictions along the estimator dimension.

    Parameters
    ----------
    predictions : Tensor
        Shape ``(n_estimators, n_samples, ...)``. Regression: ``(n_est, n_samples, 1)``.
        Classification: logits ``(n_est, n_samples, n_classes)``.
    mode : str
        One of:
        - ``mean``      : mean across estimators (regression default)
        - ``median``    : median across estimators (regression, robust to outliers)
        - ``soft_vote`` : mean of softmax probabilities, then argmax (classification)
        - ``majority``  : per-estimator argmax, then mode (classification)

    Returns
    -------
    Tensor
        Shape ``(n_samples, ...)`` for mean/median/soft_vote, ``(n_samples,)`` for majority.
    """
    if mode == "mean":
        return predictions.mean(dim=0)
    if mode == "median":
        return predictions.median(dim=0).values
    if mode == "soft_vote":
        probs = F.softmax(predictions, dim=-1)
        return probs.mean(dim=0).argmax(dim=-1)
    if mode == "majority":
        hard = predictions.argmax(dim=-1)
        return torch.mode(hard, dim=0).values
    raise ValueError(f"Unknown aggregation mode: {mode!r}")
