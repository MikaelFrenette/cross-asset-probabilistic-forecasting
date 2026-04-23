"""
Data Validation
---------------
Explicit pre- and post-transform checks for pipeline data tensors.
Raises immediately on any invalid condition — no silent fallbacks.

Functions
---------
validate_split
    Check a single (X, y) split for shape, dtype, NaN, and Inf.
validate_splits
    Apply validate_split to every split in a dict.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["validate_split", "validate_splits"]


def validate_split(X: Tensor, y: Tensor, *, split: str) -> None:
    """
    Raise if X or y contain NaN, Inf, wrong shape, or non-float dtype.

    Parameters
    ----------
    X : Tensor
        Feature tensor. Must be 2D and float.
    y : Tensor
        Target tensor. Must be finite.
    split : str
        Split name used in error messages (e.g. "train", "val (post-transform)").

    Raises
    ------
    ValueError
        On NaN, Inf, wrong shape, or non-float X.
    """
    if X.ndim != 2:
        raise ValueError(
            f"X ({split}) must be 2D; got shape {tuple(X.shape)}."
        )
    if not X.is_floating_point():
        raise ValueError(
            f"X ({split}) must have a float dtype; got {X.dtype}. "
            f"Cast explicitly before calling this function."
        )
    if torch.isnan(X).any():
        raise ValueError(f"NaN detected in X ({split}).")
    if torch.isinf(X).any():
        raise ValueError(f"Inf detected in X ({split}).")
    if y.is_floating_point():
        if torch.isnan(y).any():
            raise ValueError(f"NaN detected in y ({split}).")
        if torch.isinf(y).any():
            raise ValueError(f"Inf detected in y ({split}).")


def validate_splits(splits: Dict[str, Tuple[Tensor, Tensor]], *, tag: str = "") -> None:
    """
    Apply :func:`validate_split` to every split in a dict.

    Parameters
    ----------
    splits : dict mapping split name → (X, y)
    tag : str, optional
        Appended to each split name in error messages (e.g. "post-transform").
    """
    for split_name, (X, y) in splits.items():
        label = f"{split_name} ({tag})" if tag else split_name
        validate_split(X, y, split=label)
