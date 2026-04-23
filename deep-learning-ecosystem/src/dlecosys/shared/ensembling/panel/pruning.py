"""
Ensemble Pruning
----------------
Top-N pruning strategy for the panel bagging ensemble.

Functions
---------
prune_top_n
    Return indices of the ``keep`` estimators with the lowest out-of-bag
    validation loss.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["prune_top_n"]


def prune_top_n(*, oob_losses: Sequence[float], keep: int) -> list[int]:
    """Return indices of the top-``keep`` estimators (lowest OOB loss first)."""

    if keep < 1:
        raise ValueError("keep must be >= 1.")
    losses = np.asarray(oob_losses, dtype=float)
    if len(losses) == 0:
        return []
    keep = min(keep, len(losses))
    ordered = np.argsort(losses)
    return [int(index) for index in ordered[:keep]]
