"""
Reproducibility Utilities
--------------------------
Utilities for controlling randomness and ensuring deterministic execution.

Functions
---------
seed_everything
    Seeds Python, NumPy, PyTorch, and CUDA random number generators from a
    single integer seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """
    Seed all random number generators for reproducible runs.

    Parameters
    ----------
    seed : int
        Integer seed applied to every RNG.
    deterministic : bool, default True
        If True, configures cuDNN for fully deterministic operation by
        disabling benchmark mode and enabling deterministic algorithms.
        This guarantees reproducibility at the cost of potential throughput
        reduction on CUDA.

    Notes
    -----
    Seeds the following RNGs in order: Python ``random``, NumPy, PyTorch CPU,
    and all CUDA devices (when available).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
