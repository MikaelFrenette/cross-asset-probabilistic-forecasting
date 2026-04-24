"""
Synthetic Data Generation
--------------------------
Generates reproducible synthetic datasets for regression and classification
pipeline demos.

Functions
---------
make_regression_splits
    Linear regression dataset with Gaussian noise, split into train/val/test.
make_classification_splits
    Cluster-based multiclass dataset split into train/val/test.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from dlecosys.shared.config.schema import SyntheticSection

__all__ = [
    "make_regression_data",
    "make_classification_data",
    "make_regression_splits",
    "make_classification_splits",
    "split_tensors",
]

_Splits = Dict[str, Tuple[Tensor, Tensor]]


def make_regression_data(
    n_samples: int,
    n_features: int,
    noise: float,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    """
    Generate a synthetic linear regression dataset.

    Returns
    -------
    (X, y) where X is (n_samples, n_features) float32 and y is (n_samples, 1) float32.
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, n_features)
    w = torch.randn(n_features, 1)
    y = X @ w + noise * torch.randn(n_samples, 1)
    return X, y


def make_classification_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    noise: float,
    seed: int,
) -> Tuple[Tensor, Tensor]:
    """
    Generate a synthetic cluster-based multiclass dataset.

    Returns
    -------
    (X, y) where X is (n_samples, n_features) float32 and y is (n_samples,) long.
    """
    torch.manual_seed(seed)
    samples_per_class = n_samples // n_classes
    centers = torch.randn(n_classes, n_features) * 3
    X_parts: list[Tensor] = []
    y_parts: list[Tensor] = []
    for c in range(n_classes):
        X_c = centers[c] + noise * torch.randn(samples_per_class, n_features)
        y_c = torch.full((samples_per_class,), c, dtype=torch.long)
        X_parts.append(X_c)
        y_parts.append(y_c)
    return torch.cat(X_parts), torch.cat(y_parts)


def make_regression_splits(cfg: SyntheticSection, seed: int) -> _Splits:
    """Generate a synthetic regression dataset and split into train/val/test."""
    X, y = make_regression_data(cfg.n_samples, cfg.n_features, cfg.noise, seed)
    return _split(X, y, 0.1, 0.2, seed)


def make_classification_splits(cfg: SyntheticSection, seed: int) -> _Splits:
    """Generate a synthetic classification dataset and split into train/val/test."""
    assert cfg.n_classes is not None, "n_classes required for classification"
    X, y = make_classification_data(cfg.n_samples, cfg.n_features, cfg.n_classes, cfg.noise, seed)
    return _split(X, y, 0.1, 0.2, seed)


def split_tensors(X: Tensor, y: Tensor, val_size: float, test_size: float, seed: int) -> _Splits:
    """
    Randomly split (X, y) tensors into train / val / test.

    Parameters
    ----------
    X : Tensor of shape (N, n_features)
    y : Tensor of shape (N, ...) — any dtype
    val_size : float
        Fraction of samples for validation.
    test_size : float
        Fraction of samples for test.
    seed : int
        RNG seed for reproducible splits.

    Returns
    -------
    dict with keys "train", "val", "test" → (X, y) pairs.
    """
    return _split(X, y, val_size, test_size, seed)


def _split(X: Tensor, y: Tensor, val_size: float, test_size: float, seed: int) -> _Splits:
    n = len(X)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)

    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    return {
        "train": (X[train_idx], y[train_idx]),
        "val": (X[val_idx], y[val_idx]),
        "test": (X[test_idx], y[test_idx]),
    }
