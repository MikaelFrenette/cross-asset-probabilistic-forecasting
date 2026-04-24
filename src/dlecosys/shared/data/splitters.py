"""
Validation Splitters
--------------------
Parent class + bundled splitters that define how a tuning trial generates
its validation data. Users extend ``BaseSplitter`` to plug in custom
cross-validation schemes (temporal, group-aware, nested, etc.).

Classes
-------
BaseSplitter
    Parent contract. Subclasses implement one of two hooks to yield
    (train_dataset, val_dataset) per fold.
HoldoutSplitter
    Single-fold splitter that reuses preprocess.py's original train/val
    split verbatim. Behaviorally equivalent to the pre-splitter tuning flow.
KFoldSplitter
    K-fold cross-validation over the concatenated train+val pool.
StratifiedKFoldSplitter
    K-fold with class-proportion preservation, for classification tasks.

Functions
---------
build_splitter
    Construct the configured splitter from a SplitterSection + DataPaths layout.

Extending
---------
Two ways to define your own splitter:

1. **Index-based (recommended for most cross-validation schemes):**

    class GroupKFoldSplitter(BaseSplitter):
        def __init__(self, train_path, val_path=None, *, n_splits=5, group_col=0, seed=42):
            self.n_splits = n_splits
            self.group_col = group_col
            super().__init__(train_path, val_path, seed=seed)

        def generate_indices(self, X, y):
            # yield (train_idx, val_idx) per fold
            ...

   The base class handles loading the pool and slicing into TensorDatasets.

2. **Custom data flow** (if "single pool, index splits" doesn't fit —
   e.g. HoldoutSplitter reuses separate files):

    class MySplitter(BaseSplitter):
        n_splits = 1
        def iterate_folds(self):
            yield train_ds, val_ds
            ...

Either way the parent enforces the ``n_splits`` contract and the
(train_dataset, val_dataset) output shape.
"""

from __future__ import annotations

from abc import ABC
from typing import Iterator, Tuple

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from dlecosys.shared.config.schema import SplitterSection
from dlecosys.shared.run import DataPaths

__all__ = [
    "BaseSplitter",
    "HoldoutSplitter",
    "KFoldSplitter",
    "StratifiedKFoldSplitter",
    "build_splitter",
]


class BaseSplitter(ABC):
    """
    Parent class for validation splitters.

    Contracted attributes
    ---------------------
    n_splits : int
        Number of folds this splitter yields. Must be set as a class
        attribute or assigned to ``self`` **before** calling
        ``super().__init__()``.
    train_path : PathLike
        Path to the training pool tensor file (``torch.save``-ed
        ``{"X": ..., "y": ...}``).
    val_path : PathLike or None
        Path to the validation tensor file. When present, concatenated
        with train to form the pool for splitting. Required by
        ``HoldoutSplitter``.
    seed : int
        RNG seed for reproducible shuffling.

    Contracted output
    -----------------
    ``iterate_folds()`` yields exactly ``n_splits`` pairs of
    ``(train_dataset, val_dataset)`` where each is a ``TensorDataset``.

    Hooks
    -----
    generate_indices(X, y) -> Iterator[(train_idx, val_idx)]
        Primary extension point for index-based splitters. Default raises.
    _load_pool() -> (X, y)
        Returns the concatenated train+val tensors. Override for custom
        pool construction.
    iterate_folds() -> Iterator[(train_ds, val_ds)]
        Template method. Default loads pool + calls generate_indices.
        Override entirely for splitters that do not fit the pool/indices shape.
    """

    n_splits: int = None  # type: ignore[assignment]

    def __init__(self, train_path, val_path=None, *, seed: int = 42) -> None:
        self._validate_contract()
        self.train_path = train_path
        self.val_path = val_path
        self.seed = seed

    def _validate_contract(self) -> None:
        n = getattr(self, "n_splits", None)
        if n is None:
            raise TypeError(
                f"{type(self).__name__} must define 'n_splits' "
                f"(class attribute, or assigned to self before super().__init__())."
            )
        if not isinstance(n, int) or n < 1:
            raise ValueError(
                f"n_splits must be a positive int, got {n!r} "
                f"on {type(self).__name__}."
            )

    def iterate_folds(self) -> Iterator[Tuple[TensorDataset, TensorDataset]]:
        """Default flow: load pool → generate indices → yield (train_ds, val_ds) per fold."""
        X, y = self._load_pool()
        for train_idx, val_idx in self.generate_indices(X, y):
            yield (
                TensorDataset(X[train_idx], y[train_idx]),
                TensorDataset(X[val_idx], y[val_idx]),
            )

    def _load_pool(self) -> Tuple[Tensor, Tensor]:
        """Concatenate train + val (if val_path is set) into a single pool."""
        train = torch.load(self.train_path, weights_only=True)
        X, y = train["X"], train["y"]
        if self.val_path is not None:
            val = torch.load(self.val_path, weights_only=True)
            X = torch.cat([X, val["X"]], dim=0)
            y = torch.cat([y, val["y"]], dim=0)
        return X, y

    def generate_indices(self, X: Tensor, y: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
        """Yield (train_idx, val_idx) per fold. Subclasses implementing the index-based contract override this."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement generate_indices(X, y) "
            f"or override iterate_folds() directly."
        )


class HoldoutSplitter(BaseSplitter):
    """
    Single-fold splitter that reuses the original train.pt / val.pt written
    by preprocess.py. Phase 1 default — equivalent to the pre-splitter
    tuning behavior.
    """

    n_splits = 1

    def iterate_folds(self) -> Iterator[Tuple[TensorDataset, TensorDataset]]:
        if self.val_path is None:
            raise ValueError("HoldoutSplitter requires both train_path and val_path")
        train = torch.load(self.train_path, weights_only=True)
        val = torch.load(self.val_path, weights_only=True)
        yield (
            TensorDataset(train["X"], train["y"]),
            TensorDataset(val["X"], val["y"]),
        )


class KFoldSplitter(BaseSplitter):
    """
    K-fold cross-validation on the concatenated train+val pool.

    Each fold uses a contiguous slice of the shuffled index array as val,
    the remainder as train. Guarantees disjoint val sets and full coverage
    across folds.
    """

    def __init__(
        self,
        train_path,
        val_path=None,
        *,
        n_splits: int = 5,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        super().__init__(train_path, val_path, seed=seed)

    def generate_indices(self, X: Tensor, y: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
        n = len(X)
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed)
            idx = torch.randperm(n, generator=g)
        else:
            idx = torch.arange(n)

        fold_size = n // self.n_splits
        for k in range(self.n_splits):
            val_start = k * fold_size
            val_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            val_idx = idx[val_start:val_end]
            train_idx = torch.cat([idx[:val_start], idx[val_end:]])
            yield train_idx, val_idx


class StratifiedKFoldSplitter(BaseSplitter):
    """
    Stratified k-fold preserving class proportions per fold.

    Requires a 1-D integer label tensor (classification). Raises if y is
    floating-point or multi-target.
    """

    def __init__(
        self,
        train_path,
        val_path=None,
        *,
        n_splits: int = 5,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        super().__init__(train_path, val_path, seed=seed)

    def generate_indices(self, X: Tensor, y: Tensor) -> Iterator[Tuple[Tensor, Tensor]]:
        y_flat = y.squeeze() if y.dim() > 1 else y
        if y_flat.is_floating_point():
            raise ValueError(
                "StratifiedKFoldSplitter requires integer class labels; got floating-point y."
            )

        g = torch.Generator().manual_seed(self.seed)
        classes = torch.unique(y_flat)

        per_class_indices = []
        for c in classes:
            ci = torch.where(y_flat == c)[0]
            if self.shuffle:
                ci = ci[torch.randperm(len(ci), generator=g)]
            per_class_indices.append(ci)

        folds: list[list[Tensor]] = [[] for _ in range(self.n_splits)]
        for ci in per_class_indices:
            fold_size = len(ci) // self.n_splits
            for k in range(self.n_splits):
                start = k * fold_size
                end = (k + 1) * fold_size if k < self.n_splits - 1 else len(ci)
                folds[k].append(ci[start:end])

        folds_concat = [torch.cat(f) for f in folds]
        for k in range(self.n_splits):
            val_idx = folds_concat[k]
            train_idx = torch.cat([folds_concat[j] for j in range(self.n_splits) if j != k])
            yield train_idx, val_idx


def build_splitter(cfg: SplitterSection, layout: DataPaths) -> BaseSplitter:
    """
    Construct the splitter selected by ``cfg.type``.

    Plugging in a custom splitter (example: time-series expanding window)
    --------------------------------------------------------------------
    1. Subclass ``BaseSplitter`` and implement ``generate_indices(X, y)``::

        class ExpandingWindowSplitter(BaseSplitter):
            def __init__(self, train_path, val_path=None, *, n_splits=5, seed=42):
                self.n_splits = n_splits
                super().__init__(train_path, val_path, seed=seed)

            def generate_indices(self, X, y):
                # Assumes rows are already in temporal order.
                n = len(X)
                val_size = n // (self.n_splits + 1)
                for k in range(self.n_splits):
                    train_end = val_size * (k + 1)
                    val_end = train_end + val_size
                    yield torch.arange(train_end), torch.arange(train_end, val_end)

    2. Add the new type to ``SplitterSection.type`` Literal in
       ``src/dlecosys/shared/config/schema.py``.
    3. Add a branch to this function::

        if cfg.type == "expanding_window":
            return ExpandingWindowSplitter(train_path, val_path,
                                           n_splits=cfg.n_splits, seed=cfg.seed)

    4. Reference it in your config::

        tuning:
          splitter:
            type: expanding_window
            n_splits: 5
    """
    train_path = layout.data_path("train")
    val_path = layout.data_path("val")
    if cfg.type == "holdout":
        return HoldoutSplitter(train_path, val_path, seed=cfg.seed)
    if cfg.type == "kfold":
        return KFoldSplitter(
            train_path,
            val_path,
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )
    if cfg.type == "stratified_kfold":
        return StratifiedKFoldSplitter(
            train_path,
            val_path,
            n_splits=cfg.n_splits,
            shuffle=cfg.shuffle,
            seed=cfg.seed,
        )
    raise ValueError(f"Unknown splitter type: {cfg.type!r}")
