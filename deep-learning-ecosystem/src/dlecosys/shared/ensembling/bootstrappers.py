"""
Bootstrappers
-------------
Two parallel class hierarchies, one for rows (samples) and one for columns
(features). Each has a single ``sample(...)`` hook that subclasses must
implement. ``BaggingEnsemble`` composes one of each.

Classes
-------
BaseSampleBootstrapper
    Parent for row-level bootstrap schemes. Hook: ``sample(pool_size, id, rng)
    -> (bootstrap_idx, oob_idx)``.
WithReplacementBootstrapper
    Classic bagging — draws rows with replacement; OOB is the complement.
NoBootstrapBootstrapper
    Identity — all rows in bootstrap, zero OOB. Warns at construction since
    this disables OOB-as-val training.

BaseFeatureBootstrapper
    Parent for column-level bootstrap schemes. Hook: ``sample(n_features,
    id, rng) -> feature_idx``.
AllFeaturesBootstrapper
    Identity — every estimator sees every feature in natural order.
RandomSubspaceBootstrapper
    Draws a feature subset WITHOUT replacement (Ho 1998).

Functions
---------
build_sample_bootstrapper
    Construct a BaseSampleBootstrapper from a SampleBootstrapperSection.
build_feature_bootstrapper
    Construct a BaseFeatureBootstrapper from a FeatureBootstrapperSection.

Extending
---------
    class BlockBootstrapper(BaseSampleBootstrapper):
        def __init__(self, *, block_size: int = 32, max_samples: float = 1.0):
            self.block_size = block_size
            self.max_samples = max_samples

        def sample(self, pool_size, estimator_id, rng):
            # pick random block starts, concat into bootstrap_idx, complement = oob_idx
            ...

Then add to ``build_sample_bootstrapper`` dispatch and ``SampleBootstrapperSection.type``
Literal in ``schema.py``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor

from dlecosys.shared.config.schema import (
    FeatureBootstrapperSection,
    SampleBootstrapperSection,
)

__all__ = [
    "BaseSampleBootstrapper",
    "WithReplacementBootstrapper",
    "NoBootstrapBootstrapper",
    "BaseFeatureBootstrapper",
    "AllFeaturesBootstrapper",
    "RandomSubspaceBootstrapper",
    "build_sample_bootstrapper",
    "build_feature_bootstrapper",
]


# ---------------------------------------------------------------------------
# Sample (row) bootstrappers
# ---------------------------------------------------------------------------


class BaseSampleBootstrapper(ABC):
    """
    Parent contract for row-level bootstrap schemes.

    Subclasses implement ``sample(pool_size, estimator_id, rng)`` and return a
    pair of 1-D LongTensors: ``(bootstrap_idx, oob_idx)``. The bootstrap
    indices define this estimator's training set; OOB indices define its
    validation set (used for EarlyStopping / ModelCheckpoint).

    Subclasses that produce no OOB (e.g. ``NoBootstrapBootstrapper``) must
    return an empty tensor for ``oob_idx``; the runner handles that case by
    training without a validation phase.
    """

    @abstractmethod
    def sample(
        self,
        pool_size: int,
        estimator_id: int,
        rng: torch.Generator,
    ) -> Tuple[Tensor, Tensor]:
        """Yield ``(bootstrap_idx, oob_idx)``."""


class WithReplacementBootstrapper(BaseSampleBootstrapper):
    """
    Classic bagging: draws ``max_samples * pool_size`` rows WITH replacement;
    OOB is the set-complement. At ``max_samples=1.0`` ~37% of rows are OOB
    on expectation.
    """

    def __init__(self, *, max_samples: float = 1.0) -> None:
        if max_samples <= 0.0:
            raise ValueError(f"max_samples must be > 0, got {max_samples!r}")
        self.max_samples = max_samples

    def sample(self, pool_size, estimator_id, rng):
        sample_count = int(pool_size * self.max_samples)
        if sample_count < 1:
            raise ValueError(
                f"max_samples={self.max_samples} × pool_size={pool_size} "
                f"produced a bootstrap of size {sample_count}; must be ≥ 1."
            )
        bootstrap_idx = torch.randint(0, pool_size, (sample_count,), generator=rng)
        in_bag = torch.zeros(pool_size, dtype=torch.bool)
        in_bag[bootstrap_idx] = True
        oob_idx = torch.where(~in_bag)[0]
        if oob_idx.numel() == 0:
            raise ValueError(
                f"estimator {estimator_id}: no OOB samples with "
                f"max_samples={self.max_samples} and pool_size={pool_size}. "
                f"Reduce max_samples or increase pool size."
            )
        return bootstrap_idx, oob_idx


class NoBootstrapBootstrapper(BaseSampleBootstrapper):
    """
    Identity sampler — every estimator trains on the full pool, with **no
    OOB samples**. Callers must configure their callbacks to monitor a
    training-side metric (e.g. ``loss``), because ``val_loss`` will never
    appear in the logs.
    """

    def __init__(self) -> None:
        warnings.warn(
            "NoBootstrapBootstrapper produces no OOB samples — each estimator "
            "will train without a validation set. Configure "
            "callbacks.early_stopping.monitor and callbacks.checkpoint.monitor "
            "to a training-side metric (e.g. 'loss') or those callbacks will "
            "silently no-op.",
            stacklevel=2,
        )

    def sample(self, pool_size, estimator_id, rng):
        return torch.arange(pool_size), torch.empty(0, dtype=torch.long)


# ---------------------------------------------------------------------------
# Feature (column) bootstrappers
# ---------------------------------------------------------------------------


class BaseFeatureBootstrapper(ABC):
    """
    Parent contract for column-level bootstrap schemes.

    Subclasses implement ``sample(n_features, estimator_id, rng)`` and return
    a sorted 1-D LongTensor of column indices this estimator sees. Features
    have no OOB notion; only the selected columns are used.
    """

    @abstractmethod
    def sample(
        self,
        n_features: int,
        estimator_id: int,
        rng: torch.Generator,
    ) -> Tensor:
        """Return sorted column indices."""


class AllFeaturesBootstrapper(BaseFeatureBootstrapper):
    """Identity — every estimator sees every feature in natural order."""

    def sample(self, n_features, estimator_id, rng):
        return torch.arange(n_features)


class RandomSubspaceBootstrapper(BaseFeatureBootstrapper):
    """
    Draws a random subset of ``max_features * n_features`` columns WITHOUT
    replacement. Returned indices are sorted (stable column order).
    """

    def __init__(self, *, max_features: float = 1.0) -> None:
        if not 0.0 < max_features <= 1.0:
            raise ValueError(
                f"max_features must be in (0, 1], got {max_features!r}"
            )
        self.max_features = max_features

    def sample(self, n_features, estimator_id, rng):
        k = max(1, int(n_features * self.max_features))
        perm = torch.randperm(n_features, generator=rng)
        return perm[:k].sort().values


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_sample_bootstrapper(cfg: SampleBootstrapperSection) -> BaseSampleBootstrapper:
    if cfg.type == "with_replacement":
        return WithReplacementBootstrapper(max_samples=cfg.max_samples)
    if cfg.type == "no_bootstrap":
        return NoBootstrapBootstrapper()
    raise ValueError(f"Unknown sample bootstrapper type: {cfg.type!r}")


def build_feature_bootstrapper(cfg: FeatureBootstrapperSection) -> BaseFeatureBootstrapper:
    if cfg.type == "all":
        return AllFeaturesBootstrapper()
    if cfg.type == "random_subspace":
        return RandomSubspaceBootstrapper(max_features=cfg.max_features)
    raise ValueError(f"Unknown feature bootstrapper type: {cfg.type!r}")
