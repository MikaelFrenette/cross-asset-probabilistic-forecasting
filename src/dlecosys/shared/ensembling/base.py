"""
BaseEnsemble
------------
Parent contract for ensemble schemes. Subclasses implement ``generate_bootstrap``
to produce per-estimator sample indices; the runner handles training,
aggregation, and I/O.

Classes
-------
BaseEnsemble
    Abstract parent. Contracts ``n_estimators``, ``seed``, and two hooks.

Extending
---------
Subclass ``BaseEnsemble`` and implement ``generate_bootstrap(pool_size, estimator_id, rng)``::

    class StratifiedBaggingEnsemble(BaseEnsemble):
        def __init__(self, *, n_estimators=20, max_samples=1.0, seed=42, y=None):
            self.n_estimators = n_estimators
            self.max_samples = max_samples
            self._y = y
            super().__init__(seed=seed)

        def generate_bootstrap(self, pool_size, estimator_id, rng):
            # Sample per-class with replacement to preserve class balance
            ...

Then register in ``build.py::build_ensemble`` and add to
``EnsembleSection.type`` Literal in ``schema.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor

__all__ = ["BaseEnsemble"]


class BaseEnsemble(ABC):
    """
    Parent contract for ensemble schemes.

    Contracted attributes
    ---------------------
    n_estimators : int
        Number of estimators to train. Must be set as a class attribute or
        assigned to ``self`` before calling ``super().__init__()``.
    seed : int
        Base RNG seed; each estimator gets ``seed + estimator_id``.

    Hooks
    -----
    generate_bootstrap(pool_size, estimator_id, rng) -> (bootstrap_idx, oob_idx)
        MUST be implemented. Returns two 1-D LongTensors; ``bootstrap_idx`` is
        the sample indices (with or without replacement) this estimator trains
        on, ``oob_idx`` is the complement (samples NOT in the bootstrap) used
        as its validation set.
    aggregate(predictions, mode) -> Tensor
        Class method on aggregation module; not overridden per-ensemble.
    """

    n_estimators: int = None  # type: ignore[assignment]

    def __init__(self, *, seed: int = 42) -> None:
        self._validate_contract()
        self.seed = seed

    def _validate_contract(self) -> None:
        n = getattr(self, "n_estimators", None)
        if n is None:
            raise TypeError(
                f"{type(self).__name__} must define 'n_estimators' "
                f"(class attribute or assigned before super().__init__())."
            )
        if not isinstance(n, int) or n < 1:
            raise ValueError(
                f"n_estimators must be a positive int, got {n!r} on {type(self).__name__}."
            )

    def rng_for(self, estimator_id: int) -> torch.Generator:
        """Deterministic per-estimator generator derived from base seed + id."""
        return torch.Generator().manual_seed(self.seed + estimator_id)

    @abstractmethod
    def generate_bootstrap(
        self,
        pool_size: int,
        estimator_id: int,
        rng: torch.Generator,
    ) -> Tuple[Tensor, Tensor]:
        """Return (bootstrap_idx, oob_idx) — both 1-D LongTensors over [0, pool_size)."""

    def generate_feature_subset(
        self,
        n_features: int,
        estimator_id: int,
        rng: torch.Generator,
    ) -> Tensor:
        """
        Return a sorted 1-D LongTensor of feature column indices this estimator sees.

        Default implementation returns all features (no subsampling). Override
        for random-subspace / feature-bagging variants.
        """
        return torch.arange(n_features)
