"""
BaggingEnsemble
---------------
Concrete ensemble that composes a sample bootstrapper (rows) and a feature
bootstrapper (columns). Behavior is driven by the injected bootstrappers —
`BaggingEnsemble` itself carries no sampling knobs.

Classes
-------
BaggingEnsemble
    Bagging with injected sample + feature bootstrappers.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from dlecosys.shared.ensembling.base import BaseEnsemble
from dlecosys.shared.ensembling.bootstrappers import (
    AllFeaturesBootstrapper,
    BaseFeatureBootstrapper,
    BaseSampleBootstrapper,
    WithReplacementBootstrapper,
)

__all__ = ["BaggingEnsemble"]


class BaggingEnsemble(BaseEnsemble):
    """
    Bagging ensemble — composes a sample bootstrapper and a feature bootstrapper.

    Parameters
    ----------
    n_estimators : int
    sample_bootstrapper : BaseSampleBootstrapper
        Strategy for picking rows per estimator. Defaults to
        ``WithReplacementBootstrapper(max_samples=1.0)``.
    feature_bootstrapper : BaseFeatureBootstrapper
        Strategy for picking columns per estimator. Defaults to
        ``AllFeaturesBootstrapper()``.
    seed : int
    """

    def __init__(
        self,
        *,
        n_estimators: int = 20,
        sample_bootstrapper: Optional[BaseSampleBootstrapper] = None,
        feature_bootstrapper: Optional[BaseFeatureBootstrapper] = None,
        seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.sample_bootstrapper = sample_bootstrapper or WithReplacementBootstrapper()
        self.feature_bootstrapper = feature_bootstrapper or AllFeaturesBootstrapper()
        super().__init__(seed=seed)

    def generate_bootstrap(self, pool_size, estimator_id, rng):
        return self.sample_bootstrapper.sample(pool_size, estimator_id, rng)

    def generate_feature_subset(self, n_features, estimator_id, rng):
        return self.feature_bootstrapper.sample(n_features, estimator_id, rng)
