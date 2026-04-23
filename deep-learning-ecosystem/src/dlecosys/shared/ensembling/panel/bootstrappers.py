"""
Panel Bootstrappers
-------------------
Sample- and feature-axis bootstrappers for the panel bagging ensemble.

Classes
-------
BaseSampleBootstrapper
    Abstract base for forecast-date (sample-axis) bootstrappers.

DateBootstrapper
    With-replacement bootstrap over unique forecast dates. Produces in-bag
    dates (for training) and out-of-bag dates (for OOB validation).

NoSampleBootstrapper
    Passthrough — every estimator sees every date, no OOB.

BaseFeatureBootstrapper
    Abstract base for asset-axis (ID) bootstrappers.

AssetSubspaceBootstrapper
    Without-replacement subsample of asset identifiers.

NoFeatureBootstrapper
    Passthrough — every estimator sees every asset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

__all__ = [
    "AssetSubspaceBootstrapper",
    "BaseFeatureBootstrapper",
    "BaseSampleBootstrapper",
    "DateBootstrapper",
    "NoFeatureBootstrapper",
    "NoSampleBootstrapper",
]


class BaseSampleBootstrapper(ABC):
    """Abstract base for forecast-date bootstrappers."""

    @abstractmethod
    def sample(
        self, dates: np.ndarray, *, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ``(in_bag_dates, out_of_bag_dates)`` for one estimator.

        Parameters
        ----------
        dates : numpy.ndarray
            Sorted unique forecast dates in the training panel.
        rng : numpy.random.Generator
            Estimator-specific PRNG.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Dates for training (may contain duplicates when sampling with
            replacement) and dates reserved for OOB validation.
        """


class DateBootstrapper(BaseSampleBootstrapper):
    """With-replacement bootstrap over unique forecast dates."""

    def __init__(self, max_samples: float = 1.0) -> None:
        if not 0.0 < max_samples <= 1.0:
            raise ValueError("max_samples must lie in (0.0, 1.0].")
        self.max_samples = max_samples

    def sample(
        self, dates: np.ndarray, *, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(dates)
        draws = max(1, int(round(n * self.max_samples)))
        in_bag = rng.choice(dates, size=draws, replace=True)
        oob_mask = ~np.isin(dates, in_bag)
        out_of_bag = dates[oob_mask]
        return in_bag, out_of_bag


class NoSampleBootstrapper(BaseSampleBootstrapper):
    """Passthrough sample bootstrapper — every estimator sees every date."""

    def sample(
        self, dates: np.ndarray, *, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        return dates, np.array([], dtype=dates.dtype)


class BaseFeatureBootstrapper(ABC):
    """Abstract base for asset-axis (ID) bootstrappers."""

    @abstractmethod
    def sample(
        self, asset_ids: Sequence[Any], *, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Return the subset of ``asset_ids`` this estimator should train on.

        Parameters
        ----------
        asset_ids : sequence of Any
            Full ordered list of asset identifiers in the panel.
        rng : numpy.random.Generator
            Estimator-specific PRNG.

        Returns
        -------
        numpy.ndarray
            Selected asset identifiers (without replacement, order preserved).
        """


class AssetSubspaceBootstrapper(BaseFeatureBootstrapper):
    """Random without-replacement subsample of asset identifiers."""

    def __init__(self, max_features: float = 1.0) -> None:
        if not 0.0 < max_features <= 1.0:
            raise ValueError("max_features must lie in (0.0, 1.0].")
        self.max_features = max_features

    def sample(
        self, asset_ids: Sequence[Any], *, rng: np.random.Generator
    ) -> np.ndarray:
        n = len(asset_ids)
        keep = max(1, int(round(n * self.max_features)))
        asset_array = np.array(list(asset_ids), dtype=object)
        chosen_indices = rng.choice(n, size=keep, replace=False)
        chosen_indices.sort()
        return asset_array[chosen_indices]


class NoFeatureBootstrapper(BaseFeatureBootstrapper):
    """Passthrough feature bootstrapper — every estimator sees every asset."""

    def sample(
        self, asset_ids: Sequence[Any], *, rng: np.random.Generator
    ) -> np.ndarray:
        return np.array(list(asset_ids), dtype=object)
