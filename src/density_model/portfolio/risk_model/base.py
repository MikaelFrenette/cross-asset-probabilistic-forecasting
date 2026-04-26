"""
Covariance Estimator Base
-------------------------
Abstract base for covariance estimators. Enforces the no-lookahead
contract: ``estimate(history, as_of)`` only uses rows of ``history``
whose index is strictly ≤ ``as_of``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

__all__ = ["BaseCovarianceEstimator"]


class BaseCovarianceEstimator(ABC):
    """Covariance estimator contract (strictly-no-lookahead)."""

    @abstractmethod
    def estimate(
        self,
        history: pd.DataFrame,
        *,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """
        Estimate ``Σ`` using rows of ``history`` whose index ≤ ``as_of``.

        Parameters
        ----------
        history : pandas.DataFrame
            Wide-format realized-returns matrix indexed by date, columns
            are asset identifiers.
        as_of : pandas.Timestamp
            Most recent date whose realized return is observable.

        Returns
        -------
        pandas.DataFrame or None
            Square covariance matrix with asset identifiers on both axes,
            or ``None`` when insufficient history is available (burn-in
            not yet reached).
        """
