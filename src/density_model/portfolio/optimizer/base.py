"""
Optimizer Base
--------------
Abstract base class for portfolio weight optimizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

__all__ = ["BaseOptimizer"]


class BaseOptimizer(ABC):
    """Weight optimizer contract."""

    @abstractmethod
    def optimize(
        self,
        *,
        alpha: pd.Series,
        covariance: pd.DataFrame,
    ) -> pd.Series:
        """
        Solve for portfolio weights.

        Parameters
        ----------
        alpha : pandas.Series
            Alpha per asset, indexed by ``asset_id``. Order fixes the
            column / row order of ``covariance``.
        covariance : pandas.DataFrame
            Square covariance matrix aligned with ``alpha.index`` on
            both axes.

        Returns
        -------
        pandas.Series
            Weights per asset, indexed by ``alpha.index``.
        """
