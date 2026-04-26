"""
Alpha Base
----------
Abstract base for cross-sectional alpha builders.

Each builder consumes two pandas Series keyed by asset at a single
forecast date (``mu`` and ``sigma``) and returns an alpha Series indexed
by the same assets. The signal aggregator downstream (optimizer) expects
alpha to be pre-filtered for missing values — the builder's responsibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

__all__ = ["BaseAlphaBuilder"]


class BaseAlphaBuilder(ABC):
    """Cross-sectional alpha builder contract."""

    @abstractmethod
    def compute(self, *, mu: pd.Series, sigma: pd.Series) -> pd.Series:
        """
        Return the alpha for one date.

        Parameters
        ----------
        mu : pandas.Series
            Predicted conditional mean per asset, indexed by ``asset_id``.
        sigma : pandas.Series
            Predicted conditional sigma per asset, indexed by ``asset_id``.

        Returns
        -------
        pandas.Series
            Alpha per asset, NaN-free.
        """
