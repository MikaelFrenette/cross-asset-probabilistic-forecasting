"""
Rolling Covariance
------------------
Sample covariance over the most recent ``window_size`` dates that are
less than or equal to ``as_of``. A small diagonal ridge is added for
numerical stability — keeps the matrix invertible when the window is
close to the asset count or when two assets are near-perfectly
collinear.

Returns ``None`` when fewer than ``window_size`` dates are available
(burn-in not yet reached).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from density_model.portfolio.risk_model.base import BaseCovarianceEstimator

__all__ = ["RollingCovariance"]


class RollingCovariance(BaseCovarianceEstimator):
    """Sample covariance over the last ``window_size`` dates ≤ ``as_of``."""

    def __init__(
        self,
        *,
        window_size: int,
        ridge: float = 1e-4,
        ridge_mode: str = "relative",
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2.")
        if ridge < 0.0:
            raise ValueError("ridge must be >= 0.")
        if ridge_mode not in {"relative", "absolute"}:
            raise ValueError(
                f"ridge_mode must be 'relative' or 'absolute'; got {ridge_mode!r}."
            )
        self.window_size = window_size
        self.ridge = ridge
        self.ridge_mode = ridge_mode

    def estimate(
        self,
        history: pd.DataFrame,
        *,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame | None:
        window = history.loc[history.index <= as_of].tail(self.window_size)
        if len(window) < self.window_size:
            return None
        # Drop asset columns that are entirely NaN in the window.
        usable = window.dropna(axis=1, how="all")
        if usable.shape[1] < 2:
            return None
        # ``DataFrame.cov`` uses pairwise-complete observations; the ridge
        # guards against singularities when two columns are near-collinear.
        cov = usable.cov()
        idx = cov.index
        matrix = cov.to_numpy()
        diag_mean = float(np.mean(np.diag(matrix)))
        if self.ridge_mode == "relative":
            # Scale the ridge by the mean variance so the regularizer
            # matches the covariance's natural scale across asset classes.
            shrink = self.ridge * diag_mean
        else:
            shrink = self.ridge
        regularized = matrix + shrink * np.eye(len(idx))
        return pd.DataFrame(regularized, index=idx, columns=idx)
