"""
μ/σ Z-Score Alpha
-----------------
Baseline alpha builder:

    α_raw = μ / σ                  (risk-adjusted predicted return)
    α     = (α_raw − mean) / std   (cross-sectional z-score)

The z-score is computed across the N assets observed at a single date
— the input pandas Series is assumed to be one cross-section at a time.
Assets with NaN μ or σ are dropped before z-scoring (their row is excluded
from mean/std). A degenerate σ = 0 anywhere is dropped too — there's no
defensible raw-alpha value for a perfectly-predicted zero-variance asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from density_model.portfolio.alpha.base import BaseAlphaBuilder

__all__ = ["MuSigmaZScoreAlpha"]


class MuSigmaZScoreAlpha(BaseAlphaBuilder):
    """``α = zscore(μ / σ)`` across the cross-section."""

    def compute(self, *, mu: pd.Series, sigma: pd.Series) -> pd.Series:
        frame = pd.concat({"mu": mu, "sigma": sigma}, axis=1).dropna()
        frame = frame.loc[frame["sigma"] > 0.0]
        if frame.empty:
            return pd.Series(dtype=float)

        raw = frame["mu"] / frame["sigma"]
        mean = float(raw.mean())
        std = float(raw.std(ddof=0))
        if not np.isfinite(std) or std == 0.0:
            # All assets have the same raw alpha — z-scored to 0 everywhere.
            return pd.Series(np.zeros(len(raw)), index=raw.index, name="alpha")
        return pd.Series((raw - mean) / std, name="alpha")
