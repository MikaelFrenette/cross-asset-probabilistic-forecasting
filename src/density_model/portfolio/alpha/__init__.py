"""
Alpha Builders
--------------
Cross-sectional alpha construction from density-model outputs (μ, σ).

Classes
-------
BaseAlphaBuilder
    Abstract ``compute(mu, sigma) -> alpha_series`` contract.
MuSigmaZScoreAlpha
    ``alpha = zscore(μ / σ)`` across assets at each date.

Functions
---------
build_alpha_builder
    Dispatch from a ``PanelAlphaConfig`` to a concrete builder.
"""

from __future__ import annotations

from density_model.portfolio.alpha.base import BaseAlphaBuilder
from density_model.portfolio.alpha.build import build_alpha_builder
from density_model.portfolio.alpha.mu_sigma_zscore import MuSigmaZScoreAlpha

__all__ = ["BaseAlphaBuilder", "MuSigmaZScoreAlpha", "build_alpha_builder"]
