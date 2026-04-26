"""
Alpha Builder Dispatch
----------------------
Construct a concrete :class:`BaseAlphaBuilder` from a
:class:`PanelAlphaConfig`.

Functions
---------
build_alpha_builder
    Dispatch by ``cfg.type``.
"""

from __future__ import annotations

from density_model.portfolio.alpha.base import BaseAlphaBuilder
from density_model.portfolio.alpha.mu_sigma_zscore import MuSigmaZScoreAlpha
from density_model.portfolio.config import PanelAlphaConfig

__all__ = ["build_alpha_builder"]


def build_alpha_builder(cfg: PanelAlphaConfig) -> BaseAlphaBuilder:
    if cfg.type == "mu_sigma_zscore":
        return MuSigmaZScoreAlpha()
    raise ValueError(f"Unsupported alpha type {cfg.type!r}.")
