"""
Panel Splitter Builder
----------------------
Dispatch from :class:`PanelSplitterConfig` to a concrete
:class:`BasePanelSplitter` instance.

Functions
---------
build_panel_splitter
    Construct a panel splitter from its validated configuration section.
"""

from __future__ import annotations

from dlecosys.shared.config.panel_schema import PanelSplitterConfig
from dlecosys.shared.data.panel.splitter import (
    BasePanelSplitter,
    PanelExpandingWindowSplitter,
    PanelHoldoutSplitter,
    PanelWalkForwardSplitter,
)

__all__ = ["build_panel_splitter"]


def build_panel_splitter(cfg: PanelSplitterConfig) -> BasePanelSplitter:
    """Return the splitter instance selected by ``cfg.type``."""

    if cfg.type == "holdout":
        return PanelHoldoutSplitter(val_fraction=cfg.val_fraction)
    if cfg.type == "expanding_window":
        return PanelExpandingWindowSplitter(n_splits=cfg.n_splits)
    if cfg.type == "walk_forward":
        return PanelWalkForwardSplitter(
            n_splits=cfg.n_splits,
            train_window_size=cfg.train_window_size,
        )
    raise ValueError(f"Unsupported panel splitter type {cfg.type!r}.")
